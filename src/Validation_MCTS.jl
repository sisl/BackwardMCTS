include("gridworldpomdp.jl")

using POMDPs
using POMDPPolicies: solve
using BeliefUpdaters: DiscreteBelief, NothingUpdater
using POMDPSimulators: stepthrough, updater
using QMDP
using ProgressBars

Tqdm(obj) = length(obj) == 1 ? obj : ProgressBars.tqdm(obj)

function optimal_action_idx(Γ, b)
    ϵ = 1e-10
    intrv = 1:length(Γ)
    utilities = dot.(Γ, Ref(b))
    opt_acts = map(i -> abs(maximum(utilities) - utilities[i] < ϵ), intrv)
    return rand((intrv)[opt_acts])
end

function run_fwd_simulation_sao(pomdp, b0, des_ao_traj, max_t; verbose=false)
    simulated_s = []
    simulated_ao = []
    simulated_final_s = nothing
    policy = DefinedPolicy(des_ao_traj)
    for (s,sp,a,o,r) in stepthrough(pomdp, policy, updater(policy), b0, rand(b0), "s,sp,a,o,r", max_steps=max_t)
        if verbose
            println("In state $s")
            println("took action $a")
            println("received observation $o and reward $r")
        end
        push!(simulated_s, s)
        push!(simulated_ao, (a,o))
        simulated_final_s = sp
    end
    return push!(simulated_s, simulated_final_s), simulated_ao
end

check_ao_trajs(sim_ao, des_ao_traj, lower_bound) = !lower_bound ? sim_ao==des_ao_traj : all(getindex.(sim_ao, Ref(1)) .== getindex.(des_ao_traj, Ref(1)))

function batch_fwd_simulations(pomdp, epochs, des_final_state, b0_testing, des_ao_traj; lower_bound=false, verbose=false, max_t = length(des_ao_traj))
    init_states = []
    b0 = DiscreteBelief(pomdp, states(pomdp), b0_testing)

    for e = Tqdm(1:epochs)
        sim_s, sim_ao = run_fwd_simulation_sao(pomdp, b0, des_ao_traj, max_t; verbose=verbose)
        if length(sim_s)==max_t+1 && sim_s[end]==des_final_state && sim_s[1]!=des_final_state && check_ao_trajs(sim_ao, des_ao_traj, lower_bound)
            push!(init_states, sim_s[1])
        end
    end
    percentage = round(length(init_states) / epochs * 100; digits=3)
    if verbose println("Found $(length(init_states))/$epochs ($(percentage)%) corresponding init states.") end
    return init_states, round(percentage/100; digits=5)
end

function parse_batch_fwd_simulations(pomdp, init_states)
    no_of_states = length(states(pomdp))
    S = zeros(no_of_states)
    for item in Tqdm(stateindex.(Ref(pomdp), init_states))
        S[item] += 1
    end
    return S ./ sum(S)
end

function convert_des_ao_traj(pomdp, des_ao_traj)
    return [(item[1], (states(pomdp))[item[2]]) for item in des_ao_traj if item[1] != :end]
end


mutable struct DefinedPolicy <: Policy
    action_sequence::AbstractVector
    current_index::Int
end

# The constructor below should be used to create the policy so that the action space is initialized correctly
DefinedPolicy(action_sequence::AbstractVector) = DefinedPolicy(action_sequence, 0)

## policy execution ##
function POMDPs.action(policy::DefinedPolicy, s)
    policy.current_index += 1
    return policy.action_sequence[policy.current_index][1]  # remove the [1] if `action_sequence` is a flat vector.
end

POMDPs.updater(::DefinedPolicy) = NothingUpdater()


function validation_probs_and_scores(β_levels, pomdp, max_t, des_final_state, CMD_ARGS; lower_bound=false, verbose=false)
    probs = []
    scores = []
    items = length(β_levels[max_t].ao)

    tab_pomdp = tabulate(pomdp)
    acts = collect(actions(pomdp))

    for i in 1:items
        bel  = β_levels[max_t].β[i]
        aos  = β_levels[max_t].ao[i]
    
        prob = bayesian_prob(tab_pomdp, acts, bel, aos)
        # prob = bayesian_prob_summed(tab_pomdp, acts, bel, aos)
        _, score = batch_fwd_simulations(pomdp, CMD_ARGS[:epochs], des_final_state, bel, convert_des_ao_traj(pomdp, aos), lower_bound=lower_bound, verbose=verbose);

        println("  Item:\t\t  $(i) of $(items) \n  Approx Prob:\t  $(prob) \n  Lhood Score:\t  $(score)")
        push!(probs, prob)
        push!(scores, score)
    end
    return probs, scores
end

function bayesian_next_belief(tab_pomdp, o, a, b)
    bp = tab_pomdp.O[o, a, :] .* (create_T_bar(tab_pomdp, a) * reshape(b, :, 1))
    return normalize(vec(bp))
end

function bayesian_prob(tab_pomdp, acts, bel, aos)
    prob = 1.0
    bp = bel
    for (a_sym, o) in aos[1:end-1]
        a = findfirst(x->x==a_sym, acts)
        prob *= branch_weight(tab_pomdp, o, a, bp)
        bp = bayesian_next_belief(tab_pomdp, o, a, bp)
    end
    return round(prob; digits=5)
end

function bayesian_prob_summed(tab_pomdp, acts, bel, aos)
    prob = 1.0
    for (a_sym, o) in aos[1:end-1]
        a = findfirst(x->x==a_sym, acts)
        bp = bayesian_next_belief(tab_pomdp, o, a, bel)
        prob *= branch_weight_summed(tab_pomdp, o, a, bel, bp)
        bel = bp
    end
    return round(prob; digits=5)
end

function branch_weight_summed(tab_pomdp, o, a, b, bp)
    ## Compute p(bp|b,a) = sum_o p(bp|b,a,o) p(o|b,a)
    Oa = tab_pomdp.O[o,a,:]
    T = create_T_bar(tab_pomdp, a)

    res = 0
    for o = 1:size(T, 1)
        ddirac = (bp == bayesian_next_belief(tab_pomdp, o, a, b))
        res += Int(ddirac) * branch_weight(tab_pomdp, o, a, b)
        if ddirac @show o end
    end
    @show "returned"
    return res
end

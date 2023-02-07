include("utils.jl")
include("gridworldpomdp.jl")

using POMDPs
using POMDPPolicies: solve
using BeliefUpdaters: DiscreteBelief, NothingUpdater
using POMDPSimulators: stepthrough, updater
using QMDP
using ProgressBars

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

function convert_des_ao_traj(pomdp, des_ao_traj)
    return [(item[1], (states(pomdp))[item[2]]) for item in des_ao_traj if item[1] != :end]
end

function convert_aos(pomdp, aos)
    len_items = Int(length(aos) / 2 - 1)
    result = []

    for idx in 1:len_items
        i = (idx-1)*2 + 1
        a_sym, o = aos[i: i+1]
        push!(result, (a_sym, states(pomdp)[o]))
    end

    return result
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

function validation_probs_and_scores_UCT(TREE, pomdp, max_t, des_final_state, CMD_ARGS; lower_bound=false, verbose=false)
    probs = []
    scores = []
    items = length(TREE.P)

    tab_pomdp = tabulate(pomdp)
    acts = collect(actions(pomdp))

    for (i, belRec) in enumerate(keys(TREE.P))
        bel, aos = belRec.β, belRec.ao
        p = TREE.P[belRec]
    
        prob = bayesian_prob(tab_pomdp, acts, bel, aos)
        # prob = bayesian_prob_summed(tab_pomdp, acts, bel, aos)
        _, score = batch_fwd_simulations(pomdp, CMD_ARGS[:epochs], des_final_state, bel, convert_aos(pomdp, aos), lower_bound=lower_bound, verbose=verbose);

        println("  Item:\t\t  $(i) of $(items) \n  TREE Value:\t  $(p) \n  Approx Prob:\t  $(prob) \n  Lhood Score:\t  $(score) \n  aos:\t  $(aos)")
        push!(probs, prob)
        push!(scores, score)
    end
    return probs, scores
end

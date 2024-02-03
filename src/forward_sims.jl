include("utils.jl")

using POMDPs
using BeliefUpdaters: DiscreteBelief, NothingUpdater
using POMDPSimulators: stepthrough
using QMDP: QMDPSolver
using ProgressBars

mutable struct DefinedPolicy <: Policy
    action_sequence::AbstractVector
    current_index::Int
end

# The constructor below should be used to create the policy so that the action space is initialized correctly
DefinedPolicy(action_sequence::AbstractVector) = DefinedPolicy(action_sequence, 0)

## policy execution ##
function POMDPs.action(policy::DefinedPolicy, b)
    policy.current_index += 1
    return policy.action_sequence[policy.current_index][1]  # remove the [1] if `action_sequence` is a flat vector.
end

POMDPs.updater(::DefinedPolicy) = NothingUpdater()

function convert_des_ao_traj(pomdp, des_ao_traj)
    return [(item[1], (observations(pomdp))[item[2]]) for item in des_ao_traj if item[1] != :end]
end

function convert_aos(pomdp, aos)
    len_items = Int(length(aos) / 2 - 1)
    result = []

    for idx in 1:len_items
        i = (idx-1)*2 + 1
        a_sym, o = aos[i: i+1]
        push!(result, (a_sym, observations(pomdp)[o]))
    end

    return result
end

function run_fwd_simulation_sao(RNG, pomdp, b0, des_ao_traj, max_t; verbose=false, custom_policy=nothing)
    simulated_s = []
    simulated_ao = []
    simulated_final_s = nothing
    
    policy = isnothing(custom_policy) ? DefinedPolicy(des_ao_traj) : custom_policy
    # global PM = tab_pomdp
    # global POL = policy
    # global B0 = b0
    # # tab_pomdp = PM; policy = POL; b0 = B0;
    for (s,sp,a,o,r) in stepthrough(pomdp, policy, POMDPs.updater(policy), b0, rand(RNG, b0), "s,sp,a,o,r", rng=RNG, max_steps=max_t)
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

check_ao_trajs(sim_ao, des_ao_traj, upper_bound) = !upper_bound ? sim_ao==des_ao_traj : all(getindex.(sim_ao, Ref(1)) .== getindex.(des_ao_traj, Ref(1)))

function batch_fwd_simulations(RNG, pomdp, epochs, des_final_state, b0_testing, des_ao_traj; upper_bound=false, verbose=false, custom_policy=nothing, max_t = length(des_ao_traj))
    isempty(des_ao_traj) && return nothing, 1.0
    
    init_states = []
    b0 = DiscreteBelief(pomdp, states(pomdp), b0_testing)

    # for e = Tqdm(1:epochs)
    for e = 1:epochs
        sim_s, sim_ao = run_fwd_simulation_sao(RNG, pomdp, b0, des_ao_traj, max_t; verbose=verbose, custom_policy=custom_policy)
        if length(sim_s)==max_t+1 && sim_s[end]==des_final_state && check_ao_trajs(sim_ao, des_ao_traj, upper_bound)  # && sim_s[1]!=des_final_state
            push!(init_states, sim_s[1])
        end
    end
    percentage = round(length(init_states) / epochs * 100; digits=3)
    if verbose println("Found $(length(init_states))/$epochs ($(percentage)%) corresponding init states.") end
    return init_states, round(percentage/100; digits=5)
end

function validation_probs_and_scores_UCT(TREE, pomdp, tab_pomdp, actions_pomdp, max_t, des_final_state, CMD_ARGS; upper_bound=false, verbose=false, custom_policy=nothing)
    items = collect(keys(TREE.P))

    probs  = zeros(length(items))
    scores = zeros(length(items))
    tsteps = zeros(length(items))

    acts = collect(actions_pomdp)
        
    @info "Using $(Threads.nthreads()) threads.\nBackwardsTree has $(length(items)) nodes."
    Threads.@threads for i = Tqdm(1:length(items)) # (i, belRec) in enumerate(keys(TREE.P))
        m_RNG = MersenneTwister(CMD_ARGS[:noise_seed] + i)

        belRec = items[i]
        bel, aos = belRec.β, belRec.ao
        p = TREE.P[belRec]
    
        # bp, prob = bayesian_prob(tab_pomdp, acts, bel, aos, :debug); nonzero(bp)
        prob = !upper_bound ? bayesian_prob(tab_pomdp, acts, bel, aos) : -1
        _, score = batch_fwd_simulations(m_RNG, pomdp, CMD_ARGS[:val_epochs], des_final_state, bel, convert_aos(pomdp, aos), upper_bound=upper_bound, verbose=verbose, custom_policy=custom_policy);

        if verbose
            println("  Item:\t\t  $(i) of $(items) \n  TREE Value:\t  $(p) \n  Approx Prob:\t  $(prob) \n  Lhood Score:\t  $(score) \n  aos:\t  $(aos)")
        end

        probs[i] = prob
        scores[i] = score
        tsteps[i] = depth(aos)
    end
    return probs, scores, tsteps
end

function stats(probs, scores, tsteps; use_probs=false)
    println("MAE: $(mae(probs-scores))")
    println("STE: $(ste(probs-scores))")

    res = Dict(t => [] for t in unique(tsteps))

    for (p,s,t) in zip(probs, scores, tsteps)
        v = use_probs ? p : s
        push!(res[t], v)
    end

    for t in sort(unique(tsteps))
        a,b,c,d = round.([minimum(res[t]), median(res[t]), mean(res[t]), maximum(res[t])].*100 ; digits=6)  # percent
        l = length(res[t])
        println("Timestep $(Int(t)) has $l items: (min, median, mean, max) = ($a%, $b%, $c%, $d%)")
    end
end

function get_tree_nodes(TREE)
    # Get all nodes in the BackwardTree.
    items = collect(keys(TREE.P))
    return Set([belRec.β for belRec in items])
end

function shrink_tree_nodes!(TREE; depths=[2])
    # Shrink BackwardTree s.t. only nodes at a certain depths exists.
    items = collect(keys(TREE.P))
    foo = [pop!(TREE.P, belRec) for belRec in items if !(depth(belRec.ao) in depths)]
    return
end
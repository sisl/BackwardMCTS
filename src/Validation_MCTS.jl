include("gridworldpomdp.jl")

using POMDPs
using POMDPPolicies: solve
using BeliefUpdaters: DiscreteBelief
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

function get_concentrated_rand_init_belief(pomdp)
    no_of_states = length(states(pomdp))
    b0 = zeros(no_of_states)
    i = rand(1:no_of_states-1)
    b0[i] = 1.0
    return DiscreteBelief(pomdp, states(pomdp), b0)
end

function run_fwd_simulation(pomdp, policy, b0, max_t; verbose=false)
    beliefs = []
    states = []
    observations = []
    actions = []
    rewards = []
    for (s,a,o,r,b) in stepthrough(pomdp, policy, updater(policy), b0, rand(b0), "s,a,o,r,b", max_steps=max_t)
        if verbose
            println("In state $s")
            println("took action $a")
            println("received observation $o and reward $r")
            println("and belief was $(b.b)")
        end
        push!(states, s)
        push!(actions, a)
        push!(observations, o)
        push!(rewards, r)
        push!(beliefs, b.b)
    end
    return states, beliefs, observations, actions, rewards
end

function run_fwd_simulation_sao(pomdp, policy, b0, max_t; verbose=false)
    simulated_s = []
    simulated_ao = []
    simulated_final_s = nothing
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

function batch_fwd_simulations(pomdp, policy, max_t, epochs, des_final_state, b0_testing, des_ao_traj; verbose=false)
    init_states = []
    b0 = DiscreteBelief(pomdp, states(pomdp), b0_testing)
    for e = Tqdm(1:epochs)
        sim_s, sim_ao = run_fwd_simulation_sao(pomdp, policy, b0, max_t; verbose=verbose)
        if length(sim_s)==max_t+1 && sim_s[end]==des_final_state && sim_s[1]!=des_final_state && sim_ao==des_ao_traj
            push!(init_states, sim_s[1])
        end
    end
    percentage = round(length(init_states) / epochs * 100; digits=3)
    println("Found $(length(init_states))/$epochs ($(percentage)%) corresponding init states.")
    return init_states
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
    return [(item[1], (states(pomdp))[item[2]]) for item in des_ao_traj]
end
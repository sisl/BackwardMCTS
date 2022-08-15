include("gridworldpomdp.jl")

using POMDPs
using POMDPPolicies: solve
using BeliefUpdaters: DiscreteBelief
using POMDPSimulators: stepthrough, updater
using QMDP
using ProgressBars

# Create pomdp
pomdp = SimpleGridWorldPOMDP(size=(4,4),
                            rewards=Dict(GWPos(2,3)=>-10.0, GWPos(3,1)=>+25.0)
                            ,
                            tprob = 0.7,
                            oprob = 1.0)

# Create α-vectors
solver = QMDPSolver()
policy = solve(solver, pomdp)
Γ = policy.alphas

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

function batch_fwd_simulations(pomdp, policy, max_t, des_final_state, epochs; verbose=false)
    init_states = []
    for e = tqdm(1:epochs)
        while true
            b0 = get_concentrated_rand_init_belief(pomdp)
            ss, _, _, _, _ = run_fwd_simulation(pomdp, policy, b0, max_t; verbose=verbose)
            if ss[end] == des_final_state && ss[1] != des_final_state
                push!(init_states, ss[1])
                break
            end
        end
    end
    println("Found $(length(init_states)) corresponding init states.")
    return init_states
end

function parse_batch_fwd_simulations(pomdp, init_states)
    no_of_states = length(states(pomdp))
    S = zeros(no_of_states)
    for item in tqdm(stateindex.(Ref(pomdp), init_states))
        S[item] += 1
    end
    return S ./ sum(S)
end

max_t = 4
epochs = 1_000_000
des_final_state = GWPos(3,1)
init_states = batch_fwd_simulations(pomdp, policy, max_t, des_final_state, epochs);

parsed_init_states = parse_batch_fwd_simulations(pomdp, init_states)
reshape_GW(parsed_init_states)
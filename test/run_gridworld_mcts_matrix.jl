include("gridworldpomdp.jl")
include("GridWorld_MCTS_matrix.jl")

using JuMP, Gurobi
using POMDPPolicies: solve
using QMDP

# Params
z_val = 0.5
LP_Solver = LP_Solver_config(Gurobi.Optimizer, z_val)

# Create pomdp
pomdp = SimpleGridWorldPOMDP(size=(6,6),
                            rewards=Dict(GWPos(2,3)=>-10.0, GWPos(3,1)=>+25.0)
                            ,
                            tprob = 0.9,
                            oprob = 0.9)

tab_pomdp = tabulate(pomdp)
no_of_actions = length(actions(pomdp))
no_of_states = length(states(pomdp))

# Create α-vectors
solver = QMDPSolver()
policy = solve(solver, tab_pomdp)
Γ = policy.alphas

# Create leaf belief
β_final = zeros(no_of_states,)
β_final[3] = 1.0

max_t = 4
β_levels = backwards_MCTS(pomdp, policy, β_final, max_t, LP_Solver)

k = 1;
top_bels, top_probs, top_aos = top_likely_init_beliefs(β_levels, max_t, k)
@show max_t
@show top_probs[end]
# @show top_bels[end]
# @show top_aos[end]

b0_testing = top_bels[end]
des_ao_traj = top_aos[end]

@show b0_testing
@show des_ao_traj
reshape_GW(top_bels[end])
include("gridworldpomdp.jl")
include("GridWorld_MCTS_matrix.jl")

using JuMP, Gurobi
using POMDPPolicies: solve
using QMDP

# Params
LP_Solver = Gurobi.Optimizer

# Create pomdp
pomdp = SimpleGridWorldPOMDP(size=(9,9),
                            rewards=Dict(GWPos(2,3)=>-10.0, GWPos(5,5)=>+25.0)
                            ,
                            tprob = 1.0,
                            oprob = 1.0)

tab_pomdp = tabulate(pomdp)
no_of_actions = length(actions(pomdp))
no_of_states = length(states(pomdp))

# Create α-vectors
solver = QMDPSolver()
policy = solve(solver, tab_pomdp)
Γ = policy.alphas

# Create leaf belief
β_final = zeros(no_of_states,)
centr = 9*4 + 5
β_final[centr] = 1.0

max_t = 4
β_levels = backwards_MCTS(pomdp, policy, β_final, max_t, LP_Solver)

lvl = 1;
init_states = root_belief(β_levels, lvl; normalize_to_1 = false);
reshape_GW(init_states)


# INFO:
# 0. Use matrix form of the same problem
# 1. Get vertices
# 2. Sample from belief space
# 3. Branch out to these sampled beliefs
"""
This script formulates the LP the same way as "test_gridworld_lp.jl" but
explicitly creates the A,b,c matrix and vectors.
It does so by transforming the original problem to an equality form.
See Section 11.1.3 of M. Kochenderfer, Algorithms for Optimization.
"""

include("gridworldpomdp.jl")
include("GridWorld_LP_matrix.jl")

using JuMP, Gurobi
using POMDPPolicies: solve
using QMDP

# Params
LP_Solver = Gurobi.Optimizer

# Create pomdp
pomdp = SimpleGridWorldPOMDP(size=(4,4),
                            rewards=Dict(GWPos(2,3)=>-10.0, GWPos(3,1)=>+25.0)
                            ,
                            tprob = 0.99,
                            oprob = 1.0)

tab_pomdp = tabulate(pomdp)
no_of_actions = length(actions(pomdp))
no_of_states = length(states(pomdp))

# Create α-vectors
solver = QMDPSolver()
policy = solve(solver, tab_pomdp)
Γ = policy.alphas

# Create leaf belief
β_t = zeros(no_of_states,)
# β_t[4] = 0.45
# β_t[5] = 0.55
# obs_id = 5
# a_star = αj = 4

β_t[3] = 1.0
obs_id = 3
a_star = αj = 2

# Solve for root belief
O_bar = O = create_O_bar(tab_pomdp, obs_id)
T_bar = T = create_T_bar(tab_pomdp, a_star)

β_opt, J_opt, A, b, c = validate(O_bar, T_bar, Γ, a_star, β_t, LP_Solver)

β = reshape_GW(β_opt[1:no_of_states])

# β_t_hat = normalize(O_bar * T_bar * β_opt);

# β_opt_best = validate_all_actions(tab_pomdp, obs_id, policy, β_t, LP_Solver)
# @show β_opt_best
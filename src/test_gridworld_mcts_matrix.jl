include("gridworldpomdp.jl")
include("GridWorld_MCTS_matrix.jl")

using JuMP, Gurobi
using POMDPPolicies: solve
using QMDP

# Params
LP_Solver = Gurobi.Optimizer

# Create pomdp
pomdp = SimpleGridWorldPOMDP(size=(6,6),
                            rewards=Dict(GWPos(2,3)=>-10.0, GWPos(3,1)=>+25.0)
                            ,
                            tprob = 0.90,
                            oprob = 0.90)

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

max_t = 6
β_levels = backwards_MCTS(pomdp, policy, β_final, max_t, LP_Solver)

# lvl = 1;
# top_bels, top_probs = top_likely_init_belief(β_levels, lvl);
# @show top_prob
# reshape_GW(top_bel)

k = 1;
top_bels, top_probs, top_aos = top_likely_init_beliefs(β_levels, max_t, k)
@show max_t
@show top_probs[end]
@show top_bels[end]
@show top_aos[end]
reshape_GW(top_bels[end])


# lvl = 1;
# init_states = root_belief(β_levels, lvl; normalize_to_1 = true);
# reshape_GW(init_states)

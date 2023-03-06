include("utils.jl")
include("gridworldpomdp.jl")
include("GridWorld_MCTS_matrix.jl")
include("Validation_MCTS.jl")

using Gurobi
using QMDP
using POMDPPolicies: solve

########## Params ##########
CMD_ARGS = parse_commandline()
@show_args CMD_ARGS


# Create pomdp
pomdp = SimpleGridWorldPOMDP(size=(CMD_ARGS[:gridsize], CMD_ARGS[:gridsize]),
                            rewards=Dict(GWPos(2,3)=>-10.0, GWPos(3,1)=>+25.0)
                            ,
                            tprob = CMD_ARGS[:t_and_o_prob],
                            oprob = CMD_ARGS[:t_and_o_prob])

tab_pomdp = tabulate(pomdp)
no_of_actions = length(actions(pomdp))
no_of_states = length(states(pomdp))

# Create α-vectors
solver = QMDPSolver()
policy = solve(solver, tab_pomdp)
Γ = policy.alphas

# Create leaf belief
β_final = zeros(no_of_states,)
des_final_state = GWPos(3,1)
β_final[3] = 1.0

# Create BMCTS
max_t = CMD_ARGS[:timesteps]
LP_Solver = LP_Solver_config(Gurobi.Optimizer, CMD_ARGS[:z_threshold])
β_levels = backwards_MCTS(pomdp, policy, β_final, max_t, LP_Solver, CMD_ARGS[:obs_N], CMD_ARGS[:belief_N])

# Validate BMCTS nodes
probs, scores = validation_probs_and_scores(β_levels, pomdp, max_t, des_final_state, CMD_ARGS, lower_bound=false)

# Dump results to file
csvdump(probs, scores, CMD_ARGS)

@info "Done!"
@info "Using $(Threads.nthreads()) threads."

include("utils.jl")
include("gridworldpomdp.jl")
include("GridWorld_MCTS_matrix.jl")
include("Validation_MCTS.jl")

using Gurobi
using QMDP
using POMDPPolicies: solve
using Random

########## Params ##########
CMD_ARGS = parse_commandline()
@show_args CMD_ARGS
Random.seed!(CMD_ARGS[:noise_seed])

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
max_t = CMD_ARGS[:max_timesteps]
LP_Solver = LP_Solver_config(Gurobi.Optimizer, zDistribution_exp(exp_const=CMD_ARGS[:z_dist_exp_const]))
TREE = search!(pomdp, policy, β_final, max_t, LP_Solver, CMD_ARGS[:no_of_simulations], CMD_ARGS[:exploration_const], CMD_ARGS[:rollout_random])

# Save tree to local disk
saveTree(TREE, CMD_ARGS[:savename])

# Validate BMCTS nodes
probs, scores, tsteps = validation_probs_and_scores_UCT(TREE, pomdp, max_t, des_final_state, CMD_ARGS, lower_bound=false)

# Dump results to file
csvdump(probs, scores, tsteps, CMD_ARGS)

@info "Done!"
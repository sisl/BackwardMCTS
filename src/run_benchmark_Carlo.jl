@info "Detected $(Threads.nthreads()) threads."

include("utils.jl")
include("construct_tree.jl")
include("forward_sims.jl")
include("./Carlo/likelihood_learning.jl")

using Gurobi
using QMDP
using POMDPPolicies: solve
using Random

########## Params ##########
CMD_ARGS = parse_commandline()
@show_args CMD_ARGS
Random.seed!(CMD_ARGS[:noise_seed])

# Create pomdp
pomdp = CarloPOMDP()
actions_pomdp = actions(pomdp)
tab_pomdp = tabulate(pomdp; dir="./Carlo/")
no_of_actions = length(actions(pomdp))
no_of_states = length(states(pomdp))

# Create α-vectors
solver = QMDPSolver()
policy = solve(solver, tab_pomdp)
Γ = policy.alphas

# Create leaf belief
final_state = CarloDiscreteState(6, 6, :within_limit, :straight)
β_final = get_leaf_belief(pomdp, final_state)

# Create BMCTS
max_t = CMD_ARGS[:max_timesteps]
LP_Solver = LP_Solver_config(Gurobi.Env(), zDistribution_exp(exp_const=CMD_ARGS[:z_dist_exp_const]))
TREE = search!(tab_pomdp, actions_pomdp, policy, β_final, max_t, LP_Solver, getd(CMD_ARGS, [:noise_seed, :sims_per_thread, :no_of_threads, :exploration_const, :rollout_random])...)

# Save tree to local disk
saveTree(pomdp, TREE, CMD_ARGS[:savename])

# Validate BMCTS nodes
probs1, scores1, tsteps1 = validation_probs_and_scores_UCT(TREE, pomdp, tab_pomdp, actions_pomdp, max_t, final_state, CMD_ARGS, upper_bound=false, verbose=false)
stats(probs1, scores1, tsteps1)

# Dump results to file
csvdump(probs1, scores1, tsteps1, CMD_ARGS)

@info "Done!"
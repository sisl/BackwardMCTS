@info "Detected $(Threads.nthreads()) threads."

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
Random.seed!(CMD_ARGS[:noise_seed])

# Create pomdp
pomdp = SimpleGridWorldPOMDP(size=(CMD_ARGS[:gridsize], CMD_ARGS[:gridsize]),
                            rewards=Dict(GWPos(2,3)=>-10.0, GWPos(3,1)=>+25.0)
                            ,
                            tprob = CMD_ARGS[:t_prob],
                            oprob = CMD_ARGS[:o_prob])

tab_pomdp = tabulate(pomdp)
actions_pomdp = actions(pomdp)
no_of_actions = length(actions(pomdp))
no_of_states = length(states(pomdp))

# *** 1 *** Initially solving with a bad solver.
# Create α-vectors
solver = QMDPSolver(max_iterations=2, belres=0.1);   #! deliberately bad solver
policy = solve(solver, tab_pomdp);
Γ = policy.alphas;

# Create leaf belief
#! Deliberately aiming for the undesired final state
final_state = GWPos(2,3)
β_final = get_leaf_belief(pomdp, final_state)

# Create BMCTS
max_t = CMD_ARGS[:max_timesteps]
LP_Solver = LP_Solver_config(Gurobi.Env(), zDistribution_exp(exp_const=CMD_ARGS[:z_dist_exp_const]))
TREE = search!(tab_pomdp, actions_pomdp, policy, β_final, max_t, LP_Solver, getd(CMD_ARGS, [:noise_seed, :sims_per_thread, :no_of_threads, :exploration_const, :rollout_random])...)

# Save tree to local disk
saveTree(pomdp, TREE, CMD_ARGS[:savename])


# *** 1 *** Initially solving with a bad solver.
# Create α-vectors
solver = QMDPSolver(max_iterations=2, belres=0.1)   #! deliberately bad solver
policy1 = solve(solver, pomdp)

# Validate BMCTS nodes, with policy 1.
#* Probability of reaching β_final with the bad policy.
#! Using upper bound:  p(b'|b,a) ≥ p(b',o|b,a).
probs1, scores1, tsteps1 = validation_probs_and_scores_UCT(TREE, pomdp, tab_pomdp, actions_pomdp, max_t, final_state, CMD_ARGS, upper_bound=true)   #! deliberately upper_bound=true


#------------------------------------------------------#

# # *** 2 *** Using a better solver, but not specifically trained on the beliefs in the tree.
# Create better α-vectors
solver = QMDPSolver()   #! default params
policy2 = solve(solver, pomdp)

# Validate BMCTS nodes, with policy 2.
#* Probability of reaching β_final with this new policy.
#! Using upper bound:  p(b'|b,a) ≥ p(b',o|b,a).
probs2, scores2, tsteps2 = validation_probs_and_scores_UCT(TREE, pomdp, tab_pomdp, actions_pomdp, max_t, final_state, CMD_ARGS, upper_bound=true, custom_policy=policy2)   #! deliberately upper_bound=true

# #? You should see that scores2 ≤ scores1 for all items, because our policy has improved 
# #? and our failure probability is lower now. 


#------------------------------------------------------#

# # *** 3 *** Specifically training on belief nodes with PBVI
include("pbvi.jl")
solver = PBVISolver(max_iterations=5, verbose=true)
policy3 = solve(solver, pomdp, get_tree_nodes(TREE))

# Validate BMCTS nodes, with policy 3.
#* Probability of reaching β_final with this new policy.
#! Using upper bound:  p(b'|b,a) ≥ p(b',o|b,a).
probs3, scores3, tsteps3 = validation_probs_and_scores_UCT(TREE, pomdp, tab_pomdp, actions_pomdp, max_t, final_state, CMD_ARGS, upper_bound=true, custom_policy=policy3)   #! deliberately upper_bound=true

# #? You should see that scores3 ≤ scores2 for all items, because our policy has improved 
# #? and our failure probability is lower now. 


@info "Done!"
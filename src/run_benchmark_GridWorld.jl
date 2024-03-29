using BackwardMCTS

@info "Detected $(Threads.nthreads()) threads."

########## Params ##########
CMD_ARGS = parse_commandline()
show_args(CMD_ARGS)
seed!(CMD_ARGS[:noise_seed])
RNG = MersenneTwister(CMD_ARGS[:noise_seed])
final_state = GWPos(4,2)

# Create pomdp
pomdp = SimpleGridWorldPOMDP(size=(CMD_ARGS[:gridsize], CMD_ARGS[:gridsize]),
                            rewards=Dict(GWPos(4,2)=>+30.0, 
                                         GWPos(6,7)=>+20.0,
                                         GWPos(2,5)=>-25.0,
                                         GWPos(7,4)=>-15.0),
                            tprob = CMD_ARGS[:t_prob],
                            oprob = CMD_ARGS[:o_prob])

tab_pomdp = tabulate(pomdp)
actions_pomdp = actions(pomdp)
no_of_actions = length(actions(pomdp))
no_of_states = length(states(pomdp))

# Create α-vectors
solver = QMDPSolver();
policy = solve(solver, tab_pomdp);
Γ = policy.alphas;

# Create leaf belief
β_final = get_leaf_belief(pomdp, final_state)

# Create BMCTS
max_t = CMD_ARGS[:max_timesteps]
LP_Solver = LP_Solver_config(Env(), zDistribution_exp(exp_const=CMD_ARGS[:z_dist_exp_const]))
TREE = search!(tab_pomdp, actions_pomdp, policy, β_final, max_t, LP_Solver, getd(CMD_ARGS, [:noise_seed, :sims_per_thread, :threads, :exploration_const, :rollout_random])...)

# Save tree to local disk
saveTree(pomdp, TREE, CMD_ARGS[:savename])

# Validate BMCTS nodes
probs1, scores1, tsteps1 = validation_probs_and_scores_UCT(TREE, pomdp, tab_pomdp, actions_pomdp, max_t, final_state, CMD_ARGS, upper_bound=false, verbose=false)
stats(probs1, scores1, tsteps1)

# Dump results to file
csvdump(probs1, scores1, tsteps1, CMD_ARGS)

# Construct and benchmark Voronoi kdtree
# include("kdtree.jl")
# kdtree = create_kdtree(TREE)
# tree_probs, bayes_probs, kd_scores = benchmark_kdtree(kdtree, pomdp, final_state; sigma=0.1, upper_bound=false)

@info "Done!"
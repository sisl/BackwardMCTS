module BackwardMCTS

@info "Loading module:\nBackward Monte Carlo Tree Search (BMCTS)"

include("argparse_utils.jl")
export
    parse_commandline,
    show_args


include("construct_tree.jl")
export
    BackwardTree,
    depth,
    UCB1,
    UCT_action,
    sample_obs,
    sample_node,
    simulate_node!,
    rollout,
    merge_trees!,
    get_branch_actobs,
    branch_weight,
    bayesian_next_belief,
    search!,
    bayesian_prob,
    fetch_nodes

        
include("forward_sims.jl")
export
    QMDPSolver,
    DefinedPolicy,
    run_fwd_simulation_sao,
    check_ao_trajs,
    batch_fwd_simulations,
    validation_probs_and_scores_UCT,
    stats


include("kdtree.jl")
export
    KDTree_and_probs,
    create_kdtree,
    benchmark_kdtree,
    benchmark_kdtree_diameter
    

include("operate_lp.jl")
export
    LinearProgram,
    remove_redundant_col,
    extract_vertex,
    edge_transitions!,
    find_bases,
    get_polygon_vertices,
    get_valid_partition


include("pbvi.jl")
export
    solve
    

include("solve_lp.jl")
export
    Env,
    obj_func,
    validate,
    get_z_high,
    validate_single_action,
    validate_rollout_actions,
    sample_from_belief_subspace,
    remove_polygon_vertices!


include("utils.jl")
export
    seed!,
    MersenneTwister,
    Tqdm,
    create_T_bar,
    create_O_bar,
    fix_overflow,
    average,
    flatten,
    flatten_twice,
    nonzero,
    weighted_column_sum,
    maxk,
    maxk,
    zeros_except,
    normalize,
    normalize,
    unique_elems,
    no_nan_division,
    csvdump,
    zDistribution_exp,
    ≂,
    softmax_neg,
    getRandomSamplesOnNSphere,
    getRandomSamplesInNSphere,
    getd,
    remove,
    L1_norm,
    L2_norm,
    isValidProb,
    issingular,
    saferank,
    saveTree,
    loadTree,
    BeliefRecord,
    CappedExponential,
    LP_Solver_config,
    ste,
    mae,
    pprint,
    get_optimal_actions_matrix


include("./GridWorld/gridworldpomdp.jl")
export
    GWPos,
    SimpleGridWorldPOMDP,
    tabulate,
    actions,
    states,
    observations,
    get_leaf_belief

end  # module

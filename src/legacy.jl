""" Depricated functions are here """

function scale_weights!(w)
    wu = unique(w)
    for item in wu
        elems = (w.==item)
        amount = sum(elems)
        w[elems] .= item/amount
    end
end

function get_branch_weights(tab_pomdp, obs, programs, belief_samples)
    BW = []
    for (lp, samples) in zip(programs, belief_samples)
        optimal_act = lp.a_star
        bw = map(bel -> branch_weight(tab_pomdp, obs, optimal_act, bel), samples)
        push!(BW, bw)
    end
    return BW
end

function get_branch_weights_v2(tab_pomdp, obs, obsW, programs, belief_samples)
    BW = []
    for (lp, samples) in zip(programs, belief_samples)
        optimal_act = lp.a_star
        bw = map(bel -> branch_weight(tab_pomdp, obs, optimal_act, bel), samples)
        push!(BW, obsW * bw)
    end
    return BW
end

function root_belief(β_levels, lvl; normalize_to_1=true)
    res = weighted_column_sum(β_levels[lvl].W, hcat(β_levels[lvl].β...));
    return normalize_to_1 ? normalize(res) : res
end

function top_likely_init_belief(β_levels, lvl)
    prob, elem = findmax(β_levels[lvl].W)
    bel = (β_levels[lvl].β)[elem]
    return bel, prob
end

function top_likely_init_beliefs(β_levels, lvl, k)
    if isempty(β_levels[lvl].W)
        @warn "Your `z_val` is too high, no solution found."
        return [nothing], [nothing], [nothing]
    end
    elems, probs =  maxk(β_levels[lvl].W, k)
    bels = (β_levels[lvl].β)[elems]
    aos = (β_levels[lvl].ao)[elems]
    return bels, probs, aos
end

function unique_elems_weights(S, w)
    uS = unique(S)
    dd = Dict(uS.=>zeros(length(uS)))

    for (idx, item) in enumerate(S)
        dd[item] += w[idx]
    end
    return collect(keys(dd)), collect(values(dd))
end

function bayesian_prob_summed(tab_pomdp, acts, bel, aos)
    prob = 1.0
    for (a_sym, o) in aos[1:end-1]
        a = findfirst(x->x==a_sym, acts)
        bp = bayesian_next_belief(tab_pomdp, o, a, bel)
        prob *= branch_weight_summed(tab_pomdp, o, a, bel, bp)
        bel = bp
    end
    return round(prob; digits=5)
end

function branch_weight_summed(tab_pomdp, o, a, b, bp)
    ## Compute p(bp|b,a) = sum_o p(bp|b,a,o) p(o|b,a)
    Oa = tab_pomdp.O[o,a,:]
    T = create_T_bar(tab_pomdp, a)

    res = 0
    for o = 1:size(T, 1)
        ddirac = (bp == bayesian_next_belief(tab_pomdp, o, a, b))
        res += Int(ddirac) * branch_weight(tab_pomdp, o, a, b)
        if ddirac @show o end
    end
    @show "returned"
    return res
end

function optimal_action_idx(Γ, b)
    ϵ = 1e-10
    intrv = 1:length(Γ)
    utilities = dot.(Γ, Ref(b))
    opt_acts = map(i -> abs(maximum(utilities) - utilities[i] < ϵ), intrv)
    return rand((intrv)[opt_acts])
end

function parse_batch_fwd_simulations(pomdp, init_states)
    no_of_states = length(states(pomdp))
    S = zeros(no_of_states)
    for item in Tqdm(stateindex.(Ref(pomdp), init_states))
        S[item] += 1
    end
    return S ./ sum(S)
end

function vertices_from_belief_subspace(LP)
    X_stars = []
    for B in LP.vertices
        x_star = extract_vertex(B, LP)[1:LP.no_of_states]
        fix_overflow!(x_star)
        push!(X_stars, normalize(x_star))
    end
    return X_stars
end

function zDistribution(z_min = 0.0, z_max = 1.0)
    # Outputs a distribution whose pdf is proportional to its input.
    # `z_max` is the upper bound to the z-value we know will not output any feasible solution to its corresponding LP.
    a = z_min
    b = c = z_max
    return TriangularDist(a, b, c)
end

""" LEGACY. To be used with `z_threshold` value. """
function validate_single_action_old(tab_pomdp, obs_id, policy, β_next, LP_Solver, αj)
    Γ = policy.alphas
    O = create_O_bar(tab_pomdp, obs_id)

    z_max = 1.0
    Vals = Dict()

    while z_max > LP_Solver.z_threshold
        z_val = rand(zDistribution(LP_Solver.z_threshold, z_max))
        X, J, A, b, c = @suppress validate(O, create_T_bar(tab_pomdp, policy.action_map[αj]), Γ, αj, β_next, LP_Solver.model, z_val)
        push!.(Ref(Vals), (:X, :J, :A, :b, :c).=>(X, J, A, b, c))

        # @show (αj, J)
        if !(J == Inf)
            break
            # @show z_val
        end
        z_max = z_val
    end

    if (Vals[:J] == Inf)
        return nothing
    end

    # @show keys(Vals)
    # @show Vals[:J]
    # @warn "aa"
    LP = LinearProgram(Vals[:A], Vals[:b], Vals[:c], Vals[:X], no_of_states, Set(), αj);
    B = get_valid_partition(Vals[:A], Vals[:X]);

    @suppress get_polygon_vertices!(B, LP);
    @suppress remove_polygon_vertices!(LP, Γ, αj);
    return LP
end

""" This function is no longer used. It is the tree search without UCT. """
function backwards_MCTS(pomdp, policy, β_final, max_t, LP_Solver, obs_N=1, belief_N=1)
    # obs_N    = 1
    # belief_N = 1

    tab_pomdp = tabulate(pomdp)
    actions_pomdp = actions(pomdp)

    β_levels = Dict()
    push!(β_levels, 0 => βts_and_history([β_final], [(:end, -1)]))

    for t = Tqdm(1:max_t)

        lvl = β_levels[t-1]
        β = []
        # W = []
        AO = []
        
        for l = Tqdm(1:length(lvl.ao))
            β_next = lvl.β[l]
            # W_next = lvl.W[l]
            AO_next = lvl.ao[l]

            # Sample possible observations (with weights)
            nonzero_weights, nonzero_states = nonzero(β_next)
            obs_weights = weighted_column_sum(nonzero_weights, tab_pomdp.O[:, end, nonzero_states])
            
            if t==1  # enforce fully-observable for final (sink) state
                obs_samples, obs_weights = maxk(obs_weights, 1)
            else
                obs_samples, obs_weights = maxk(obs_weights, obs_N)
            end

            ## This part is backwards in time (from leaf to root)
            # Get previous beliefs, given the sampled observation and optimal policy
            LPs = map(obs_id -> validate_all_actions(tab_pomdp, obs_id, policy, β_next, LP_Solver), obs_samples);
            S = map(item -> sample_from_belief_subspace.(item[2], Ref(tab_pomdp), Ref(obs_samples[item[1]]), Ref(belief_N)), enumerate(LPs));  # item := (idx, LP) 

            ## This part is forward in time (from root to leaf)
            # Compute weights for branches
            # Weights = get_branch_weights.(Ref(tab_pomdp), obs_samples, LPs, S)
            # Weights = get_branch_weights_v2.(Ref(tab_pomdp), obs_samples, obs_weights, LPs, S)
            ActObs = get_branch_actobs.(Ref(actions_pomdp), Ref(AO_next), obs_samples, LPs, S)

            S = flatten_twice(S);
            # Weights = flatten_twice(Weights);
            ActObs = flatten_twice(ActObs);

            append!(β, S)
            # append!(W, Weights * W_next)
            append!(AO, ActObs)
        end
        
        push!(β_levels, t => βts_and_history(β, AO))
    end
    return β_levels
end

function validate_all_actions(tab_pomdp, obs_id, policy, β_next, LP_Solver)
    Γ = policy.alphas
    O = create_O_bar(tab_pomdp, obs_id)

    res = @suppress map(αj->validate(O, create_T_bar(tab_pomdp, policy.action_map[αj]), Γ, αj, β_next, LP_Solver), 1:length(Γ))
    
    
    # J_min = minimum(getindex.(res, Ref(2)))   # index=2 is the obj value
    resRef2 = round.(getindex.(res, Ref(2)); digits=4)
    J_min = minimum(resRef2)   # index=2 is the obj value, rounded-off to 4 decimals 

    if J_min == Inf
        return []
    end

    a_star = resRef2 .== J_min
    a_star_idxs = (1:length(a_star))[a_star]

    X_inits = getindex.(res[a_star], Ref(1))    # index=1 is the x value

    A_matrices = getindex.(res[a_star], Ref(3))   # index=3 is the A matrix
    A_matrices = collect.(A_matrices);
    
    b_vectors = getindex.(res[a_star], Ref(4))   # index=4 is the b matrix
    b_vectors = collect.(b_vectors);

    c_vectors = getindex.(res[a_star], Ref(5))   # index=5 is the c matrix
    c_vectors = collect.(c_vectors);

    emptySets = [Set() for _ in 1:sum(a_star)]
    LPs = LinearProgram.(A_matrices, b_vectors, c_vectors, X_inits, Ref(no_of_states), emptySets, a_star_idxs);
    Bs = get_valid_partition.(A_matrices, X_inits);

    @suppress get_polygon_vertices!.(Bs, LPs);
    @suppress remove_polygon_vertices!.(LPs, Ref(Γ), a_star_idxs);
    return LPs
end

function validation_probs_and_scores(RNG, β_levels, pomdp, max_t, des_final_state, CMD_ARGS; upper_bound=false, verbose=false)
    probs = []
    scores = []
    items = length(β_levels[max_t].ao)

    tab_pomdp = tabulate(pomdp)
    acts = collect(actions(pomdp))

    for i in 1:items
        bel  = β_levels[max_t].β[i]
        aos  = β_levels[max_t].ao[i]
    
        prob = bayesian_prob(tab_pomdp, acts, bel, aos)
        # prob = bayesian_prob_summed(tab_pomdp, acts, bel, aos)
        _, score = batch_fwd_simulations(RNG, pomdp, CMD_ARGS[:val_epochs], des_final_state, bel, convert_des_ao_traj(pomdp, aos), upper_bound=upper_bound, verbose=verbose);

        println("  Item:\t\t  $(i) of $(items) \n  Approx Prob:\t  $(prob) \n  Lhood Score:\t  $(score)")
        push!(probs, prob)
        push!(scores, score)
    end
    return probs, scores
end
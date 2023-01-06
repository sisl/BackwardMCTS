include("utils.jl")
include("VertexPivot.jl")
include("GridWorld_LP_matrix.jl")

struct βts_and_history
    β
    ao
end

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
            S = map(item -> samples_from_belief_subspace.(item[2], Ref(tab_pomdp), Ref(obs_samples[item[1]]), Ref(belief_N)), enumerate(LPs));  # item := (idx, LP) 

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

function unique_elems(S)
    elems = unique(i -> S[i], 1:length(S))
    return S[elems], elems
end

function get_branch_actobs(actions_pomdp, AO_next, obs, programs, belief_samples)
    AO = []
    for (lp, samples) in zip(programs, belief_samples)
        optimal_act = actions_pomdp[lp.a_star]
        push!(AO, [vcat((optimal_act, obs), AO_next) for _ in 1:length(samples)])
    end
    return AO
end

function branch_weight(tab_pomdp, o, a, b)
    ## Compute p(o|a,b) = sum_s sum_s' O(o|a,s') T(s'|a,s) b(s)
    # Tb = create_T_bar(tab_pomdp, a) * reshape(b, :, 1)
    # return dot(tab_pomdp.O[o,a,:], Tb)
    Oa = tab_pomdp.O[o,a,:]
    T = create_T_bar(tab_pomdp, a)
    return dot(Oa' * T, b)
end

include("GridWorld_LP_matrix.jl")
include("VertexPivot.jl")

using ProgressBars

Tqdm(obj) = length(obj) == 1 ? obj : ProgressBars.tqdm(obj)

flatten(A) = collect(Iterators.flatten(A))
flatten_twice(A) = flatten(flatten(A))

function nonzero(A)
    idx = A .!= 0.0
    elems = 1:length(A)
    return A[idx], elems[idx]
end

function weighted_column_sum(weights, cols)
    res = (weights .* cols')'
    return vec(sum(res, dims=2))
end

function maxk(A, k)
    idx = partialsortperm(A, 1:k, rev=true)
    vals = A[idx]
    elems = (vals .> 0.0)
    return idx[elems], vals[elems]
end

struct βts_and_weights
    β
    W
    ao
end

function backwards_MCTS(pomdp, policy, β_final, max_t, LP_Solver, obs_N=1, belief_N=1)
    # obs_N    = 1
    # belief_N = 1

    tab_pomdp = tabulate(pomdp)
    actions_pomdp = actions(pomdp)

    β_levels = Dict()
    push!(β_levels, 0 => βts_and_weights([β_final], [1.0], [(:end, -1)]))

    for t = Tqdm(1:max_t)

        lvl = β_levels[t-1]
        β = []
        W = []
        AO = []
        
        for l = Tqdm(1:length(lvl.W))
            β_next = lvl.β[l]
            W_next = lvl.W[l]
            AO_next = lvl.ao[l]

            # Sample possible observations (with weights)
            nonzero_weights, nonzero_states = nonzero(β_next)
            obs_weights = weighted_column_sum(nonzero_weights, tab_pomdp.O[:, end, nonzero_states])
            obs_samples, _ = maxk(obs_weights, obs_N)

            # Get previous beliefs, given the sampled observation and optimal policy
            ## This part is backwards in time (from leaf to root)
            LPs = map(obs_id -> validate_all_actions(tab_pomdp, obs_id, policy, β_next, LP_Solver), obs_samples);
            # S = map(LP -> samples_from_belief_subspace.(LP, Ref(belief_N)), LPs);
            S = map(item -> samples_from_belief_subspace.(item[2], Ref(tab_pomdp), Ref(obs_samples[item[1]]), Ref(belief_N)), enumerate(LPs));  # item := (idx, LP) 


            # Compute weights for branches
            ## This part is forward in time (from root to leaf)
            Weights = get_branch_weights.(Ref(tab_pomdp), obs_samples, LPs, S)
            ActObs = get_branch_actobs.(Ref(actions_pomdp), Ref(AO_next), obs_samples, LPs, S)

            S = flatten_twice(S);
            Weights = flatten_twice(Weights);
            ActObs = flatten_twice(ActObs);

            append!(β, S)
            append!(W, Weights * W_next)
            append!(AO, ActObs)
        end
        
        push!(β_levels, t => βts_and_weights(β, W, AO))
    end
    return β_levels
end

function scale_weights!(w)
    wu = unique(w)
    for item in wu
        elems = (w.==item)
        amount = sum(elems)
        w[elems] .= item/amount
    end
end

function unique_elems(S)
    elems = unique(i -> S[i], 1:length(S))
    return S[elems], elems
end

function unique_elems_weights(S, w)
    uS = unique(S)
    dd = Dict(uS.=>zeros(length(uS)))

    for (idx, item) in enumerate(S)
        dd[item] += w[idx]
    end
    return collect(keys(dd)), collect(values(dd))
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
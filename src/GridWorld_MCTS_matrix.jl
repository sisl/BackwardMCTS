include("GridWorld_LP_matrix.jl")
include("VertexPivot.jl")

using ProgressBars

Tqdm(obj) = length(obj) == 1 ? obj : ProgressBars.tqdm(obj)

flatten(A) = collect(Iterators.flatten(A))

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
    return idx, vals
end

struct βts_and_weights
    βt
    W
end

function backwards_MCTS(pomdp, policy, β_final, max_t, LP_Solver)
    obs_N = 5
    belief_N = 5
    tab_pomdp = tabulate(pomdp)

    β_levels = Dict()
    push!(β_levels, max_t => βts_and_weights([β_final], [1.0]))

    for t = tqdm(max_t-1 :-1 :1)

        lvl = β_levels[t+1]
        β = []
        W = []
        
        for l = Tqdm(1:length(lvl.W))
            β_next = lvl.βt[l]
            W_next = lvl.W[l]

            # Sample possible observations (with weights)
            nonzero_weights, nonzero_states = nonzero(β_next)
            obs_weights = weighted_column_sum(nonzero_weights, tab_pomdp.O[:, end, nonzero_states])
            obs_samples, obs_samples_weights = maxk(obs_weights, obs_N)

            # Get previous beliefs, given the sampled observation and optimal policy
            LPs = map(obs_id -> validate_all_actions(tab_pomdp, obs_id, policy, β_next, LP_Solver), obs_samples);
            
            obs_repeat = map(id -> length(LPs[id]), 1:obs_N) .* belief_N
            obs_weights = vcat(fill.(obs_samples_weights, obs_repeat)...)
            
            LPs = flatten(LPs);
            S = sample_from_belief_subspace.(LPs, Ref(belief_N));
            S = flatten(S);

            # Backpropagate belief samples (as separate branches)
            w_prevs = W_next.*obs_weights 
            _, elems = nonzero(w_prevs)
            append!(β, S[elems])
            append!(W, w_prevs[elems])
        end
        
        push!(β_levels, t => βts_and_weights(β, W))
    end
    return β_levels
end

function root_belief(β_levels, lvl; normalize_to_1=true)
    res = weighted_column_sum(β_levels[lvl].W, hcat(β_levels[lvl].βt...));
    ϵ = 1e-10
    res[abs.(res) .< ϵ] .= 0.0
    return normalize_to_1 ? normalize(res) : res
end
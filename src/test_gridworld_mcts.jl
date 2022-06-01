using StatsBase

function nonzero(A)
    idx = A .!= 0
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

function backwards_MCTS(pomdp, policy, β_final, max_t)
    obs_N = 5
    tab_pomdp = tabulate(pomdp)

    β_levels = Dict()
    push!(β_levels, max_t => βts_and_weights([β_final], [1.0]))

    for t = max_t-1 :-1 :1

        lvl = β_levels[t+1]
        β = []
        W = []
        
        for l = 1:length(lvl.W)
            β_next = lvl.βt[l]
            W_next = lvl.W[l]

            nonzero_weights, nonzero_states = nonzero(β_next)
            obs_weights = weighted_column_sum(nonzero_weights, tab_pomdp.O[:, end, nonzero_states])
            obs_samples, obs_samples_weights = maxk(obs_weights, obs_N)

            b_prevs = map(obs_id -> validate_all_actions(tab_pomdp, obs_id, policy, β_next), obs_samples)
            w_prevs = W_next.*obs_samples_weights 

            append!(β, b_prevs)
            append!(W, w_prevs)
        end
        
        push!(β_levels, t => βts_and_weights(β, W))
    end
    return β_levels
end

# Create pomdp
pomdp = SimpleGridWorldPOMDP(size=(4,4),
                            rewards=Dict(GWPos(2,3)=>-10.0, GWPos(3,1)=>+25.0))
                            # ,
                            # tprob = 0.7,
                            # oprob = 0.7)

tab_pomdp = tabulate(pomdp)
no_of_actions = length(actions(pomdp))
no_of_states = length(states(pomdp))

# Create α-vectors
solver = QMDPSolver()
policy = solve(solver, tab_pomdp)
Γ = policy.alphas

# Create leaf belief
β_final = zeros(no_of_states,)
β_final[3] = 1.0

max_t = 4
β_levels = backwards_MCTS(pomdp, policy, β_final, max_t)

init_states = weighted_column_sum(β_levels[1].W, hcat(β_levels[1].βt...))
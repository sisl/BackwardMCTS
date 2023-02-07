include("utils.jl")
include("VertexPivot.jl")
include("GridWorld_LP_matrix.jl")

using DataStructures: DefaultDict
using StatsBase: sample, Weights
using Parameters: @with_kw

struct BeliefRecord
    β
    ao
end

@with_kw mutable struct BackwardTree
    AO = Set{Tuple}()                                                   # histories of (a,o,...)
    T  = Dict{Int, Set{Tuple}}()                                        # timestep -> corresponding values of AO
    
    β  = Dict{Tuple, Set{AbstractArray}}()                              # hist -> set of belief nodes
    Q  = DefaultDict{Tuple, Float64}(0.0)                               # hist -> Q-value
    N  = DefaultDict{Tuple, Int}(0.0)                                   # hist -> N-value

    P = Dict{BeliefRecord, Float64}()                                  # belief -> reachability probability
end

# Instantiate:
BackwardTree(max_t) = BackwardTree(T=Dict(i=>Set{Tuple}() for i=0:max_t))


# Get a list of items from a Dict or DefaultDict.
getd(d::Union{Dict, DefaultDict}, k::Union{Set, AbstractArray}) = getindex.(Ref(d), k)

depth(hist::Tuple) = length(hist)÷2 - 1   # integer division: ÷

UCB1(TREE, exploration_const, oh, aoh) = aoh in TREE.AO ? TREE.Q[aoh] + exploration_const*sqrt(log(TREE.N[oh]) / TREE.N[aoh]) : Inf

function UCT_action(TREE, exploration_const, actions_pomdp, obs, hist)
    vals = map(a -> UCB1(TREE, exploration_const, (obs, hist...), (a, obs, hist...)), actions_pomdp)
    max_vals = maximum(vals)
    return rand((1:length(vals))[vals.==max_vals])
end

function Base.push!(TREE::BackwardTree; hist::Tuple=(), with_time=true, belief::AbstractArray=[])
    if !isempty(hist)
        push!(TREE.AO, hist)
        if with_time
            push!(TREE.T[depth(hist)], hist)
        end
    end
    if !isempty(belief)
        hist in keys(TREE.β) ? nothing : TREE.β[hist] = Set{AbstractArray}()
        push!(TREE.β[hist], belief)
    end
end

function sample_obs(β, d, tab_pomdp)
    # Sample possible observations (with weights)
    nonzero_weights, nonzero_states = nonzero(β)
    obs_weights = weighted_column_sum(nonzero_weights, tab_pomdp.O[:, end, nonzero_states])
    obs_samples, obs_weights = maxk(obs_weights)
    
    if d==0  # enforce fully-observable for final (sink) state
        return first(obs_samples)
    else
        return sample(obs_samples, Weights(obs_weights))
    end
end

function sample_node(TREE, t)
    aos = collect(TREE.T[t])
    index = sample(1:length(aos), Weights(getd(TREE.Q, aos)))
    return rand(TREE.β[aos[index]]), aos[index]
end

function simulate_node!(TREE, Params, β, h)
    # Termination condition due to depth
    if depth(h) == Params[:max_t]
        return bayesian_prob(Params[:tab_pomdp], Params[:actions_pomdp], β, h)
    end

    if !(h in TREE.AO)
        return rollout(β, h, Params)
    end

    obs = sample_obs(β, depth(h), Params[:tab_pomdp])
    act = UCT_action(TREE, Params[:exploration_const], Params[:actions_pomdp], obs, h)
    
    oh = (obs, h...)
    aoh = (Params[:actions_pomdp][act], obs, h...)

    ## This part is backwards in time (from leaf to root)
    # Get previous belief, given the sampled observation and selected action
    LP = validate_single_action(Params[:tab_pomdp], obs, Params[:policy], β, Params[:LP_Solver], act)
    if isnothing(LP)
        q = 0.0
        push!(TREE, hist=aoh, with_time=false)
    else
        β_prev = sample_from_belief_subspace(LP, Params[:tab_pomdp], obs)
        q = simulate_node!(TREE, Params, β_prev, aoh)
        push!(TREE, hist=aoh, belief=β_prev)
    end

    # Update Tree
    TREE.N[oh] += 1
    TREE.N[aoh] += 1
    TREE.Q[aoh] += (q - TREE.Q[aoh]) / TREE.N[aoh]

    if !(q==0.0)
        TREE.P[BeliefRecord(β_prev, aoh)] = q
    end

    # @show aoh
    return q
end


function rollout(β, h, Params)
    # Termination condition due to depth
    if depth(h) == Params[:max_t]
        return bayesian_prob(Params[:tab_pomdp], Params[:actions_pomdp], β, h)
    end

    obs = sample_obs(β, depth(h), Params[:tab_pomdp])
    act = rollout_action(h, Params[:actions_pomdp])

    ## This part is backwards in time (from leaf to root)
    # Get previous belief, given the sampled observation and selected action
    LP = validate_single_action(Params[:tab_pomdp], obs, Params[:policy], β, Params[:LP_Solver], act)
    if isnothing(LP)
        return 0.0
    end

    β_prev = sample_from_belief_subspace(LP, Params[:tab_pomdp], obs)
    aoh = (Params[:actions_pomdp][act], obs, h...)
    return rollout(β_prev, aoh, Params)
end

function rollout_action(h, acts)
    # TODO. Change this.
    return rand(1:length(acts))
end


function search!(pomdp, policy, β_final, max_t, LP_Solver, no_of_simulations=5, exploration_const=1.0)
    tab_pomdp = tabulate(pomdp)
    actions_pomdp = actions(pomdp)

    Params = Dict([:policy, :max_t, :LP_Solver, :exploration_const, :max_t, :tab_pomdp, :actions_pomdp]
                    .=> [policy, max_t, LP_Solver, exploration_const, max_t, tab_pomdp, collect(actions_pomdp)])

    # Initialize tree with single leaf node
    TREE = BackwardTree(max_t)
    push!(TREE, belief=β_final, hist=(:end, -1))

    for t = Tqdm(1:max_t)
        println("  Timestep:\t  $(t) of $(max_t)")
        for m = Tqdm(1:no_of_simulations)
            β, h = sample_node(TREE, t-1)
            simulate_node!(TREE, Params, β, h)
        end
    end
    
    return TREE
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

function bayesian_next_belief(tab_pomdp, o, a, b)
    bp = tab_pomdp.O[o, a, :] .* (create_T_bar(tab_pomdp, a) * reshape(b, :, 1))
    return normalize(vec(bp))
end

function bayesian_prob(tab_pomdp, acts, bel, aos)
    prob = 1.0
    bp = bel

    len_items = Int(length(aos) / 2 - 1)
    for idx in 1:len_items
        i = (idx-1)*2 + 1
        a_sym, o = aos[i: i+1]
        a = findfirst(x->x==a_sym, acts)
        prob *= branch_weight(tab_pomdp, o, a, bp)
        bp = bayesian_next_belief(tab_pomdp, o, a, bp)
    end

    # for (a_sym, o) in aos[1:end-2]
    #     a = findfirst(x->x==a_sym, acts)
    #     prob *= branch_weight(tab_pomdp, o, a, bp)
    #     bp = bayesian_next_belief(tab_pomdp, o, a, bp)
    # end
    return round(prob; digits=5)
end
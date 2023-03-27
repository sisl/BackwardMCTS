include("utils.jl")
include("VertexPivot.jl")
include("GridWorld_LP_matrix.jl")


@with_kw mutable struct BackwardTree
    AO = Set{Tuple}()                                                   # histories of (a,o,...)
    T  = Dict{Int, Set{Tuple}}()                                        # timestep -> corresponding values of AO
    
    β  = Dict{Tuple, Set{AbstractArray}}()                              # hist -> set of belief nodes
    Q  = DefaultDict{Tuple, Float64}(0.0)                               # hist -> Q-value
    N  = DefaultDict{Tuple, Int}(0.0)                                   # hist -> N-value

    P = Dict{BeliefRecord, Float64}()                                   # belief/hist -> reachability probability
end

# Instantiate:
BackwardTree(max_t) = BackwardTree(T=Dict(i=>Set{Tuple}() for i=0:max_t))
Base.length(T::BackwardTree) = length(T.P)

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
    isempty(aos) && return true, nothing, nothing

    index = sample(1:length(aos), Weights(getd(TREE.Q, aos)))
    return false, rand(TREE.β[aos[index]]), aos[index]
end

function simulate_node!(TREE, Params, β, h)
    # Save belief, history, and reachability probability
    p = bayesian_prob(Params[:tab_pomdp], Params[:actions_pomdp], β, h)
    belRec = BeliefRecord(β, h)
    if !(p==0.0) && !(belRec in TREE.P)
        TREE.P[belRec] = p
    end

    # Termination condition due to depth
    if depth(h) == Params[:max_t]
        return p
    end

    if !(h in TREE.AO)
        return rollout(TREE, β, h, Params)
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
    TREE.N[oh]  += 1
    TREE.N[aoh] += 1
    TREE.Q[aoh] += (q - TREE.Q[aoh]) / TREE.N[aoh]

    return q
end


function rollout(TREE, β, h, Params)
    # Save belief, history, and reachability probability
    p = bayesian_prob(Params[:tab_pomdp], Params[:actions_pomdp], β, h)
    belRec = BeliefRecord(β, h)
    if !(p==0.0) && !(belRec in TREE.P)
        TREE.P[belRec] = p
    end

    # Termination condition due to depth
    if depth(h) == Params[:max_t]
        return bayesian_prob(Params[:tab_pomdp], Params[:actions_pomdp], β, h)
    end

    obs = sample_obs(β, depth(h), Params[:tab_pomdp])
    
    ## This part is backwards in time (from leaf to root)
    if Params[:rollout_random]
        # Select random action
        act = rand(1:length(Params[:actions_pomdp]))
        LP = validate_single_action(Params[:tab_pomdp], obs, Params[:policy], β, Params[:LP_Solver], act)
        if isnothing(LP)
            return 0.0
        end
    
    else
        # Get previous belief, given the sampled observation and selected action
        act, LP = validate_rollout_actions(Params[:tab_pomdp], obs, Params[:policy], β, Params[:LP_Solver])
        if isnothing(LP)
            return 0.0
        end
    end

    β_prev = sample_from_belief_subspace(LP, Params[:tab_pomdp], obs)
    aoh = (Params[:actions_pomdp][act], obs, h...)
    return rollout(TREE, β_prev, aoh, Params)
end

function merge_trees!(master::BackwardTree, localtrees::AbstractVector{BackwardTree})

    function update_broadcastable!(master::BackwardTree, branch::BackwardTree)        
        union!(master.AO, branch.AO)         # OK
        merge!(union, master.T,  branch.T)   # OK
        
        merge!(union, master.β,  branch.β)   # OK
        merge!(master.P,  branch.P)          # OK, when belRec is rounded (default)
    end
    
    function get_Q_times_N(branch::BackwardTree)
        temp_Q = Dict{Tuple, Float64}()
        ky = keys(branch.Q)
        vl = getd(merge(*, branch.N, branch.Q), ky)
    
        Q_times_N = Dict(ky.=>vl)
        return Q_times_N
    end
    
    function weighted_Q_avg(branch::BackwardTree, total_N)
        ky = keys(branch.Q)
        q_times_n = getd(merge(*, branch.N, branch.Q), ky)
        push!(total_N, (:end, -1)=>0)
    
        n_sum = getd(total_N, ky)
        q_avg = no_nan_division.(q_times_n, n_sum)
        return Dict(ky.=>q_avg)
    end
    
    function update_Q!(master::BackwardTree, localtrees::AbstractVector{BackwardTree})
        total_N = merge(+, [b.N for b in localtrees]...)   # the sum of all the N values
        Q_avg = weighted_Q_avg.(localtrees, Ref(total_N))
        master.Q = DefaultDict(0.0, merge(+, Q_avg...))
    end
    
    function extract_N!(master::BackwardTree, branch::BackwardTree)
        temp_N = deepcopy(master.N)
        merge!(-, branch.N, temp_N)
    end
    
    function update_N!(master::BackwardTree, localtrees::AbstractVector{BackwardTree})
        extract_N!.(Ref(master), localtrees);   # compute how much each branch has walked over master
        total_N_walked = merge(+, [b.N for b in localtrees]...)   # the sum of the line above
        merge!(+, master.N, total_N_walked)   # update master with correct new N    
    end

    # Update AO, T, β, P
    update_broadcastable!.(Ref(master), localtrees);

    # Update Q, N
    update_Q!(master, localtrees)
    update_N!(master, localtrees)

    return master
end


function search!(pomdp, policy, β_final, max_t, LP_Solver, sims_per_thread, no_of_threads, exploration_const=1.0, rollout_random=false)
    
    tab_pomdp = tabulate(pomdp)
    actions_pomdp = actions(pomdp)

    Params = Dict([:policy, :max_t, :LP_Solver, :exploration_const, :rollout_random, :max_t, :tab_pomdp, :actions_pomdp]
                    .=> [policy, max_t, LP_Solver, exploration_const, rollout_random, max_t, tab_pomdp, collect(actions_pomdp)])

    # Initialize global tree with single leaf node
    TREE = BackwardTree(max_t)
    push!(TREE, belief=β_final, hist=(:end, -1))

    for t = 1:max_t
        println("  Timestep:\t  $(t) of $(max_t)")
        
        for trd = Tqdm(1:sims_per_thread)
            localtrees = Array{BackwardTree}(undef, no_of_threads)  # undef initialization

            Threads.@threads for m = 1:no_of_threads
                # Initialize local tree from the global one
                m_Tree = deepcopy(TREE)
        
                empty_flag, β, h = sample_node(m_Tree, t-1)
                !empty_flag ? simulate_node!(m_Tree, Params, β, h) : BackwardTree()
                localtrees[m] = m_Tree
            end

            merge_trees!(TREE, localtrees);
            @show length(TREE)
        end
    end
    
    return TREE
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
        
        if prob == 0.0
            return 0.0
        end
    end

    # for (a_sym, o) in aos[1:end-2]
    #     a = findfirst(x->x==a_sym, acts)
    #     prob *= branch_weight(tab_pomdp, o, a, bp)
    #     bp = bayesian_next_belief(tab_pomdp, o, a, bp)
    # end
    return round(prob; digits=5)
end
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
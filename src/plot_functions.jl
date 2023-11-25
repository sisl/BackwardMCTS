using Plots
using StatsPlots
include("KDTree.jl")

# function heatmap_Tree_on_gridworld(pomdp, TREE; metric=:local)
#     """ Plot all nodes in the tree as a heatmap onto the gridworld. """
#     vals = zeros(pomdp.size)
#     N = zeros(pomdp.size)

#     for (i, belRec) in enumerate(keys(TREE.P))
#         bel, aos = belRec.β, belRec.ao
#         p = TREE.P[belRec]
        
#         if metric==:total
#             vals += reshape_GW(bel)
#         else
#             N_new = N + (reshape_GW(bel) .> 0)
#             # vals = (vals*N + reshape_GW(bel)) ./ (N_new)
#             vals += reshape_GW(bel)
#             replace!(vals, NaN=>0.0)
#         end

#         N = N_new
#     end

#     vals = vals ./ N
#     @show vals
#     @show N

#     f = heatmap(reverse(vals, dims=1), color=:grayC)
#     title!(f, "Reachability Probability Comparisons")
#     xlabel!(f, "Nodes in the Tree (Sorted)")
#     ylabel!(f, "Reachability Probability")

#     savefig(f, "../runs/" * CMD_ARGS[:savename] * "plot3.pdf")
#     return f
# end

function empty_gridworld(pomdp)
    f = heatmap(zeros(pomdp.size), color=:grayC, aspect_ratio=:equal,
    xlims=(0.5, pomdp.size[1]+0.5),
    xticks=collect(1:pomdp.size[1]),
    ylims=(0.5, pomdp.size[end]+0.5),
    yticks=collect(1:pomdp.size[end]),
    legend=false,
    grid=:all, minorgrid=true)

    title!(f, "State and Reward Space of the Gridworld")
    savefig(f, "../runs/" * CMD_ARGS[:savename] * "plot0.pdf")
    return f
end


function heatmap_Tree_on_gridworld(pomdp, TREE; metric=:total)
    """ Plot all nodes in the tree as a heatmap onto the gridworld. """
    vals = zeros(pomdp.size)
    N = 0

    for (i, belRec) in enumerate(keys(TREE.P))
        bel, aos = belRec.β, belRec.ao
        p = TREE.P[belRec]
        
        if metric==:total
            vals += reshape_GW(bel)
        else
            vals = (vals*N + reshape_GW(bel)) / (N+1)
        end

        N += 1

    end
    f = heatmap(reverse(vals, dims=1), color=:grayC, aspect_ratio=:equal,
                xlims=(0.5, pomdp.size[1]+0.5),
                xticks=collect(1:pomdp.size[1]),
                ylims=(0.5, pomdp.size[end]+0.5),
                yticks=collect(1:pomdp.size[end]))

    title!(f, "Distribution of Nodes onto the State Space")
    savefig(f, "../runs/" * CMD_ARGS[:savename] * "plot1.pdf")
    return f
end


function plot_Tree_avg_reachability_curves(TREE; custom_policy=nothing)
    """ Plot the three the curves of reachability. """
    p_bmcts, s_lower, tsteps = validation_probs_and_scores_UCT(TREE, pomdp, tab_pomdp, actions_pomdp, max_t, final_state, CMD_ARGS, upper_bound=false, custom_policy=custom_policy)
    _, s_upper, _ = validation_probs_and_scores_UCT(TREE, pomdp, tab_pomdp, actions_pomdp, max_t, final_state, CMD_ARGS, upper_bound=true, custom_policy=custom_policy)

    vals_sort_func((vals, times)) = (times, -vals)
    sp = sortperm(tuple.(p_bmcts, tsteps), by=vals_sort_func)
    popfirst!(sp)  # pop the t=0 entry

    tsteps = tsteps[sp]
    p_bmcts = p_bmcts[sp]
    s_lower = s_lower[sp]
    s_upper = s_upper[sp]

    f = plot()
    plot!(f, s_upper, linewidth=2, label="Empirical Value (Any Observations)")
    plot!(f, s_lower, linewidth=2, label="Empirical Value (Given Observations)")
    plot!(f, p_bmcts, linewidth=2, label="BMCTS Approximation", color=:black, legend=:topright)
    title!(f, "Reachability Probabilities of Belief Nodes")
    xlabel!(f, "Nodes in the Tree (Sorted by Timesteps)")
    ylabel!(f, "Reachability Probability")


    # Vertical span (highlight) in the figure:
    tdiff = (tsteps[1:end-1] - tsteps[2:end]) .!= 0
    pushfirst!(tdiff, true)
    tdiff[end] = true
    tdiff = [i for (i,v) in enumerate(tdiff) if v>0]

    for (idx, _) in enumerate(tdiff[1:end-1])
        if isodd(idx)
            vspan!(f, collect(tdiff[idx:idx+1]), alpha=0.15, alphaline=0.0, color=:black, label=nothing)
            # @show idx
        end
    end

    savefig(f, "../runs/" * CMD_ARGS[:savename] * "plot2.pdf")
    return f
end


function plot_Tree_avg_reachability_curves(TREE)
    """ Plot the three the curves of reachability. """
    p_bmcts, s_lower, tsteps = validation_probs_and_scores_UCT(TREE, pomdp, tab_pomdp, actions_pomdp, max_t, final_state, CMD_ARGS, upper_bound=false)
    _, s_upper, _ = validation_probs_and_scores_UCT(TREE, pomdp, tab_pomdp, actions_pomdp, max_t, final_state, CMD_ARGS, upper_bound=true)

    vals_sort_func((vals, times)) = (times, -vals)
    sp = sortperm(tuple.(p_bmcts, tsteps), by=vals_sort_func)
    popfirst!(sp)  # pop the t=0 entry

    tsteps = tsteps[sp]
    p_bmcts = p_bmcts[sp]
    s_lower = s_lower[sp]
    s_upper = s_upper[sp]

    f = plot()
    N = length(p_bmcts)
    A = repeat(tsteps, 2)
    B = vcat([["Forward Monte Carlo Simulations" for _ in 1:N],
              ["BMCTS Approximation" for _ in 1:N]]...)
    C = vcat([s_upper, p_bmcts]...)

    groupedboxplot!(f, A, C, group = B, bar_width = 0.8, legend=true, outliers=false)

    title!(f, "Reachability Probabilities of Belief Nodes")
    xlabel!(f, "Timesteps Backward")
    ylabel!(f, "Reachability Probability")

    savefig(f, "../runs/" * CMD_ARGS[:savename] * "plot21.pdf")

    
    g = plot()

    for t in sort(unique(tsteps))
        spp = (tsteps.==t)
        boxplot!(g, ["$(Int(t))"], abs.(s_lower[spp]-p_bmcts[spp]), outliers=false, color=:black, alpha=0.5, legend=false)
    end

    title!(g, "Errors in Reachability Probabilities of Belief Nodes")
    xlabel!(g, "Timesteps Backward")
    ylabel!(g, "BMCTS Absolute Error in Reachability Prob.\n(w.r.t. Forward MC Simulations w/ Preset Obs.)")

    savefig(g, "../runs/" * CMD_ARGS[:savename] * "plot22.pdf")

    return f, g
end



function plot_Validation_curves(TREE; epochs=[1e2, 1e3, 1e4, 1e5])
    default_value = CMD_ARGS[:val_epochs]
    probs_bmcts = nothing
    tsteps = nothing
    s_lower_vals = []
    f = plot()

    for e in epochs
        CMD_ARGS[:val_epochs] = e
        p_bmcts, s_lower, t = validation_probs_and_scores_UCT(TREE, pomdp, tab_pomdp, actions_pomdp, max_t, final_state, CMD_ARGS, upper_bound=false)
        probs_bmcts = p_bmcts
        push!(s_lower_vals, s_lower)
        tsteps = t
    end

    sp = sortperm(s_lower_vals[end], rev=false)
    pop!(sp)   # pop the last value as it has p=1.0

    for (e, s_lower) in zip(epochs, s_lower_vals)
        plot!(f, s_lower[sp], linewidth=2, label="Empirical Value for $(Int(e)) Epochs", ylims=(0.0, 1.0))    
    end

    CMD_ARGS[:val_epochs] = default_value
    plot!(f, probs_bmcts[sp], linewidth=2, label="BMCTS Approximation", ylims=(0.0, 1.0), color=:black, legend=:topleft)
    title!(f, "Epoch Value Comparisons")
    xlabel!(f, "Nodes in the Tree (Sorted by Reachability Probability)")
    ylabel!(f, "Reachability Probability")
    
    savefig(f, "../runs/" * CMD_ARGS[:savename] * "plot3.pdf")
    return f
end



function plot_Validation_curves(TREE; epochs=[1e2, 1e3, 1e4, 1e5, 1e6], name=["10²", "10³", "10⁴", "10⁵", "10⁶"])
    default_value = CMD_ARGS[:val_epochs]
    probs_bmcts = nothing
    tsteps = nothing
    s_lower_vals = []
    f = plot()

    for e in epochs
        CMD_ARGS[:val_epochs] = e
        p_bmcts, s_lower, t = validation_probs_and_scores_UCT(TREE, pomdp, tab_pomdp, actions_pomdp, max_t, final_state, CMD_ARGS, upper_bound=false)
        push!(s_lower_vals, s_lower)
        probs_bmcts = p_bmcts
        tsteps = t
    end

    sp = sortperm(s_lower_vals[end], rev=false)
    pop!(sp)   # pop the last value as it has p=1.0

    for (i, e, s_lower) in zip(1:length(name), epochs, s_lower_vals)
        boxplot!(f, [name[i]], abs.(s_lower-p_bmcts), outliers=false, color=:black, alpha=0.5, legend=false)
    end

    CMD_ARGS[:val_epochs] = default_value
    title!(f, "Benchmarking Number of Epochs")
    xlabel!(f, "Epochs")
    ylabel!(f, "BMCTS Absolute Error\n(w.r.t. Empirical Value)")
    
    savefig(f, "../runs/" * CMD_ARGS[:savename] * "plot30.pdf")
    return f
end



function plot_Tree_kdtree_curves_tracking(TREE; sigma_vals=[1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 3e-1, 5e-1, 1e0])
    kdtree = create_kdtree(TREE)
    kdtree_approx = []
    kdtree_scores = []
    kdtree_tsteps = []
    f = plot()

    for sigma in sigma_vals
        _, kdbayes_probs, kd_scores, kd_tsteps = benchmark_kdtree_diameter(kdtree, pomdp, final_state; sigma=sigma, upper_bound=false)
        push!(kdtree_approx, kdbayes_probs)
        push!(kdtree_scores, kd_scores)
        push!(kdtree_tsteps, kd_tsteps)
    end

    vals_sort_func((vals, times)) = (times, -vals)
    sp = sortperm(tuple.(kdtree_approx[1], kdtree_tsteps[1]), by=vals_sort_func)
    popfirst!(sp)  # pop the t=0 entry

    for (idx, sigma, val1, val2) in zip(reverse(1:length(sigma_vals)), reverse(sigma_vals), reverse(kdtree_approx), reverse(kdtree_scores))
        plot!(f, val1[sp]-val2[sp], linewidth=2, label="Voronoi Approximation for σ=$sigma", legend=:topright, ylims=(-0.06, 0.1)) 
        # if idx==1
        #     label = "Respective Empirical Value(s)"
        # else
        #     label = nothing
        # end
        # plot!(f, val2[sp], linewidth=1, label=label, color=:black, legend=:topright)
    end

    title!(f, "Any Belief (Voronoi) Approximations Empirical Accuracy")
    xlabel!(f, "Nodes in the Tree (Sorted by Timesteps)")
    ylabel!(f, "Discrepancy in Reachability Probability\n(Close to Zero is Better)")

    
    # Vertical span (highlight) in the figure:
    tsteps = kdtree_tsteps[1][sp]
    tdiff = (tsteps[1:end-1] - tsteps[2:end]) .!= 0
    pushfirst!(tdiff, true)
    tdiff[end] = true
    tdiff = [i for (i,v) in enumerate(tdiff) if v>0]

    for (idx, _) in enumerate(tdiff[1:end-1])
        if isodd(idx)
            vspan!(f, collect(tdiff[idx:idx+1]), alpha=0.15, alphaline=0.0, color=:black, label=nothing)
            # @show idx
        end
    end

    savefig(f, "../runs/" * CMD_ARGS[:savename] * "plot40.pdf")
    return f
end



function plot_Tree_kdtree_curves(TREE; sigma_vals=[1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 3e-1, 5e-1, 1e0])
    kdtree = create_kdtree(TREE)
    kdtree_approx = []
    kdtree_scores = []
    kdtree_tsteps = []
    f = plot()

    for sigma in sigma_vals
        _, kdbayes_probs, kd_scores, kd_tsteps = benchmark_kdtree_diameter(kdtree, pomdp, final_state; sigma=sigma, upper_bound=false)
        push!(kdtree_approx, kdbayes_probs)
        push!(kdtree_scores, kd_scores)
        push!(kdtree_tsteps, kd_tsteps)
    end

    vals_sort_func((vals, times)) = (times, -vals)
    sp = sortperm(tuple.(kdtree_approx[end], kdtree_tsteps[end]), by=vals_sort_func)
    popfirst!(sp)  # pop the t=0 entry

    val0 = kdtree_approx[1][sp]

    for (idx, sigma, val1, val2) in zip(reverse(1:length(sigma_vals)), reverse(sigma_vals), reverse(kdtree_approx), reverse(kdtree_scores))
        plot!(f, abs.(val1[sp].-val0), linewidth=2, label="Voronoi Approximation for σ=$sigma") 
        # if idx==length(sigma_vals)
        #     label = "Respective Empirical Value(s)"
        # else
        #     label = nothing
        # end
        # plot!(f, val2[sp], linewidth=1, label=label, ylims=(0.0, 1.0), color=:black, legend=:topright)
    end

    title!(f, "Discrepancies of Any Belief (Voronoi) Approximations")
    xlabel!(f, "Nodes in the Tree (Sorted by Timesteps)")
    ylabel!(f, "Discrepancy in Reachability Probability\n(Lower is Better)")


    # Vertical span (highlight) in the figure:
    tsteps = kdtree_tsteps[1][sp]
    tdiff = (tsteps[1:end-1] - tsteps[2:end]) .!= 0
    pushfirst!(tdiff, true)
    tdiff[end] = true
    tdiff = [i for (i,v) in enumerate(tdiff) if v>0]

    for (idx, _) in enumerate(tdiff[1:end-1])
        if isodd(idx)
            vspan!(f, collect(tdiff[idx:idx+1]), alpha=0.15, alphaline=0.0, color=:black, label=nothing)
            # @show idx
        end
    end

    savefig(f, "../runs/" * CMD_ARGS[:savename] * "plot41.pdf")
    return f
end



function heatmap_Tree_kdtree(TREE; sigma_vals=[1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 3e-1, 5e-1, 1e0], upper_bound=false)
    kdtree = create_kdtree(TREE)
    kdtree_approx = []
    kdtree_scores = []
    kdtree_tsteps = []

    for sigma in sigma_vals
        _, kdbayes_probs, kd_scores, kd_tsteps = benchmark_kdtree_diameter(kdtree, pomdp, final_state; sigma=sigma, upper_bound=upper_bound)
        push!(kdtree_approx, kdbayes_probs)
        push!(kdtree_scores, kd_scores)
        push!(kdtree_tsteps, kd_tsteps)
    end

    val0 = kdtree_approx[1]
    tsteps = kdtree_tsteps[1]

    X = sigma_vals
    Y = Int.(sort(unique(tsteps)))
    popfirst!(Y)  # pop the t=0 entry
    Z = fill(NaN, size(Y, 1), size(X, 1))
    Q = fill(NaN, size(Y, 1), size(X, 1))
    
    for i in eachindex(X), j in Y
        spp = (tsteps .== j)
        val1 = kdtree_approx[i]
        Z[j, i] = mean(abs.((val1[spp] .- val0[spp]) ./ val0[spp]))
        Q[j, i] = mean(abs.(val1[spp] .- val0[spp]))
    end

    f = heatmap(eachindex(X),Y,Z, xticks = (eachindex(X), X))

    title!(f, "Mean Relative Errors of Reachability Probabilities")
    xlabel!(f, "Distance from Voronoi Cell Centroid (× Cell Radius)")
    ylabel!(f, "Timesteps Backward")

    savefig(f, "../runs/" * CMD_ARGS[:savename] * "plot42.pdf")

    # g = heatmap(eachindex(X),Y,Q.*1e3, xticks = (eachindex(X), X), colorbar_title="\n× 10⁻³", right_margin = 3Plots.mm)
    g = heatmap(eachindex(X),Y,Q, xticks = (eachindex(X), X))

    title!(g, "Mean Absolute Errors of Reachability Probabilities")
    xlabel!(g, "Distance from Voronoi Cell Centroid (× Cell Radius)")
    ylabel!(g, "Timesteps Backward")

    savefig(g, "../runs/" * CMD_ARGS[:savename] * "plot43.pdf")

    return (f,g)
end


function stats_Tree_kdtree(TREE; sigma_vals=[5e-2, 1e0], upper_bound=true, custom_policy=nothing)
    kdtree = create_kdtree(TREE)
    kdtree_approx_pol1 = []
    kdtree_scores_pol1 = []
    kdtree_tsteps_pol1 = []

    kdtree_approx_pol2 = []
    kdtree_scores_pol2 = []
    kdtree_tsteps_pol2 = []

    for sigma in sigma_vals
        _, kdbayes_probs, kd_scores, kd_tsteps = benchmark_kdtree_diameter(kdtree, pomdp, final_state; sigma=sigma, upper_bound=upper_bound)
        push!(kdtree_approx_pol1, kdbayes_probs)
        push!(kdtree_scores_pol1, kd_scores)
        push!(kdtree_tsteps_pol1, kd_tsteps)

        _, kdbayes_probs, kd_scores, kd_tsteps = benchmark_kdtree_diameter(kdtree, pomdp, final_state; sigma=sigma, upper_bound=upper_bound, custom_policy=custom_policy)
        push!(kdtree_approx_pol2, kdbayes_probs)
        push!(kdtree_scores_pol2, kd_scores)
        push!(kdtree_tsteps_pol2, kd_tsteps)
    end

    tsteps = kdtree_tsteps[1]

    X = sigma_vals
    Y = Int.(sort(unique(tsteps)))
    popfirst!(Y)  # pop the t=0 entry
    Z = fill(NaN, size(Y, 1), size(X, 1))
    Q = fill(NaN, size(Y, 1), size(X, 1))
    
    for i in eachindex(X), j in Y
        spp = (tsteps .== j)
        val1 = kdtree_scores_pol1[i]
        val2 = kdtree_scores_pol2[i]
        Z[j, i] = mean(abs.((val2[spp] .- val1[spp]) ./ val1[spp]))
        Q[j, i] = mean(abs.(val2[spp] .- val1[spp]))
    end

    f = heatmap(eachindex(X),Y,Z, xticks = (eachindex(X), X))

    title!(f, "Mean Relative Errors of Reachability Probabilities")
    xlabel!(f, "Distance from Voronoi Cell Centroid (× Cell Radius)")
    ylabel!(f, "Timesteps Backward")

    savefig(f, "../runs/" * CMD_ARGS[:savename] * "plot52.pdf")

    # g = heatmap(eachindex(X),Y,Q.*1e3, xticks = (eachindex(X), X), colorbar_title="\n× 10⁻³", right_margin = 3Plots.mm)
    g = heatmap(eachindex(X),Y,Q, xticks = (eachindex(X), X))

    title!(g, "Mean Absolute Errors of Reachability Probabilities")
    xlabel!(g, "Distance from Voronoi Cell Centroid (× Cell Radius)")
    ylabel!(g, "Timesteps Backward")

    savefig(g, "../runs/" * CMD_ARGS[:savename] * "plot53.pdf")

    return (f,g)
    # return (kdtree_approx_pol1, kdtree_scores_pol1, kdtree_tsteps_pol1, kdtree_approx_pol2, kdtree_scores_pol2, kdtree_tsteps_pol2)

end
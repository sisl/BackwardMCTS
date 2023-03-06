using Plots

function plot_Tree_on_gridworld(pomdp, TREE, probs, scores)
    # Plots all the entries in the TREE to a gridworld. Darkness of each grid is prop. to probability.
    # Adjust the `alpha` value in the `scatter!` function to vary the darkness level.
    f = plot()
    len = length(keys(TREE.P))

    for (i, belRec) in enumerate(keys(TREE.P))
        bel, aos = belRec.β, belRec.ao
        p = TREE.P[belRec]

        # if p==1 continue end  # skip the entry in the tree that is already the sink state


        vals, idxs = nonzero(bel) 
        for (e,j) in enumerate(idxs)
            loc = states(pomdp)[j]
            scatter!([first(loc)], [last(loc)], alpha= p* vals[e]/2, markersize=10, label="", color=:black, grid=false)
        end
    end
    return f
end

# # Use above function as follows:
f = plot_Tree_on_gridworld(pomdp, TREE, probs, scores)
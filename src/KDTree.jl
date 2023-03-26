include("utils.jl")
include("GridWorld_MCTS_matrix.jl")

using NearestNeighbors
using Distributions: MvNormal
using POMDPs

mutable struct KDTree_and_probs
    tree::NearestNeighbors.KDTree
    hist_and_probs::Dict
end

function create_kdtree(TREE::BackwardTree)
    # Construct a KDTree with probs from a BackwardTree.

    beliefs = getfield.(collect(keys(TREE.P)), Ref(:β))
    beliefs, elems = unique_elems(beliefs)

    histories = getfield.(collect(keys(TREE.P)), Ref(:ao))
    histories = histories[elems]

    tree_probs = collect(values(TREE.P))
    tree_probs = tree_probs[elems]

    vals = [(h,p) for (h,p) in zip(histories, tree_probs)]


    kdtree = NearestNeighbors.KDTree(hcat(beliefs...))
    return KDTree_and_probs(kdtree, Dict(beliefs .=> vals))    
end

Base.length(kdtree::KDTree_and_probs) = length(kdtree.tree.data)



function benchmark_kdtree(kdtree::KDTree_and_probs, pomdp::POMDP, des_final_state; sigma::Float64, lower_bound=false, verbose=false)
    # Benchmark a consturcted KDTree by perturbing its elements.
    
    tree_probs = zeros(length(kdtree))
    bayes_probs = zeros(length(kdtree))
    scores = zeros(length(kdtree))

    tab_pomdp = tabulate(pomdp)
    acts = collect(actions(pomdp))

    function perturb(d, sigma)
        S = getRandomSamplesOnNSphere(d, sigma)  # samples of beliefs located on the ϵ-ball centered at belief
        S = S[:, isValidProb(S)]  # filter out samples that are not valid probability distributions
        idx = argmin(abs.(L1_norm(S) .- 1.0))  # find the sample whose L1 norm is closest to 1.0 (i.e. a valid probability)
        return normalize(S[:, last(idx.I)])
    end
    
    @info "Using $(Threads.nthreads()) threads.\nBackwardsTree has $(length(kdtree)) nodes."
    Threads.@threads for i = Tqdm(1:length(kdtree))  # (i,d) in Tqdm(enumerate(kdtree.tree.data))
        d = kdtree.tree.data[i]
        dp = perturb(d, sigma)

        # @show d
        # @show dp
        # @show L2_norm(dp-d)

        # KDTree data:
        hist, tree_prob = kdtree.hist_and_probs[d]  # Don't use the probs form the KDTree, but instead, estimate using Bayes' Rule
        bayes_prob = bayesian_prob(tab_pomdp, acts, dp, hist)

        # Validation:
        _, score = batch_fwd_simulations(pomdp, CMD_ARGS[:val_epochs], des_final_state, dp, convert_aos(pomdp, hist), lower_bound=lower_bound, verbose=verbose);

        if verbose
            println("  Item:\t\t  $(i) of $(items) \n  TREE Value:\t  $(p) \n  Approx Prob:\t  $(prob) \n  Lhood Score:\t  $(score) \n  aos:\t  $(aos)")
        end

        tree_probs[i] = tree_prob
        bayes_probs[i] = bayes_prob
        scores[i] = score
    end
    return tree_probs, bayes_probs, scores
end
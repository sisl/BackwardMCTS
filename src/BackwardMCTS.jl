module BackwardMCTS

@info "Loading module:\nBackward Monte Carlo Tree Search (BMCTS)"

greet() = print("Hello World!")

include("construct_tree.jl")
export
	BackwardTree,
    depth,
    search!,
    bayesian_prob


end  # module

# add POMDPPolicies POMDPSimulators QMDP NearestNeighbors Distances POMDPModelTools BeliefUpdaters JuMP Plots StatsBase StatsPlots Gurobi Memoize Dates Random Parameters StaticArrays CSV DataFrames LinearAlgebra Statistics DelimitedFiles ProgressBars Distributions Suppressor JLD DataStructures,
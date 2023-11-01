include("argparse_utils.jl")

using LinearAlgebra: Diagonal, dot, svd, cond, rank, diag, norm, transpose, qr, LAPACK.getrf!, checknonsingular, SingularException
using Statistics: mean, std
using DelimitedFiles
using ProgressBars
using Memoize
using Distributions: TriangularDist
using StatsBase: sample, Weights
using Suppressor
using JLD
using DataStructures: DefaultDict
using StatsBase: sample, Weights
using Parameters: @with_kw
using Dates: now
using Random

Tqdm(obj) = length(obj) == 1 ? obj : ProgressBars.tqdm(obj)

create_T_bar(tab_pomdp, act) = tab_pomdp.T[:, act, :]
create_O_bar(tab_pomdp, obs) = Diagonal(tab_pomdp.O[obs, 1, :])

add_columns = hcat
add_rows = vcat

fix_overflow!(val, ϵ=1e-10) = val[val .< ϵ] .= 0.0
average(A) = length(A) == 0 ? 0.0 : mean(A)

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

function maxk(A)
    idx = sortperm(A, rev=true)
    vals = A[idx]
    elems = (vals .> 0.0)
    return idx[elems], vals[elems]
end

function zeros_except(N::Int, idx::Int)
    res = zeros(N,)
    res[idx] = 1.0
    return res
end

function normalize(A::AbstractArray)
    return A ./ sum(A, dims=1)
end

function normalize!(A::AbstractArray)
    A[:] = A ./ sum(A, dims=1)
end

""" Get a list of items from a Dict or DefaultDict. """
getd(d::Union{Dict, DefaultDict}, k::Union{Base.KeySet, Set, AbstractArray}) = getindex.(Ref(d), k)

function unique_elems(S)
    elems = unique(i -> S[i], 1:length(S))
    return S[elems], elems
end

remove(list, item) = list[list .!= item]

# Returns float zero if result was actually NaN.
function no_nan_division(a::Number,b::Number)
    res = a/b
    @assert !isinf(res) "Result to infinity! a, b: $a, $b"
    return (isnan(res) ? 0.0 : res)
end

function csvdump(probs, scores, tsteps, CMD_ARGS; pop_name=false)
    mkpath("../runs/")  # mkdir if it doesn't exist
    if pop_name
        f = "../runs/" * pop!(CMD_ARGS, :savename) * ".csv"
    else
        f = "../runs/" * CMD_ARGS[:savename] * ".csv"
    end

    println("Saving to $f")

    perm = sortperm(string.(keys(CMD_ARGS)))
    header = reshape(collect(string.(keys(CMD_ARGS)))[perm], 1, :)
    vals = reshape(collect(values(CMD_ARGS))[perm], 1, :)

    open(f, "a") do io
        writedlm(io, hcat(["approxprob" "lhoodscore" "tsteps"], header), ", ")
        
        for (p,s,t) in zip(probs, scores, tsteps)
            writedlm(io, hcat([p s t], vals), ", ")
        end
    end
end

struct BeliefRecord
    β
    ao
end

# Enforce Dict has unique keys when keys <: BeliefRecord.
Base.isequal(t1::BeliefRecord, t2::BeliefRecord) = (t1.β == t2.β) && (t1.ao == t2.ao)
Base.hash(t::BeliefRecord) = Base.hash([t.β..., t.ao...])


struct CappedExponential
    vals
    probs
    len
end

struct LP_Solver_config 
    model
    z_dist_exp::CappedExponential
end

function zDistribution_exp(linspace_size=10000, z_min=0.0, z_max=1.0; exp_const=1.0)
    # Outputs a distribution whose pdf is proportional to exp(cx).
    vals = LinRange(z_min, z_max, linspace_size)
    probs = exp.(exp_const * vals)
    return CappedExponential(vals, probs, linspace_size)
end

# Sample from a CappedExponential with maximum value of `z_high`.
Base.rand(RNG, D::CappedExponential, z_high) = sample(RNG, view(D.vals, 1:Int(round(z_high*D.len))), Weights(view(D.probs, 1:Int(round(z_high*D.len)))))

# Check equality of structs x and y of same type
@generated function ≂(x, y)
    if !isempty(fieldnames(x)) && x == y
        mapreduce(n -> :(x.$n == y.$n), (a,b)->:($a && $b), fieldnames(x))
    else
        :(x == y)
    end
end

function softmax_neg(vals::AbstractArray)
    e = exp.(-vals)
    return e ./ sum(e)
end

L1_norm(A::AbstractArray) = sum(A, dims=1)
L2_norm(A::AbstractArray) = sqrt.(sum(A.^2, dims=1)) 

function getRandomSamplesOnNSphere(center, radius, num_of_samples=100000)
    # Gets uniformly random samples located on an N-dimensional sphere with radius R, centered at C.
    # center/radius: R/C of sphere, num_of_samples: how many samples to return.
    # From: https://stackoverflow.com/a/63354046
    dim = length(center)
    
    X = randn((dim, num_of_samples))
    abs_X = abs.(X)

    zr = (center .== 0)
    Y = (X .* .!zr) + (abs_X .* zr)

    normalizer = sqrt.(sum(Y.^2, dims=1))
    return center .+ (Y ./ normalizer .* radius)
end


function getRandomSamplesInNSphere(RNG, center, radius, num_of_samples=100000)
    # Gets uniformly random samples located inside an N-dimensional sphere with radius R, centered at C.
    # center/radius: R/C of sphere, num_of_samples: how many samples to return.
    # From: https://stackoverflow.com/a/63354046

    dim = length(center)

    X = randn((dim, num_of_samples))
    U = rand(RNG, (1, num_of_samples))

    normalizer = sqrt.(sum(X.^2, dims=1))
    return radius .* U.^(1/dim) ./ normalizer .* X
end


isValidProb(A::AbstractArray) = all.(eachcol(A.>0)) .& all.(eachcol(A.<1))

# function issingular(A::AbstractArray)
#     # lpt = getrf!(A)
#     try 
#         # checknonsingular(lpt[3])
#         inv(A)
#     catch e
#         (e isa SingularException) && return true
#     end
#     return false
# end

function issingular(A::AbstractArray)
    try
        singularvals = (svd(A)).S
        if minimum(singularvals) < 10*eps()
            return true
        else
            return false
        end
        
    catch
        return true
    end
end

# Finds the rank of `A` safely. 
function saferank(A::AbstractArray)
    if issingular(A)
        return 0
    else
        return rank(A)
    end
end

function saveTree(pomdp, T, fname)
    @suppress JLD.save("../runs/" * fname * "_pomdp.jld", "pomdp", pomdp)
    @suppress JLD.save("../runs/" * fname * "_tree.jld", "tree", T)
end

function loadTree(fname)
    CMD_ARGS[:savename] = "benchmarkRun_" * split(fname, '_')[end]
    pomdp = @suppress JLD.load("../runs/" * fname * "_pomdp.jld")["pomdp"]
    TREE = @suppress JLD.load("../runs/" * fname * "_tree.jld")["tree"]
    return (pomdp, TREE)
end

# Check if item is in dict or keys(dict)
Base.in(item::BeliefRecord, keys::Base.KeySet{BeliefRecord,Dict{BeliefRecord,Float64}}) = any(Ref(item) .≂ keys)
Base.in(item::BeliefRecord, dict::Dict{BeliefRecord,Float64}) = Base.in(item, keys(dict))

# Standard Error
ste(A::AbstractArray) = std(A) / sqrt(length(A))

# Mean Absolute Error
mae(A::AbstractArray) = sum(abs.(A)) / length(A)

function pprint(D::AbstractDict)
    for (k,v) in D
        println("$k\t=>  $v")
    end
end

function get_optimal_actions_matrix(pomdp, policy)
    Γ = policy.alphas
    no_of_states = length(states(pomdp))

    optimal_actions = zeros(no_of_states,)
    for i in 1:no_of_states
        belief = zeros(no_of_states,)
        belief[i] = 1.0
    
        act = argmax([dot(α, belief) for α in Γ])
        optimal_actions[i] = act
    end
    
    N = Int(sqrt(no_of_states-1))
    # @show states_matrix = reshape_GW(states(pomdp))
    # @show rewards_matrix = reshape_GW(tab_pomdp.R[:,1])
    optimal_actions_matrix = map(x->actions(pomdp)[x], reshape_GW(optimal_actions))
    
    return optimal_actions_matrix
end

include("argparse_utils.jl")

using LinearAlgebra: Diagonal, dot, rank, diag
using Statistics: mean
using DelimitedFiles
using ProgressBars
using Memoize
using Distributions: TriangularDist

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

function normalize(A::AbstractVector)
    return A ./ sum(A)
end

function normalize!(A::AbstractVector)
    A[:] .= A ./ sum(A)
end

remove(list, item) = list[list .!= item]

function csvdump(probs, scores, CMD_ARGS)
    f = pop!(CMD_ARGS, :savename)

    perm = sortperm(string.(keys(CMD_ARGS)))
    header = reshape(collect(string.(keys(CMD_ARGS)))[perm], 1, :)
    vals = reshape(collect(values(CMD_ARGS))[perm], 1, :)

    open(f, "a") do io
        writedlm(io, hcat(["approxprob" "lhoodscore"], header), ", ")
        
        for (p,s) in zip(probs, scores)
            writedlm(io, hcat([p s], vals), ", ")
        end
    end
end

struct LP_Solver_config 
    model
    z_threshold
end

struct BeliefRecord
    β
    ao
end

function zDistribution(z_max = 1.0)
    # Outputs a distributtion whose pdf is proportional to its input.
    # `z_max` is the upper bound to the z-value we know will not output any feasible solution to its corresponding LP.
    a = 0.0
    b = c = z_max
    return TriangularDist(a, b, c)
end

# Check equality of structs x and y of same type
@generated function ≂(x, y)
    if !isempty(fieldnames(x)) && x == y
        mapreduce(n -> :(x.$n == y.$n), (a,b)->:($a && $b), fieldnames(x))
    else
        :(x == y)
    end
end

# Check if item is in dict or keys(dict)
Base.in(item::BeliefRecord, keys::Base.KeySet{BeliefRecord,Dict{BeliefRecord,Float64}}) = any(Ref(item) .≂ keys)
Base.in(item::BeliefRecord, dict::Dict{BeliefRecord,Float64}) = Base.in(item, keys(dict))

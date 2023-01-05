using LinearAlgebra: Diagonal, dot, rank, diag
using Statistics: mean
using DelimitedFiles

include("argparse_utils.jl")

create_T_bar(tab_pomdp, act) = tab_pomdp.T[:, act, :]
create_O_bar(tab_pomdp, obs) = Diagonal(tab_pomdp.O[obs, 1, :])

add_columns = hcat
add_rows = vcat

fix_overflow!(val, ϵ=1e-10) = val[val .< ϵ] .= 0.0

average(A) = length(A) == 0 ? 0.0 : mean(A)

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
    z_val
end
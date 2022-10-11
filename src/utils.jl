using LinearAlgebra: Diagonal, dot, rank, diag

create_T_bar(tab_pomdp, act) = tab_pomdp.T[:, act, :]
create_O_bar(tab_pomdp, obs) = Diagonal(tab_pomdp.O[obs, 1, :])

add_columns = hcat
add_rows = vcat

fix_overflow!(val, ϵ=1e-10) = val[val .< ϵ] .= 0.0

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
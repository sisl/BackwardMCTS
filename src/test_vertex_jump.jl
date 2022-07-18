function edge_transition(LP, B, q)
    A, b, c = LP.A, LP.b, LP.c
    n = size(A, 2)
    b_inds = sort(B)
    n_inds = sort!(setdiff(1:n, B))
    AB = A[:,b_inds]
    d, xB = AB\A[:,n_inds[q]], AB\b
    p, xq′ = 0, Inf
    for i in 1 : length(d)
        if d[i] > 0
            v = xB[i] / d[i]
            if v < xq′
                p, xq′ = i, v
            end
        end
    end
    return (p, xq′)
end

mutable struct LinearProgram
    A
    b
    c
end

function get_vertex(B, LP)
    A, b, c = LP.A, LP.b, LP.c
    b_inds = sort!(collect(B))
    AB = A[:,b_inds]
    @assert size(AB,1) == size(AB,2)    # AB must be square; i.e. B must have rows(A) amount of elements
    xB = AB\b
    x = zeros(length(c))
    x[b_inds] = xB
    return x
end

function step_lp!(B, LP)
    A, b, c = LP.A, LP.b, LP.c
    n = size(A, 2)
    b_inds = sort!(B)
    n_inds = sort!(setdiff(1:n, B))
    AB, AV = A[:,b_inds], A[:,n_inds]
    xB = AB\b
    cB = c[b_inds]
    λ = AB' \ cB
    cV = c[n_inds]
    μV = cV - AV'*λ
    q, p, xq′, Δ = 0, 0, Inf, Inf
    for i in 1 : length(μV)
        if μV[i] < 0
            pi, xi′ = edge_transition(LP, B, i)
            if μV[i]*xi′ < Δ
                q, p, xq′, Δ = i, pi, xi′, μV[i]*xi′
            end
        end
    end
    @show q
    if q == 0
        return (B, true) # optimal point found
    end
    if isinf(xq′)
        error("unbounded")
    end
    j = findfirst(isequal(b_inds[p]), B)
    B[j] = n_inds[q] # swap indices
    @show B'
    return (B, false) # new vertex but not optimal
    end

function minimize_lp!(B, LP)
    done = false
    while !done
        @show time()
        B, done = step_lp!(B, LP)
    end
    return B
end

function get_valid_partition(A, X)
    @assert rank(A) == size(A, 1) # matrix A must be full-rank

    # Get non-zero vertex indices
    B = (1:length(X))[JuMP.value.(X) .!= 0.0]

    if length(B) == size(A, 1)
        return B
    else
        # Add zero vertices if required
        V = collect(1:length(X))
        deleteat!(V, B)
        rank_des = length(B) + 1

        for idx = 1 : length(V)
            temp = vcat(B, [V[idx]])
            AB = A[:,temp];
            if rank(AB) == rank_des
                push!(B, V[idx])
                rank_des += 1
            end
            if rank(AB) == size(A, 1)
                break
            end
        end

        return B
    end
end

include("test_gridworld_lp_matrix.jl")    # get A,b,C matrices

A = collect(A);
LP = LinearProgram(A, b, c);
B = get_valid_partition(A, X)

minimize_lp!(B, LP)
x_star = get_vertex(B, LP)

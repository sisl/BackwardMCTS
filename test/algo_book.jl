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
    if q == 0
    return (B, true) # optimal point found
    end
    if isinf(xq′)
    error("unbounded")
    end
    j = findfirst(isequal(b_inds[p]), B)
    B[j] = n_inds[q] # swap indices
    return (B, false) # new vertex but not optimal
    end

function minimize_lp!(B, LP)
    done = false
    while !done
    B, done = step_lp!(B, LP)
    end
    return B
end

# # Example 11.7
# A = [[1 1 1 0]; [-4 2 0 1]]
# b = [9; 2]
# c = [3; -1; 0; 0]
# LP = LinearProgram(A, b, c)

# B = [3, 4]
# minimize_lp!(B, LP)
# x_star = get_vertex(B, LP)

# TODOs:
# 1. Pre-solve and find the A,b,c matrices.
    # - Maybe gurobi can return me the Ax ≤ b version, and I can make it Ax = b.
# 2. Change the code above s.t. we traverse optimal vertices.
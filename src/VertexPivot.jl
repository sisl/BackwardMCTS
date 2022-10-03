remove(col, item) = col[col .!= item]

function remove_redundant_col(A, B, rank_des)
    for idx in reverse(B)  #keep reverse
        temp = remove(B, idx)
        AB = A[:,temp];
        if rank(AB) == rank_des
            return temp
        end
    end
end

mutable struct LinearProgram
    A
    b
    c
    X_init
    no_of_states
    vertices
    a_star
end

function extract_vertex(B, LP)
    A, b, c = LP.A, LP.b, LP.c
    b_inds = sort!(collect(B))
    AB = A[:,b_inds]
    @assert size(AB,1) == size(AB,2)    # AB must be square; i.e. B must have rows(A) amount of elements
    xB = AB\b
    x = zeros(length(c))
    x[b_inds] = xB
    return x
end

function edge_transitions!(LP, B, q)
    ϵ = 1e-10
    A, b, c, no_of_states = LP.A, LP.b, LP.c, LP.no_of_states
    rank_des = rank(A)
    n = size(A, 2)
    b_inds = sort(B)
    n_inds = sort!(setdiff(1:n, B))
    AB = A[:,b_inds]
    d, xB = AB\A[:,q], AB\b

    current_sol = xB[B .< no_of_states]
    current_vertex = B[B .< no_of_states]
    current_vertex = current_vertex[current_sol .> ϵ]

    for p in current_vertex
        flag, B′ = find_bases(LP, B, rank_des, q, p)
        if flag
            # b_inds = sort!(B′)
            # AB = A[:,b_inds]
            # xB = AB\b
            # temp_sol = xB[B′ .< no_of_states]
            # temp_vertex = B′[B′ .< no_of_states]
            # temp_vertex = temp_vertex[temp_sol .> ϵ]
            # push!(LP.vertices, temp_vertex)

            push!(LP.vertices, sort(B′))
        end
    end
end


function find_bases(LP, B, rank_des, q, p)
    # entering: q
    # leaving:  p

    B0 = B
    A = LP.A

    B = remove(B, p)
    n = size(LP.A, 2)

    # Add zero vertices if required
    V = collect(1:n)
    deleteat!(V, B)
    sort!(push!(B, q))
    V = remove(V, q)

    # see if by just swapping q and p, we can have full-rank
    if rank(A[:,B]) == rank_des
        return (true, B)

    # see if with q and p coexisting, we can have full-rank
    else
        push!(B, p)
        if rank(A[:,B]) == rank_des
            B = remove_redundant_col(A, B, rank_des)
            return (true, B)
        end 
    end

    # try to find another col to add for full rank
    V = remove(V, p)
    for idx = length(V) : -1 : 1  #keep reverse
        temp = vcat(B, [V[idx]])
        AB = A[:,temp];
        if rank(AB) == rank_des
            return (true, sort(B))
        end
    end

    return (false, B0)
end


function get_polygon_vertices!(B, LP)
    ϵ = 1e-10
    A, b, c, no_of_states = LP.A, LP.b, LP.c, LP.no_of_states
    n = size(A, 2)
    # b_inds = sort(B)
    # n_inds = sort(setdiff(1:n, B))
    # @assert b_inds == B
    # @assert n_inds == setdiff(1:n, B)
    b_inds = B
    n_inds = setdiff(1:n, B)
    AB, AV = A[:,b_inds], A[:,n_inds]
    @show(size(AB))
    # AB = AB .+ 1e-6*LinearAlgebra.I(size(A,1))
    xB = AB\b
    cB = c[b_inds]
    λ = AB' \ cB
    cV = c[n_inds]
    μV = cV - AV'*λ
    @show λ
    @show μV

    # current_sol = xB[B .< no_of_states]
    # current_vertex = B[B .< no_of_states]
    # current_vertex = current_vertex[current_sol .> ϵ]
    push!(LP.vertices, B)

    possible_pivots = n_inds[(1:length(μV))[abs.(μV) .<= ϵ]]
    possible_pivots = possible_pivots[possible_pivots .< no_of_states]
    
    for q in possible_pivots
        edge_transitions!(LP, B, q) # --> TODO: This should call itself recursively
    end

end


function get_valid_partition(A, X)
    @assert rank(A) == size(A, 1)    # matrix A must be full-rank

    # Get non-zero vertex indices
    ϵ = 1e-10
    B = (1:length(X))[abs.(JuMP.value.(X)) .> ϵ]

    if length(B) == size(A, 1)
        return B
    else
        # Add zero vertices if required
        V = collect(1:length(X))
        deleteat!(V, B)
        rank_des = length(B) + 1

        for idx = length(V) : -1 : 1  #keep reverse
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

        return sort(B)
    end
end


# USAGE:

# include("test_gridworld_lp_matrix.jl")    # get A,b,C matrices

# A = collect(A);
# LP = LinearProgram(A, b, c, JuMP.value.(X), no_of_states, Set());
# B = get_valid_partition(A, X)

# get_polygon_vertices!(B, LP);
# x_star = extract_vertex(B, LP)

# β = reshape_GW(x_star[1:no_of_states])
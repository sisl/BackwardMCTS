include("utils.jl")

mutable struct LinearProgram
    A
    b
    c
    X_init
    no_of_states
    vertices
    a_star
end

function remove_redundant_col(A, B, rank_des)
    for idx in reverse(B)  #keep reverse
        temp = remove(B, idx)
        AB = @view A[:,temp];
        if rank(AB) == rank_des
            return temp
        end
    end
end

@memoize function extract_vertex(B, LP)
    A, b, c = LP.A, LP.b, LP.c
    b_inds = sort!(collect(B))
    AB = @view A[:,b_inds]
    @assert size(AB,1) == size(AB,2)    # AB must be square; i.e. B must have rows(A) amount of elements
    xB = AB\b
    x = zeros(length(c))
    x[b_inds] = xB
    return x
end

@memoize function edge_transitions!(LP, B, q)
    ϵ = 1e-10
    A, b, c, no_of_states = LP.A, LP.b, LP.c, LP.no_of_states
    rank_des = rank(A)
    n = size(A, 2)
    b_inds = sort(B)
    n_inds = sort!(setdiff(1:n, B))
    AB = @view A[:,b_inds]
    d, xB = AB\A[:,q], AB\b

    current_sol = xB[B .< no_of_states]
    current_vertex = B[B .< no_of_states]
    current_vertex = current_vertex[current_sol .> ϵ]

    for p in current_vertex
        flag, B′ = find_bases(LP, B, rank_des, q, p)
        if flag
            # b_inds = sort!(B′)
            # AB = @view A[:,b_inds]
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
    if rank(@view A[:,B]) == rank_des
        return (true, B)

    # see if with q and p coexisting, we can have full-rank
    else
        push!(B, p)
        if rank(@view A[:,B]) == rank_des
            B = remove_redundant_col(A, B, rank_des)
            return (true, B)
        end 
    end

    # try to find another col to add for full rank
    V = remove(V, p)
    for idx = length(V) : -1 : 1  #keep reverse
        temp = vcat(B, [V[idx]])
        AB = @view A[:,temp];
        if rank(AB) == rank_des
            return (true, sort(B))
        end
    end

    return (false, B0)
end


@memoize function get_polygon_vertices!(B, LP)
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
    # @show(size(AB))
    # AB = AB .+ 1e-6*LinearAlgebra.I(size(A,1))
    xB = AB\b
    cB = c[b_inds]
    λ = AB' \ cB
    cV = c[n_inds]
    μV = cV - AV'*λ
    # @show λ
    # @show μV

    # current_sol = xB[B .< no_of_states]
    # current_vertex = B[B .< no_of_states]
    # current_vertex = current_vertex[current_sol .> ϵ]
    push!(LP.vertices, B)

    possible_pivots = n_inds[(1:length(μV))[abs.(μV) .<= ϵ]]
    possible_pivots = possible_pivots[possible_pivots .< no_of_states]
    
    for q in possible_pivots
        edge_transitions!(LP, B, q)
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
            AB = @view A[:,temp];
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


# function get_valid_partition(A, X)
#     # 1. Instead of adding columns, remove them one by one.
#     # or 2. Solve an auxiliary LP that find you which columns (of amount size(A,1)) that 
#     # should be chosen s.t. the chosen columns are linearly independent. 
# end


# function get_valid_auxiliary_partition(A, b, X, LP_Solver)
#     model = Model(LP_Solver)

#     (m, n) = size(A);
#     # Z = diagm(vec(b .>= 0.0) - vec(b .< 0.0));
    
#     # @variable(model, x[1:n] >= 0);
#     @variable(model, z[1:m] >= 0);
    
#     # @constraint(model, A*x + Z*z .== b);
#     @objective(model, Max, sum([z*i for ]));
#     # optimize!(model);


# end


function get_valid_partition_aux(A, X)
    """ Find the indices of a valid partition on the optimal poligon. """
    # @assert rank(A) == size(A, 1)    # matrix A must be full-rank

    # Get non-zero vertex indices
    ϵ = 1e-10
    B = (1:length(X))[abs.(JuMP.value.(X)) .> ϵ]
    AB = @view A[:,B];
    rAB = rank(AB)


    if length(B) == size(A, 1)
        return B
    else
        while rAB != size(A, 1)
            # Add zero vertices as required
            V = collect(1:length(X))
            deleteat!(V, B)
            
            # Compute the rank-revealing QR factorization of the default AB
            AB = A[:,B];
            Q, R = qr(AB, Val(false));

            # Find most promising vertices
            Avs = A[:,V];
            ys = Q' * Avs;
            prenorm = Avs - Q * ys;
            num_of_cols_needed = size(A,1) - rAB
            v_idxs = partialsortperm(vec(L2_norm(prenorm)), 1:num_of_cols_needed, rev=true)
            
            # Construct new AB matrix
            temp = vcat(B, V[v_idxs])
            B = sort(temp)
            AB = @view A[:,B];
            rAB = rank(AB)
        end
    end

    if rAB == size(AB,2)  # no redundant vertices, great job!
        return B
    end

    not_to_be_removed = (1:length(X))[abs.(JuMP.value.(X)) .> ϵ]
    V = copy(B)
    setdiff!(V, not_to_be_removed)

    V_removed = []

    while rAB != size(AB,2)  # AB needs to be a full-rank square matrix

        AV = A[:,V];
        Q, R = qr(AV' * AV, Val(false));
        # @show length(V), size(AV' * AV)

        # global VR = V_removed
        # global VV = V
        # global RR = R

        abs_diagR = abs.(diag(R))
        # v_val = nothing
        # while true
        #     # idx_of_V = rand((1:length(abs_diagR))[abs_diagR .== minimum(abs_diagR)])  # but not in V_removed
        #     idx_of_V = argmin(abs_diagR)  # but not in V_removed

        spR = sortperm(abs_diagR)  # argmin is first, argmax is last element
        v_vals = V[spR]
        ff = findfirst(map(x-> !(x in V_removed), v_vals))
        v_val = v_vals[ff]

        #     if !(v_val in V_removed)
        #         # @show abs_diagR[idx_of_V]
        #         break
        #     else
        #         deleteat!(abs_diagR, idx_of_V)
        #         # @show length(abs_diagR)
        #     end 
        # end

        temp = @view A[:, B[B.!=v_val]];  # AB without column `v_val`
        rTemp = rank(temp)

        # @show v_val, length(V), length(B), size(temp)
        # @show length(V_removed)
        # @show rTemp, rAB

        
        if rTemp == rAB
            V = remove(V, v_val);
            B = remove(B, v_val)
            # @show length(V)
            rAB = rTemp
        else
            push!(V_removed, v_val)
            # @show "Removed", v_val
        end


        # # DEBUG
        # if length(V)<=1185
        #     error("Sth happened")
        # end
        
        rAB == length(B) && return B  # return if AB is now square

    # @show length(V_removed)
    # @show "---------------------------"
    end
    return B
end


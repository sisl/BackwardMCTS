include("VertexPivot.jl")

using Suppressor
using JuMP
using LinearAlgebra: Diagonal, dot, rank
using Random

Random.seed!(1)

create_T_bar(tab_pomdp, act) = tab_pomdp.T[:, act, :]
create_O_bar(tab_pomdp, obs) = Diagonal(tab_pomdp.O[obs, 1, :])

add_columns = hcat
add_rows = vcat

fix_overflow!(val, ϵ=1e-10) = val[abs.(val) .< ϵ] .= 0.0

function reshape_GW(A::AbstractVector)
    no_of_states = length(A)
    N = Int(sqrt(no_of_states-1))
    res = reshape(A[1:end-1], (N,N))
    return rotl90(res)
end

function zeros_except(N::Int, idx::Int)
    res = zeros(N,)
    res[idx] = 1.0
    return res
end

reshape_GW(A::AbstractMatrix) = reshape_GW(vec(A))

function normalize(A::AbstractVector)
    return A ./ sum(A)
end

function normalize!(A::AbstractVector)
    A[:] .= A ./ sum(A)
end

function obj_func(O, T, β_t, x)
    no_of_states = length(x)
    y = ones(1,no_of_states)*O*T*x
    β_t_hat = O*T*x
    res = abs.(y.*β_t - β_t_hat)
    return sum(res)
end

function validate(O, T, Γ, αj, β_t, LP_Solver)
    no_of_states = length(β_t)
    eps_var = 1.0

    # @variable(model, x[1:no_of_states])
    # @variable(model, u[1:no_of_states])
    # @variable(model, y)

    # @variable(model, s1)
    # @variable(model, s2[1:no_of_states])

    # @variable(model, g1)
    # @variable(model, g2)
    # @variable(model, g3)

    # @variable(model, m[1:no_of_states])
    # @variable(model, n[1:no_of_states])

    model = Model(LP_Solver)

    no_of_LP_vars = 4 * no_of_states + 4
    @variable(model, X[1 : no_of_LP_vars])
    
    zr = zeros(1, no_of_states);
    Zr = zeros(no_of_states, no_of_states);
    vn = ones(1, no_of_states);
    Vn = ones(no_of_states, no_of_states);
    Eye = Diagonal(ones(no_of_states));
    
    # Default constraints
    A = [[vn*O*T zr 0];
         [zr zr 1]]
    
    b = [1.0; eps_var]
    
    
    # Alpha-vector constraints
    A = add_columns(A, zeros(size(A, 1), length(Γ)-1));
    
    function add_alpha_constraints(A, Γ, αj)
        counter = 0
        for αk in 1:length(Γ)
            if (αk != αj)
                counter += 1
                
                temp = zeros(1, length(Γ)-1)
                temp[counter] = -1
                A = add_rows(A, [(Γ[αj]-Γ[αk])' zr 0 temp]);
                
            end
        end
    
        return A
    end
    
    A = add_alpha_constraints(A, Γ, αj);
    b = add_rows(b, [0; 0; 0]);
    
    
    # L1 norm constraints
    A = add_columns(A, zeros(size(A, 1), 2*no_of_states));
    A = add_rows(A, [-O*T -Eye β_t zr' zr' zr' Eye Zr]);
    A = add_rows(A, [-O*T  Eye β_t zr' zr' zr' Zr -Eye]);
    b = add_rows(b, [zr'; zr']);
    
    
    # Nullspace constraint
    function add_nullspace_constraint(A, b, O, T)
        sum_cols = sum(O*T, dims=1)
        nullspace_columns = (sum_cols .== 0)
    
        for i in 1:no_of_states
            if nullspace_columns[i]
                row = zeros(1, size(A, 2))
                row[i] = 1
                A = add_rows(A, row)
            end
        end
    
        b = add_rows(b, zeros(sum(nullspace_columns), 1))
        return A, b
    end
    
    A, b = add_nullspace_constraint(A, b, O, T);
    
    c = zeros(1, no_of_LP_vars);
    c[no_of_states+1 : 2*no_of_states] .= 1
    @objective(model, Min, dot(c,X))
    
    # @assert rank(A) == size(A, 1)    # assert that A is full rank
    if rank(A) != size(A, 1)
        return zeros(1,no_of_states), Inf, A, b, c
    end
    
    @constraint(model, A*X .== b)
    @constraint(model, X .>= 0)
    
    # print(model)
    optimize!(model)
    @show termination_status(model)
    
    if termination_status(model) == JuMP.MathOptInterface.INFEASIBLE || termination_status(model) == JuMP.MathOptInterface.OTHER_ERROR || termination_status(model) == JuMP.MathOptInterface.INFEASIBLE_OR_UNBOUNDED
        return zeros(1,no_of_states), Inf, A, b, c

    else
        return JuMP.value.(X), JuMP.objective_value(model), A, b, c
    end
end

function validate_all_actions(tab_pomdp, obs_id, policy, β_t, LP_Solver)
    Γ = policy.alphas
    O = create_O_bar(tab_pomdp, obs_id)

    res = @suppress map(αj->validate(O, create_T_bar(tab_pomdp, policy.action_map[αj]), Γ, αj, β_t, LP_Solver), 1:length(Γ))
    

    J_min = minimum(getindex.(res, Ref(2)))   # index=2 is the obj value
    a_star = getindex.(res, Ref(2)) .== J_min

    X_inits = getindex.(res[a_star], Ref(1))    # index=1 is the x value

    A_matrices = getindex.(res[a_star], Ref(3))   # index=3 is the A matrix
    A_matrices = collect.(A_matrices);
    
    b_vectors = getindex.(res[a_star], Ref(4))   # index=4 is the b matrix
    b_vectors = collect.(b_vectors);

    c_vectors = getindex.(res[a_star], Ref(5))   # index=5 is the c matrix
    c_vectors = collect.(c_vectors);

    emptySets = [Set() for _ in 1:sum(a_star)]
    LPs = LinearProgram.(A_matrices, b_vectors, c_vectors, X_inits, Ref(no_of_states), emptySets);
    # global gA_matrices = A_matrices
    # global gX_inits = X_inits
    Bs = get_valid_partition.(A_matrices, X_inits);

    @suppress get_polygon_vertices!.(Bs, LPs);
    @suppress remove_polygon_vertices!.(LPs, Ref(Γ), (1:length(a_star))[a_star]);
    return LPs
end

function samples_from_belief_subspace(LP, belief_N)
    X_stars = reshape(Float64[], LP.no_of_states, 0)
    samples = []

    for B in LP.vertices
        x_star = extract_vertex(B, LP)[1:LP.no_of_states]
        fix_overflow!(x_star)
        X_stars = add_columns(X_stars, normalize(x_star))
    end

    n = size(X_stars, 2)
    for _ in 1:belief_N
        w = normalize(rand(n))
        s = X_stars * w
        push!(samples, s)
    end

    return samples
end

function vertices_from_belief_subspace(LP)
    X_stars = []

    for B in LP.vertices
        x_star = extract_vertex(B, LP)[1:LP.no_of_states]
        fix_overflow!(x_star)
        push!(X_stars, normalize(x_star))
    end

    return X_stars
end

function remove_polygon_vertices!(LP, Γ, act)
    @show act
    ϵ = 1e-10
    for B in LP.vertices
        x_star = extract_vertex(B, LP)
        utilities = dot.(Γ, Ref(x_star[1:LP.no_of_states]))
        if abs(maximum(utilities) - utilities[act]) > ϵ
            delete!(LP.vertices, B)
        end
    end
end
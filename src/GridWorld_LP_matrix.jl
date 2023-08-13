include("utils.jl")
include("VertexPivot.jl")

using Suppressor
using JuMP
using LinearAlgebra: Diagonal, dot, rank
using StatsBase: sample, Weights

function obj_func(O, T, β_t, x)
    no_of_states = length(x)
    y = ones(1,no_of_states)*O*T*x
    β_t_hat = O*T*x
    res = abs.(y.*β_t - β_t_hat)
    return sum(res)
end

function validate(O, T, Γ, αj, β_t, LP_Solver_model, z_val)
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
    
    model = Model(() -> Gurobi.Optimizer(LP_Solver_model))

    set_silent(model)  # no verbose 
    set_optimizer_attribute(model, "Threads", 1)  # single Gurobi call should use single thread 
    
    no_of_LP_vars = (4 * no_of_states) + 1 + (length(Γ)-1) + 1   # (x, u, m, n) + y + (g) + r
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
    b = add_rows(b, zeros(length(Γ)-1, 1));
    
    
    # L1 norm constraints
    A = add_columns(A, zeros(size(A, 1), 2*no_of_states));
    A = add_rows(A, [-O*T -Eye β_t repeat(zr, length(Γ)-1)' Eye Zr]);
    A = add_rows(A, [-O*T  Eye β_t repeat(zr, length(Γ)-1)' Zr -Eye]);
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
    

    # Min prob of (a,o) reachability constraint
    Oa = reshape(diag(O), 1, :)
    A = add_columns(A, zeros(size(A, 1), 1));
    item = Oa*T - z_val*vn
    A = add_rows(A, [item zr 0 zeros(1, length(Γ)-1) zr zr -1]);
    b = add_rows(b, [0])


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
    # @show termination_status(model)
    
    if termination_status(model) == JuMP.MathOptInterface.INFEASIBLE || termination_status(model) == JuMP.MathOptInterface.OTHER_ERROR || termination_status(model) == JuMP.MathOptInterface.INFEASIBLE_OR_UNBOUNDED
        return zeros(1,no_of_states), Inf, A, b, c

    else
        return JuMP.value.(X), JuMP.objective_value(model), A, b, c
    end
end

function get_z_high(O, T, Γ, αj, β_t, LP_Solver_model)
    no_of_states = length(β_t)

    model = Model(() -> Gurobi.Optimizer(LP_Solver_model))
    
    set_silent(model)  # no verbose 
    set_optimizer_attribute(model, "Threads", 1)  # single Gurobi call should use single thread 

    @variable(model, x[1:no_of_states])
    @variable(model, z)

    # Constraint 1: Updated belief should be a valid prob. distribution (RM: this is actually just the normalizing constant for next belief)
    # @constraint(model, 1.0 .== ones(1,no_of_states)*O*T*x)
    
    # Constraint 2: Alpha-vector constraints
    for αk in 1:length(Γ)
        if (αk != αj)
            @constraint(model, dot(Γ[αj], x) >= dot(Γ[αk], x))
        end
    end
    
    # Constraint 3: Belief x should be a valid prob. distribution
    @constraint(model, x .>= 0.0)
    @constraint(model, x .<= 1.0)
    @constraint(model, sum(x) == 1.0)

    # Constraint 4: Nullspace constraint
    sum_cols = sum(O*T, dims=1)
    nullspace_columns = (sum_cols .== 0)
    for i in 1:no_of_states
        if nullspace_columns[i]
            @constraint(model, x[i] .== 0)
        end
    end

    # Constraint 5: Reachability constraint
    Oa = reshape(diag(O), 1, :)
    @constraint(model, dot(Oa*T, x) == z)

    # # Constraint 6: Real vs Approx next belief should not be far
    # @variable(model, p1)
    # @constraint(model, O*T*x - β_t .<= p1)
    # @constraint(model, β_t - O*T*x .<= p1)
    # @constraint(model, p1 <= 0.7)

    @objective(model, Max, z)
    optimize!(model)

    if termination_status(model) == JuMP.MathOptInterface.INFEASIBLE || termination_status(model) == JuMP.MathOptInterface.OTHER_ERROR || termination_status(model) == JuMP.MathOptInterface.INFEASIBLE_OR_UNBOUNDED
        return 0.0
        
    else
        return JuMP.value.(z)
    end
end

function validate_single_action(RNG, tab_pomdp, obs_id, policy, β_next, LP_Solver, αj; with_J=false)
    Γ = policy.alphas
    O = create_O_bar(tab_pomdp, obs_id)
    T = create_T_bar(tab_pomdp, policy.action_map[αj])

    z_high = get_z_high(O, T, Γ, αj, β_next, LP_Solver.model)
    if (z_high == 0.0)
        return nothing
    end

    z_val = rand(RNG, LP_Solver.z_dist_exp, z_high)
    X, J, A, b, c = validate(O, T, Γ, αj, β_next, LP_Solver.model, z_val)
    # push!.(Ref(Vals), (:X, :J, :A, :b, :c).=>(X, J, A, b, c))
    
    if (J == Inf || J>0.5)
        return nothing
    end
    @show J

    LP = LinearProgram(A, b, c, X, no_of_states, Set(), αj);

    B = get_valid_partition_aux(RNG, A, X; verbose=false);
    
    if isnothing(B)
        return nothing
    end
    
    get_polygon_vertices!(B, LP);
    remove_polygon_vertices!(LP, Γ, αj);
    if !with_J
        return LP
    else
        return LP, J
    end
end


function validate_rollout_actions(RNG, tab_pomdp, obs_id, policy, β_next, LP_Solver)
    # Choose next rollout action w.r.t. softmax(-J).
    Γ = policy.alphas

    LPs = []
    Js = Float64[]

    for αj in 1:length(Γ)
        res = validate_single_action(RNG, tab_pomdp, obs_id, policy, β_next, LP_Solver, αj; with_J=true)
        !isnothing(res) ? (lp,j)=res : continue
        push!(LPs, lp)
        push!(Js, j)
    end
    
    if isempty(LPs)
        return nothing, nothing
    end

    # @show Js
    # @show softmax_neg(Js)
    chosen_act = sample(RNG, 1:length(Γ), Weights(softmax_neg(Js)))
    return chosen_act, LPs[chosen_act]
end


function sample_from_belief_subspace(RNG, LP, tab_pomdp, obs_id)
    X_stars = reshape(Float64[], LP.no_of_states, 0)
    X_stars_rchblty_probs = Float64[]
    samples = []

    for B in LP.vertices
        x_star = extract_vertex(B, LP)[1:LP.no_of_states]
        fix_overflow!(x_star)
        normalize!(x_star)
        X_stars = add_columns(X_stars, x_star)
        # @show obs_id
        push!(X_stars_rchblty_probs, branch_weight(tab_pomdp, obs_id, LP.a_star, x_star))
    end
    # @show X_stars_rchblty_probs

    n = size(X_stars, 2)
    # @show n
    if n==1
        return round.(vec(X_stars); digits=6)
    else
        # dirc = Dirichlet(ones(n))
        denom = minimum(X_stars_rchblty_probs)
        # @show X_stars_rchblty_probs
        # @show X_stars_rchblty_probs/denom
        dirc = Dirichlet(X_stars_rchblty_probs / denom)
        w = normalize(rand(RNG, dirc))
        s = X_stars * w
        return round.(s; digits=6)
    end
end

function samples_from_belief_subspace(RNG, LP, tab_pomdp, obs_id, belief_N)   # old (also has different input fields)
    X_stars = reshape(Float64[], LP.no_of_states, 0)
    X_stars_rchblty_probs = Float64[]
    samples = []

    for B in LP.vertices
        x_star = extract_vertex(B, LP)[1:LP.no_of_states]
        fix_overflow!(x_star)
        normalize!(x_star)
        X_stars = add_columns(X_stars, x_star)
        # @show obs_id
        push!(X_stars_rchblty_probs, branch_weight(tab_pomdp, obs_id, LP.a_star, x_star))
    end
    # @show X_stars_rchblty_probs

    n = size(X_stars, 2)
    if n==1
        return [vec(X_stars)]
    else
        # dirc = Dirichlet(ones(n))
        denom = minimum(X_stars_rchblty_probs)
        dirc = Dirichlet(X_stars_rchblty_probs./denom)
        for _ in 1:belief_N
            w = normalize(rand(RNG, dirc))
            s = X_stars * w
            push!(samples, s)
        end
    end

    return samples
end


@memoize function remove_polygon_vertices!(LP, Γ, act)
    # @show act
    ϵ = 1e-10
    for B in LP.vertices
        x_star = extract_vertex(B, LP)
        utilities = dot.(Γ, Ref(x_star[1:LP.no_of_states]))
        if abs(maximum(utilities) - utilities[act]) > ϵ
            delete!(LP.vertices, B)
        end
    end
end
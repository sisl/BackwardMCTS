include("utils.jl")

using Suppressor
using JuMP, HiGHS
using LinearAlgebra: Diagonal, dot, rank
using Random

Random.seed!(1)

function zeros_except(N::Int, idx::Int)
    res = zeros(N,)
    res[idx] = 1.0
    return res
end

function normalize(A::AbstractVector)
    return A ./ sum(A)
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
    slack_var = 1.0

    model = Model(LP_Solver)

    @variable(model, x[1:no_of_states])
    @variable(model, u[1:no_of_states])
    @variable(model, y)
    
    @constraint(model, 1.0 .== ones(1,no_of_states)*O*T*x)
    @constraint(model, y == slack_var)
    # @constraint(model, x .<= 1)  # --> this constraint is actually redundant
    @constraint(model, x .>= 0)
    # @constraint(model, 1 .== ones(1,no_of_states)*x)
    
    # Constraint: α-vector must be optimal for `x`
    for αk in 1:length(Γ)
        if (αk != αj)
            @constraint(model, dot(Γ[αj], x) >= dot(Γ[αk], x))
        end
    end
    
    # Constraint: no component of `x` can lie in the nullspace of O*T
    sum_cols = sum(O*T, dims=1)
    nullspace_columns = (sum_cols .== 0)
    for i in 1:no_of_states
        if nullspace_columns[i]
            @constraint(model, x[i] .== 0)
        end
    end

    @constraint(model, y*β_t - O*T*x .<= u)
    @constraint(model, y*β_t - O*T*x .>= -u)
    @objective(model, Min, sum(u))
    
    # print(model)
    optimize!(model)
    @show termination_status(model)
    
    if termination_status(model) == JuMP.MathOptInterface.INFEASIBLE || termination_status(model) == JuMP.MathOptInterface.OTHER_ERROR || termination_status(model) == JuMP.MathOptInterface.INFEASIBLE_OR_UNBOUNDED
        return zeros(1,no_of_states), Inf, nothing

    elseif JuMP.result_count(model) > 1    # there are more than 1 solutions to the LP
        solutions_x = [JuMP.value.(x; result=si).data for si in 1:result_count(model)]
        error("There are multiple solutons to the LP!")
        return solutions_x, JuMP.objective_value(model)

    else    # there is only 1 solution to the LP
        @show JuMP.value.(x)
        @show JuMP.value.(y)
        return JuMP.value.(x), JuMP.objective_value(model), model
    end
end

function validate_all_actions(tab_pomdp, obs_id, policy, β_t, LP_Solver)
    Γ = policy.alphas
    O = create_O_bar(tab_pomdp, obs_id)

    res = @suppress map(αj->validate(O, create_T_bar(tab_pomdp, policy.action_map[αj]), Γ, αj, β_t, LP_Solver), 1:length(Γ))
    
    # a_star = argmin(getindex.(res, Ref(2)))   # index=2 is the obj value
    # β_opt = res[a_star][1]                    # index=1 is the x value

    J_min = minimum(getindex.(res, Ref(2)))   # index=2 is the obj value
    a_star = getindex.(res, Ref(2)) .== J_min
    β_opt = getindex.(res[a_star], Ref(1))    # index=1 is the x value
    return unique(β_opt)
end

function validate_all_actions_multiple_solutions(tab_pomdp, obs_id, policy, β_t, LP_Solver)
    """
    This functions is for the case where there are multiple solutions to the LP. 
    It is not fully implemented as this case has never arose.
    """
    Γ = policy.alphas
    O = create_O_bar(tab_pomdp, obs_id)

    res = @suppress map(αj->validate(O, create_T_bar(tab_pomdp, policy.action_map[αj]), Γ, αj, β_t, LP_Solver), 1:length(Γ))

    J_values = collect(Iterators.flatten(getindex.(res, Ref(2))))   # index=2 is the obj value
    J_min = minimum(J_values)
    a_star = (J_values .== J_min)

    β_values = collect(Iterators.flatten(getindex.(res, Ref(1))))    # index=1 is the x value
    β_opt = β_values[a_star]
    return unique(β_opt)
end
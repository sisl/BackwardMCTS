include("gridworldpomdp.jl")

using Suppressor
using JuMP, HiGHS
using LinearAlgebra: Diagonal, dot
using Random
using POMDPPolicies: solve
using QMDP

Random.seed!(1)

create_T_bar(tab_pomdp, act) = tab_pomdp.T[:, act, :]
create_O_bar(tab_pomdp, obs) = Diagonal(tab_pomdp.O[obs, 1, :])

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

function obj_func(O, T, β_t, x)
    no_of_states = length(x)
    y = ones(1,no_of_states)*O*T*x
    β_t_hat = O*T*x
    res = abs.(y.*β_t - β_t_hat)
    return sum(res)
end

function add_soln_as_constraint!(model, x, x_soln)
    no_of_states = length(x_soln)
    ϵ = 1e-6
    M = 1e6
    v = ones(17)
    
    @variable(model, z, Bin)
    
    # @constraint(model, x - x_soln .<= -ϵ + M*z)
    # @constraint(model, x - x_soln .>=  ϵ - M*(1.0-z))
    
    # @constraint(model, !z => {sum(U) <= -ϵ + M*z})
    # @constraint(model, z => {sum(U) >=  ϵ - M*(1.0-z)})
    
    # @variable(model, U[1:no_of_states])
    # @constraint(model, x - x_soln .>= U)
    # @constraint(model, x - x_soln .<= -U)
    # @constraint(model, sum(U) >= ϵ)

    # @constraint(model, !z => {x[6] - x_soln[6] <= -ϵ + M*z})
    # @constraint(model, z => {x[6] - x_soln[6] >=  ϵ - M*(1.0-z)})

    @constraint(model, !z => {sum(x - x_soln) <= -ϵ + M*z})
    @constraint(model, z => {sum(x - x_soln) >=  ϵ - M*(1.0-z)})

    # return model
    return
end

function validate(O, T, Γ, αj, β_t, LP_Solver)
    no_of_states = length(β_t)
    slack_var = 1e-5

    model = Model(LP_Solver)

    @variable(model, x[1:no_of_states])
    @variable(model, u[1:no_of_states])
    @variable(model, y)
    
    @constraint(model, y .== ones(1,no_of_states)*O*T*x)
    @constraint(model, y >= slack_var)
    @constraint(model, x .<= 1)
    @constraint(model, x .>= 0)
    @constraint(model, 1 .== ones(1,no_of_states)*x)
    
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
    @show JuMP.result_count(model)
    
    if termination_status(model) == JuMP.MathOptInterface.INFEASIBLE || termination_status(model) == JuMP.MathOptInterface.OTHER_ERROR
        return zeros(1,no_of_states), Inf

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


# Params
LP_Solver = HiGHS.Optimizer

# Create pomdp
pomdp = SimpleGridWorldPOMDP(size=(4,4),
                            rewards=Dict(GWPos(2,3)=>-10.0, GWPos(3,1)=>+25.0)
                            ,
                            tprob = 0.6,
                            oprob = 0.7)

tab_pomdp = tabulate(pomdp)
no_of_actions = length(actions(pomdp))
no_of_states = length(states(pomdp))

# Create α-vectors
solver = QMDPSolver()
policy = solve(solver, tab_pomdp)
Γ = policy.alphas

# Create leaf belief
β_t = zeros(no_of_states,)
β_t[7] = 1.0
obs_id = 7

# Solve for root belief
a_star = 2
O_bar = create_O_bar(tab_pomdp, obs_id)
T_bar = create_T_bar(tab_pomdp, a_star)
β_opt, J_opt, model_val = validate(O_bar, T_bar, Γ, a_star, β_t, LP_Solver)

@show β_opt

# β_t_hat = normalize(O_bar * T_bar * β_opt);

# @show β_t
# @show β_opt

# β_opt_best = validate_all_actions(tab_pomdp, obs_id, policy, β_t, LP_Solver)
# @show β_opt_best
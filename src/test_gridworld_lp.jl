include("gridworldpomdp.jl")
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

function normalize(A::AbstractVector)
    return A ./ sum(A)
end

function obj_func(O, T, β_t, x)
    no_of_states = length(x)
    β_t_hat = normalize(O*T*x)
    res = abs.(β_t - β_t_hat)
    return sum(res)
    # return res
end

function validate(O, T, Γ, αj, β_t)
    no_of_states = length(β_t)
    slack_var = 1e-5

    @variable(model, x[1:no_of_states])
    @variable(model, u[1:no_of_states])
    @variable(model, y)
    
    @constraint(model, y .== ones(1, no_of_states)*O*T*x)
    @constraint(model, y >= slack_var)
    @constraint(model, x .<= 1)
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
    
    optimize!(model)
    
    # print(model)
    # @show termination_status(model)
    @show JuMP.value.(x)
    @show JuMP.value.(y)
    return JuMP.value.(x)
end

# Create pomdp
pomdp = SimpleGridWorldPOMDP(size=(4,4), rewards=Dict(GWPos(2,3)=>-10.0, GWPos(3,1)=>+25.0))
tab_pomdp = tabulate(pomdp)
no_of_actions = length(actions(pomdp))
no_of_states = length(states(pomdp))

# Create α-vectors
solver = QMDPSolver()
policy = solve(solver, tab_pomdp)
Γ = policy.alphas

# LP Model
model = Model(HiGHS.Optimizer)

# Create leaf belief
β_t = zeros(no_of_states,)
β_t[3] = 1.0

# Solve for root belief
a_star = 2
obs_id = 2
O_bar = create_O_bar(tab_pomdp, obs_id)
T_bar = create_T_bar(tab_pomdp, a_star)
β_opt = validate(O_bar, T_bar, Γ, a_star, β_t)

β_t_hat = normalize(O_bar * T_bar * β_opt);

@show β_t
@show β_opt
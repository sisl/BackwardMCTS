using JuMP
using HiGHS
using LinearAlgebra

# Model and vars
model = Model(HiGHS.Optimizer)

function validate(O, T, Γ, j, β_t)
    state_size = length(β_t)
    slack_var = -0.0001

    @variable(model, x[1:state_size])
    @variable(model, u[1:state_size])
    @variable(model, y)
    
    @constraint(model, y .== ones(1, state_size)*O*T*x)
    @constraint(model, y+slack_var >=0)
    @constraint(model, x .<= 1)
    @constraint(model, x .>= 0)
    
    for k in 1:length(Γ)
        if (k != j)
            @constraint(model, Γ[j]*x .>= Γ[k]*x)
        end
    end
    
    @constraint(model, y*β_t - O*T*x .<= u)
    @constraint(model, y*β_t - O*T*x .>= -u)
    @objective(model, Min, sum(u))
    
    optimize!(model)
    
    print(model)
    @show termination_status(model)
    @show value.(x)
    @show value.(y)
    return value.(x)/value.(y)
end

normalize_Func(Trans_Func) = Trans_Func ./ sum(Trans_Func, dims=1)

state_size = 100
action_number = 4

T = normalize_Func(rand(state_size,state_size)) #needs to be updated with T(a_j)
O = Diagonal(rand(state_size)) #needs to be updated with O^i

Γ = []
for i in 1:action_number
    alpha = rand(1,state_size)
    push!(Γ, alpha) 
end

β_t = normalize_Func(rand(state_size))
β_opt = validate(O, T, Γ, 1, β_t)

print(β_opt)

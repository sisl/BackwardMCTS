using JuMP
using Gurobi
using LinearAlgebra
using Random

# Model and vars
model = Model(Gurobi.Optimizer)
set_optimizer_attribute(model, "PoolSearchMode", 2)
set_optimizer_attribute(model, "PoolSolutions", 10)
set_optimizer_attribute(model, "PoolGap", 0)

@variable(model, x)
@variable(model, y)

@constraint(model, x <= 1)
@constraint(model, -x <= 0)
@constraint(model, y <= 1)
@constraint(model, -y <= 0)

@objective(model, Max, y)

optimize!(model)

print(model)
@show termination_status(model)
@show value.(x)
@show value.(y)

print("number of solutions ")
print(result_count(model))

solutions = Dict()
for i in 1:result_count(model)
    solutions[i] = value.(x, result = i)
end

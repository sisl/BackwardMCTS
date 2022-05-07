using JuMP
using HiGHS

# Model and vars
model = Model(HiGHS.Optimizer)
@variable(model, x[1:2])

# Constraints
@constraint(model, x[1] + 2*x[2] >= 10)
@constraint(model, x[1] - x[2] == 5)

# Slack variables/constraints
@variable(model, u[1:2])
for i in 1:2
    @constraint(model, x[i] <= u[i])
    @constraint(model, x[i] >= -u[i])
end

# Objective
@objective(model, Min, u[1] + 3*u[2])

optimize!(model)
print(model)
@show value.(x)
@show value.(u)
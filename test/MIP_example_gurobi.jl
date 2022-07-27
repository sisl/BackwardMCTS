using JuMP, Gurobi
print_solutions(var, model) = [JuMP.value.(var; result=si) for si in 1:result_count(model)]

model = Model(Gurobi.Optimizer)

@variable(model, x)
@variable(model, y)

@objective(model, Max, 500*x + 300*y)

@constraint(model, y>=0)

@constraint(model, x>=0)

@constraint(model, 15x + 5y <= 300)

@constraint(model, 10x + 6y <= 240)

@constraint(model, 8x + 12y <= 450)

# Constraint that the next solution should be far
# |x - b| ≤ eps
b = 2.5
eps = 100000
@constraint(model, x <= eps + b)
@constraint(model, -x <= eps - b)


optimize!(model)
result_count(model)

print_solutions(x, model)
print_solutions(y, model)






using JuMP, Gurobi
print_solutions(var, model) = [JuMP.value.(var; result=si) for si in 1:result_count(model)]

model = Model(Gurobi.Optimizer)

@variable(model, x)
@variable(model, y)
@variable(model, z1, Int)
@variable(model, z2, Int)

@constraint(model, y+z1 == z2)

@objective(model, Max, 500*x + 300*y)

@constraint(model, y>=0)

@constraint(model, x>=0)

@constraint(model, 15x + 5y <= 300)

@constraint(model, 10x + 6y <= 240)

@constraint(model, 8x + 12y <= 450)

set_optimizer_attribute(model, "PoolSearchMode", 2)
set_optimizer_attribute(model, "PoolSolutions", 10)

# set_optimizer_attribute(model, "PoolGap", 0.2)
# set_optimizer_attribute(model, "TimeLimit", 5)

optimize!(model)
result_count(model)

print_solutions(x, model)
print_solutions(y, model)
print_solutions(z1, model)
print_solutions(z2, model)




using JuMP, Gurobi
print_solutions(var, model) = [JuMP.value.(var; result=si) for si in 1:result_count(model)]

model = Model(Gurobi.Optimizer)

@variable(model, x)
@variable(model, y)

@variable(model, z, Int)
@constraint(model, z>=0)

@objective(model, Max, 500*x + 300*y - 00000001*z)

@constraint(model, y>=0)

@constraint(model, x>=0)

@constraint(model, 15x + 5y <= 300)

@constraint(model, 10x + 6y <= 240)

@constraint(model, 8x + 12y <= 450)

set_optimizer_attribute(model, "PoolSolutions", 3)
set_optimizer_attribute(model, "PoolSearchMode", 2)

optimize!(model)
result_count(model)

print_solutions(x, model)
print_solutions(y, model)
print_solutions(z, model)

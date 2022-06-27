#Copyright (c) 2021 James D Foster, and contributors                            #src
#                                                                                #src
# Permission is hereby granted, free of charge, to any person obtaining a copy   #src
# of this software and associated documentation files (the "Software"), to deal  #src
# in the Software without restriction, including without limitation the rights   #src
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell      #src
# copies of the Software, and to permit persons to whom the Software is          #src
# furnished to do so, subject to the following conditions:                       #src
#                                                                                #src
# The above copyright notice and this permission notice shall be included in all #src
# copies or substantial portions of the Software.                                #src
#                                                                                #src
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     #src
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       #src
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    #src
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         #src
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  #src
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  #src
# SOFTWARE.                                                                      #src

# _Author: James Foster (@jd-foster)_

# This tutorial demonstrates how to formulate and solve a combinatorial problem
# with multiple feasible solutions. In fact, we will see how to find _all_
# feasible solutions to our problem. We will also see how to enforce an
# "all-different" constraint on a set of integer variables.

# This post is in the same form as tutorials in the JuMP
# [documentation](https://jump.dev/JuMP.jl/stable/tutorials/Getting%20started/getting_started_with_JuMP/)
# but is hosted here since we depend on using some commercial solvers (Gurobi
# and CPLEX) that are not currently accomodated by the JuMP GitHub repository.

# ## Symmetric number squares

# Symmetric [number squares](https://www.futilitycloset.com/2012/12/05/number-squares/)
# and their sums often arise in recreational mathematics. Here are a few examples:
# ```
#    1 5 2 9       2 3 1 8        5 2 1 9
#    5 8 3 7       3 7 9 0        2 3 8 4
# +  2 3 4 0     + 1 9 5 6      + 1 8 6 7
# =  9 7 0 6     = 8 0 6 4      = 9 4 7 0
# ```

# Notice how all the digits 0 to 9 are used at least once,
# the first three rows sum to the last row,
# and the columns in each are the same as the corresponding rows (forming a symmetric matrix).

# We will answer the question: how many such squares are there?

# ### Model Specifics

# We start by creating a JuMP model:
using JuMP
model = Model();
number_of_digits = 4
PLACES = 0:(number_of_digits-1)

ROWS = 1:number_of_digits

@variable(model, 0 <= Digit[i = ROWS, j = PLACES] <= 9, Int)
@variable(model, Term[ROWS] >= 1, Int)

@constraint(model, NonZeroLead[i in ROWS], Digit[i, (number_of_digits-1)] >= 1)

@constraint(
    model,
    TermDef[i in ROWS],
    Term[i] == sum((10^j) * Digit[i, j] for j in PLACES)
)

@constraint(
    model,
    SumHolds,
    Term[number_of_digits] == sum(Term[i] for i in 1:(number_of_digits-1))
)


@constraint(
    model,
    Symmetry[i in ROWS, j in PLACES; i + j <= (number_of_digits - 1)],
    Digit[i, j] == Digit[number_of_digits-j, number_of_digits-i]
)

COMPS = [
    (i, j, k, m) for i in ROWS for j in PLACES for k in ROWS for m in PLACES
    if (
        i + j <= number_of_digits &&
        k + m <= number_of_digits &&
        (i, j) < (k, m)
    )
];

@variable(model, BinDiffs[COMPS], Bin);

@constraint(
    model,
    AllDiffLo[(i, j, k, m) in COMPS],
    Digit[i, j] <= Digit[k, m] - 1 + 42 * BinDiffs[(i, j, k, m)]
);

@constraint(
    model,
    AllDiffHi[(i, j, k, m) in COMPS],
    Digit[i, j] >= Digit[k, m] + 1 - 42 * (1 - BinDiffs[(i, j, k, m)])
);

import Gurobi
set_optimizer(model,Gurobi.Optimizer)

set_optimizer_attribute(model, "PoolSearchMode", 2)
set_optimizer_attribute(model, "PoolSolutions", 100)


optimize!(model)
# solution_summary(model)

# Let's check it worked:

@assert termination_status(model) == MOI.OPTIMAL
@assert primal_status(model) == MOI.FEASIBLE_POINT

value.(Digit)

# Note the display of `Digit` is reverse of the usual order.

# ### Viewing the Results

# Now that we have results, we can access the feasible solutions
# by using the `value` function with the `result` keyword:

TermSolutions = Dict()
for i in 1:result_count(model)
    TermSolutions[i] = convert.(Int64, round.(value.(Term; result = i).data))
end

a_feasible_solution = TermSolutions[1]

# and we can nicely print out all the feasible solutions with
function print_solution(x::Int)
    for i in (1000, 100, 10, 1)
        y = div(x, i)
        print(y, " ")
        x -= i * y
    end
    println()
    return
end

function print_solution(x::Vector)
    print("  ")
    print_solution(x[1])
    print("  ")
    print_solution(x[2])
    print("+ ")
    print_solution(x[3])
    print("= ")
    print_solution(x[4])
end

for i in 1:result_count(model)
    @assert has_values(model; result = i)
    println("Solution $(i): ")
    print_solution(TermSolutions[i])
    print("\n")
end

# The result is the full list of feasible solutions.
# So the answer to "how many such squares are there?" turns out to be 20.


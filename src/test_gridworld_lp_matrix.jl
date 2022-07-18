"""
This script formulates the LP the same way as "test_gridworld_lp.jl" but
explicitly creates the A,b,c matrix and vectors.
"""

include("gridworldpomdp.jl")
include("GridWorld_LP.jl")

using JuMP, HiGHS
using POMDPPolicies: solve
using QMDP

# Params
LP_Solver = HiGHS.Optimizer

# Create pomdp
pomdp = SimpleGridWorldPOMDP(size=(4,4),
                            rewards=Dict(GWPos(2,3)=>-10.0, GWPos(3,1)=>+25.0)
                            ,
                            tprob = 0.7,
                            oprob = 0.6)

tab_pomdp = tabulate(pomdp)
no_of_actions = length(actions(pomdp))
no_of_states = length(states(pomdp))

# Create α-vectors
solver = QMDPSolver()
policy = solve(solver, tab_pomdp)
Γ = policy.alphas

# Create leaf belief
β_t = zeros(no_of_states,)
β_t[4] = 0.45
β_t[5] = 0.55
obs_id = 5
a_star = αj = 4

# β_t[3] = 1.0
# obs_id = 3
# a_star = αj = 2

# Solve for root belief
O_bar = O = create_O_bar(tab_pomdp, obs_id)
T_bar = T = create_T_bar(tab_pomdp, a_star)


add_columns = hcat
add_rows = vcat

model = Model(LP_Solver)

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


no_of_LP_vars = 5 * no_of_states + 5
@variable(model, X[1 : no_of_LP_vars])


zr = zeros(1, no_of_states);
Zr = zeros(no_of_states, no_of_states);
vn = ones(1, no_of_states);
Vn = ones(no_of_states, no_of_states);
Eye = Diagonal(ones(no_of_states));

# Default constraints
A = [[vn*O*T zr -1 0 zr];
     [zr zr 1 -1 zr];
     [Eye Zr zr' zr' Eye];
     [vn zr 0 0 zr]]

eps_var = 1e-5
b = [0; eps_var; vn'; 1]



# Alpha-vector constraints
A = add_columns(A, zeros(size(A, 1), length(Γ)-1));

function add_alpha_constraints(A, Γ, αj)
    counter = 0
    for αk in 1:length(Γ)
        if (αk != αj)
            counter += 1
            
            temp = zeros(1, length(Γ)-1)
            temp[counter] = -1
            A = add_rows(A, [(Γ[αj]-Γ[αk])' zr 0 0 zr temp]);
            
        end
    end

    return A
end

A = add_alpha_constraints(A, Γ, αj);
b = add_rows(b, [0; 0; 0]);


# L1 norm constraints
A = add_columns(A, zeros(size(A, 1), 2*no_of_states));
A = add_rows(A, [-O*T -Eye β_t zr' Zr zr' zr' zr' Eye Zr]);
A = add_rows(A, [-O*T  Eye β_t zr' Zr zr' zr' zr' Zr -Eye]);
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



@assert rank(A) == size(A, 1)    # assert that A is full rank
@constraint(model, A*X .== b)
@constraint(model, X .>= 0)

c = zeros(1, no_of_LP_vars);
c[no_of_states+1 : 2*no_of_states] .= 1
@objective(model, Min, dot(c,X))

# print(model)
optimize!(model)
β_opt, J_opt = JuMP.value.(X)[1:no_of_states], JuMP.objective_value(model)

@show β_opt
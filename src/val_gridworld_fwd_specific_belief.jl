include("Validation_MCTS.jl")
include("utils.jl")

# Create pomdp
pomdp = SimpleGridWorldPOMDP(size=(6,6),
                            rewards=Dict(GWPos(2,3)=>-10.0, GWPos(3,1)=>+25.0)
                            ,
                            tprob = 0.9,
                            oprob = 0.9)

# Create α-vectors
solver = QMDPSolver()
policy = solve(solver, pomdp)
Γ = policy.alphas

epochs = 1_00_000
des_final_state = GWPos(3,1)

b0_testing = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.08401406019970141, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8321119210787887, 0.08387401872150986, 0.0, 0.0, 0.0, 0.0]
des_ao_traj = [(:right, 33), (:down, 27), (:down, 21), (:down, 15), (:down, 9), (:down, 3), (:end, -1)]

init_states = batch_fwd_simulations(pomdp, policy, epochs, des_final_state, b0_testing, convert_des_ao_traj(pomdp, des_ao_traj));
parsed_init_states = parse_batch_fwd_simulations(pomdp, init_states)
reshape_GW(parsed_init_states)
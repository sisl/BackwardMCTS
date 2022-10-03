include("Validation_MCTS.jl")

# Create pomdp
pomdp = SimpleGridWorldPOMDP(size=(4,4),
                            rewards=Dict(GWPos(2,3)=>-10.0, GWPos(3,1)=>+25.0)
                            ,
                            tprob = 0.7,
                            oprob = 0.8)

# Create α-vectors
solver = QMDPSolver()
policy = solve(solver, pomdp)
Γ = policy.alphas

epochs = 1_000_000
des_final_state = GWPos(3,1)

# # What BMCTS finds: 0.3290462262734401 
# max_t = 1
# b0_testing = [0.0, 0.0, 0.0, 0.0, 0.0, 0.07048470036865795, 0.5232565728017723, 0.134024400520869, 0.0, 0.0, 0.27223432630870076, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# des_ao_traj = [(:down, 3)]

# What BMCTS finds: 0.07151947914127925 
max_t = 2
b0_testing = [0.0, 0.0, 0.0, 0.011771534529397758, 0.36269950810188617, 0.0029749102326139297, 0.1967849922172984, 0.0094221794446463, 0.0, 0.0, 0.20815648796383224, 0.038911766090334, 0.0, 0.0, 0.16927862141999134, 0.0, 0.0]
des_ao_traj = [(:down, 7), (:down, 3)]

init_states = batch_fwd_simulations(pomdp, policy, max_t, epochs, des_final_state, b0_testing, convert_des_ao_traj(pomdp, des_ao_traj));

parsed_init_states = parse_batch_fwd_simulations(pomdp, init_states)
reshape_GW(parsed_init_states)
include("Validation_MCTS.jl")
include("utils.jl")

# Create pomdp
pomdp = SimpleGridWorldPOMDP(size=(6,6),
                            rewards=Dict(GWPos(2,3)=>-10.0, GWPos(3,1)=>+25.0)
                            ,
                            tprob = 0.90,
                            oprob = 0.90)

# Create α-vectors
solver = QMDPSolver()
policy = solve(solver, pomdp)
Γ = policy.alphas

epochs = 1_00_000
des_final_state = GWPos(3,1)

# # What BMCTS finds: 0.3290462262734401 
# b0_testing = [0.0, 0.0, 0.0, 0.0, 0.0, 0.07048470036865795, 0.5232565728017723, 0.134024400520869, 0.0, 0.0, 0.27223432630870076, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# des_ao_traj = [(:down, 3), (:end, -1)]

b0_testing =   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05306471083169063, 0.0, 0.0, 0.0, 0.010823847445793202, 0.08107450984761602, 0.03771581092374405, 0.7767758958272822, 0.012667956388807532, 0.02787726873506632, 0.0]
# b0_testing += 0.01*rand(17)
normalize!(b0_testing)
des_ao_traj =  [(:right, 15), (:down, 11), (:down, 7), (:down, 3), (:end, -1)]

# b0_testing = ones(17)
# b0_testing[14] = 99
# normalize!(b0_testing)
# des_ao_traj = [(:right, 15), (:down, 11), (:down, 7), (:down, 3), (:end, -1)]

# # What BMCTS finds: 0.07151947914127925 
# b0_testing = [0.0, 0.0, 0.0, 0.011771534529397758, 0.36269950810188617, 0.0029749102326139297, 0.1967849922172984, 0.0094221794446463, 0.0, 0.0, 0.20815648796383224, 0.038911766090334, 0.0, 0.0, 0.16927862141999134, 0.0, 0.0]
# des_ao_traj = [(:down, 7), (:down, 3), (:end, -1)]


b0_testing = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8930481283422448, 0.0, 0.10695187165775528, 0.0]
des_ao_traj = [(:right, 15), (:down, 11), (:down, 7), (:down, 3), (:end, -1)]

b0_testing = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06684116771500326, 0.0, 0.0004093658932861984, 0.0, 0.038951953324997436, 0.8937975130667132, 0.0, 0.0, 0.0]
des_ao_traj = [(:up, 14), (:up, 14), (:right, 15), (:down, 11), (:down, 7), (:down, 3), (:end, -1)]

b0_testing = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9227513227513225, 0.0772486772486775, 0.0, 0.0, 0.0]
des_ao_traj = [(:down, 27), (:down, 21), (:down, 15), (:down, 9), (:down, 3), (:end, -1)]

b0_testing = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04873231189873489, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8949163039102268, 0.056351384191038155, 0.0, 0.0, 0.0, 0.0]
des_ao_traj = [(:right, 33), (:down, 27), (:down, 21), (:down, 15), (:down, 9), (:down, 3), (:end, -1)]


init_states = batch_fwd_simulations(pomdp, policy, epochs, des_final_state, b0_testing, convert_des_ao_traj(pomdp, des_ao_traj));

parsed_init_states = parse_batch_fwd_simulations(pomdp, init_states)
reshape_GW(parsed_init_states)
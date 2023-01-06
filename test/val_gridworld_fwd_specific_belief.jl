include("Validation_MCTS.jl")
include("utils.jl")

# Create pomdp
pomdp = SimpleGridWorldPOMDP(size=(6,6),
                            rewards=Dict(GWPos(2,3)=>-10.0, GWPos(3,1)=>+25.0)
                            ,
                            tprob = 0.9,
                            oprob = 0.9)


epochs = 100_000
des_final_state = GWPos(3,1)

# BMCTS finds 1.80%
b0_testing = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.020265872796979507, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6943354257961931, 0.07091082155835422, 0.2144878798484732, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
des_ao_traj = [(:left, 19), (:right, 20), (:right, 21), (:down, 15), (:down, 9), (:down, 3), (:end, -1)]

b0_testing = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8598726114649682, 0.14012738853503187, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
des_ao_traj = [(:right, 21), (:down, 15), (:down, 9), (:down, 3), (:end, -1)]

b0_testing = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.005925033833010426, 0.0, 0.0, 0.0, 0.7700721941538006, 0.22400277201318897, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
des_ao_traj = [(:right, 20), (:right, 21), (:down, 15), (:down, 9), (:down, 3), (:end, -1)]

init_states, score = batch_fwd_simulations(pomdp, epochs, des_final_state, b0_testing, convert_des_ao_traj(pomdp, des_ao_traj));

parsed_init_states = parse_batch_fwd_simulations(pomdp, init_states)
reshape_GW(parsed_init_states)
using CSV
using DataFrames: DataFrame, nrow
using ProgressBars: tqdm
using Statistics: mean, median, std

include("carlopomdp.jl")

normalize_Func(A::AbstractArray{Float64, 3}) = A ./ sum(A, dims=1)
validate_Func(A::AbstractArray{Float64, 3}, eps=1e-5) = all(1.0 + eps .> sum(A, dims=1) .> 1.0 - eps)

function get_ego_action_from_row(pomdp, u)
    if u > 0
        return :go
    elseif u==0
        return :coast
    else
        return :brake
    end
end

function get_rival_vel(pomdp, vel)
    if first(vel) < pomdp.speed_limit
        return :within_limit
    else
        return :above_limit
    end
end

function normalize_rival_func!(Obs_or_Trans_Func_Rival)
    kys = collect(keys(Obs_or_Trans_Func_Rival))
    s_list = unique(getindex.(kys, Ref(2)))

    for s in s_list
        k = [item for item in kys if last(item)==s]
        N = sum(getd(Obs_or_Trans_Func_Rival, k))
        [Obs_or_Trans_Func_Rival[item] /= N for item in k];
    end
end


function normalize_ego_func!(Trans_Func_Ego)
    kys = collect(keys(Trans_Func_Ego))
    s_list = unique(getindex.(kys, Ref(2:3)))

    for s in s_list
        k = [item for item in kys if item[2:3]==s]
        N = sum(getd(Trans_Func_Ego, k))
        [Trans_Func_Ego[item] /= N for item in k];
    end
end

function get_rival_state_from_row(pomdp, x,y,start,final,vel)
    p = Point(x,y)
    rival_box = get_box_id(pomdp, p; agent=:rival)
    rival_vel = get_rival_vel(pomdp, vel)
    rival_itn = rival_intention(start,final)
    return (rival_box, rival_vel, rival_itn)
end

function get_ego_state_from_row(pomdp, x,y)
    p = Point(x,y)
    ego_box = get_box_id(pomdp, p; agent=:ego)
    return ego_box
end

# [-45, 45], [45, 135], [135, 225], [225, 315] --> 1, 2, 3, 4
get_obs_quadrant(angle_rad) = [:east, :north, :west, :south][Int(mod(angle_rad+pi/4, 2*pi) ÷ (pi/2) + 1)]

function get_rival_obs_from_row(pomdp, x,y,h,vel)
    p = Point(x,y)
    rival_box = get_box_id(pomdp, p; agent=:rival)
    rival_hdg = get_obs_quadrant(h)
    rival_vel = get_rival_vel(pomdp, vel)
    return (rival_box, rival_vel, rival_hdg)
end


function tabulate_learn(pomdp::CarloPOMDP; dir="../")
    num_states = length(states(pomdp))
    num_actions = length(actions(pomdp))
    num_obs = length(observations(pomdp))
    

    Trans_Func_Rival = DefaultDict(0)
    Obs_Func_Rival = DefaultDict(0)

    Trans_Func_Ego = DefaultDict(0)


    ### Learn from rivals ###
    files = readdir("$(dir)csvfiles_rival/"; join=true)

    for i = tqdm(1:length(files))
        df = DataFrame(CSV.File(files[i]))

        for r in 1:nrow(df)-1
            s  = get_rival_state_from_row(pomdp, df[r, :X_Position], df[r, :Y_Position], df[r, :StartDir], df[r, :FinalDir], L2_norm([df[r, :X_Velocity], df[r, :Y_Velocity]]))
            sp = get_rival_state_from_row(pomdp, df[r+1, :X_Position], df[r+1, :Y_Position], df[r+1, :StartDir], df[r+1, :FinalDir], L2_norm([df[r+1, :X_Velocity], df[r+1, :Y_Velocity]]))
            
            Trans_Func_Rival[(sp,s)] += 1
            
            o = get_rival_obs_from_row(pomdp, df[r+1, :X_Position], df[r+1, :Y_Position], df[r+1, :Heading], L2_norm([df[r+1, :X_Velocity], df[r+1, :Y_Velocity]]))
            Obs_Func_Rival[(o,sp)] += 1

            # if sp ==(1, :within_limit, :left)
            #     @show df
            # end
        end
    end


    ### Learn from egos ###
    files = readdir("$(dir)csvfiles_ego/"; join=true)

    for i = tqdm(1:length(files))
        df = DataFrame(CSV.File(files[i]))

        for r in 1:nrow(df)-1
            s  = get_ego_state_from_row(pomdp, df[r, :X_Position], df[r, :Y_Position])
            sp = get_ego_state_from_row(pomdp, df[r+1, :X_Position], df[r+1, :Y_Position])
            a  = get_ego_action_from_row(pomdp, df[r, :U_Throttle]) 
            Trans_Func_Ego[(sp,a,s)] += 1
        end
    end

    # return Trans_Func_Rival, Trans_Func_Ego, Obs_Func_Rival


    ### Combine both ###
    oprob = pomdp.oprob
    normalize_rival_func!(Trans_Func_Rival)
    normalize_rival_func!(Obs_Func_Rival)
    normalize_ego_func!(Trans_Func_Ego)
    

    Trans_Func = zeros(num_states, num_actions, num_states)  # |S|×|A|×|S|, T[s', a, s] = p(s'|a,s)
    Obs_Func = zeros(num_obs, num_actions, num_states)       # |O|×|A|×|S|, O[o, a, s'] = p(o|a,s')
    Reward_Func = zeros(num_states, num_actions)             # |S|×|A|,     R[s,a]




    for (ai, a) in enumerate(actions(pomdp))
        Reward_Func[:,ai] = reward.(Ref(pomdp), states(pomdp), Ref(a))
    end    

    for (ai, a) in enumerate(actions(pomdp))
        for (spi, sp) in enumerate(states(pomdp))

            for (oi, o) in enumerate(observations(pomdp))
                prob_itn = Obs_Func_Rival[((o.rival_box, o.rival_vel, o.rival_dir), (sp.rival_box, sp.rival_vel, sp.rival_dir))]

                ego_neighbors = get_box_neighbors(sp.ego_box; ego=true)
                if o.ego_box == sp.ego_box
                    prob_ego_pos = oprob
                elseif o.ego_box in ego_neighbors
                    prob_ego_pos = (1.0-oprob)/length(ego_neighbors)
                else
                    prob_ego_pos = 0.0
                end

                rival_neighbors = get_box_neighbors(sp.rival_box)
                if o.rival_box == sp.rival_box
                    prob_rival_pos = oprob
                elseif o.rival_box in rival_neighbors
                    prob_rival_pos = (1.0-oprob)/length(rival_neighbors)
                else
                    prob_rival_pos = 0.0
                end
                
                Obs_Func[oi,ai,spi] += prob_ego_pos * prob_rival_pos * prob_itn
            end

            for (si, s) in enumerate(states(pomdp))
                Trans_Func[spi,ai,si] = Trans_Func_Rival[((sp.rival_box, sp.rival_vel, sp.rival_dir), (s.rival_box, s.rival_vel, s.rival_dir))] * Trans_Func_Ego[(sp.ego_box, a, s.ego_box)]
            end
        end
    end

    Obs_Func = normalize_Func(Obs_Func)
    # return (Trans_Func, Reward_Func, Obs_Func, Trans_Func_Rival, Trans_Func_Ego, Obs_Func_Rival)
    return (Trans_Func, Reward_Func, Obs_Func)
end

function tabulate(pomdp::CarloPOMDP; dir="../")
    Trans_Func, Reward_Func, Obs_Func = tabulate_learn(pomdp; dir=dir)
    return global LLTAB = TabularPOMDP(Trans_Func, Reward_Func, Obs_Func, pomdp.discount);
end


# Overload T(s'|a,s) from the learned data
function POMDPs.transition(pomdp::CarloPOMDP, s::CarloDiscreteState, a::Symbol)
    si = POMDPs.stateindex(pomdp, s)
    ai = POMDPs.actionindex(pomdp, a)

    destinations = states(pomdp)
    probs = LLTAB.T[:,ai,si]
    return SparseCat(destinations, probs)
end

# Overload O(s'|a) from the learned data
function POMDPs.observation(pomdp::CarloPOMDP, sp::CarloDiscreteState)
    spi = POMDPs.stateindex(pomdp, sp)

    destinations = observations(pomdp)
    probs = LLTAB.O[:,1,spi]
    return SparseCat(destinations, probs)
end
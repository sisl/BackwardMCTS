using POMDPs
using POMDPModels: TabularPOMDP
using POMDPModelTools: SparseCat, Deterministic
using StaticArrays
using Parameters
using Random
using Distributions

include("utils.jl")
include("CarloCar_dynamics.jl")

mutable struct CarloDiscreteState
    ego_box::Int
    rival_box::Int
    rival_vel::Symbol
    rival_dir::Symbol
end

@with_kw struct CarloPOMDP <: POMDP{CarloDiscreteState, Symbol, CarloDiscreteState}
    dt::Float64 = 0.1
    oprob::Float64 = 0.9
    discount::Float64 = 0.95
    speed_limit::Float64 = 9999.0  # [m/s]   # TODO: Change this back to 7 m/s

    # Ego goes from south to west
    # ego_box_space::AbstractArray = union(get_box_centroids(bottom_left=[60,50], top_right=[65,65], spacing=5), get_box_centroids(bottom_left=[50,60], top_right=[65,65], spacing=5))
    ego_box_idxs = [4,8,6,5,2]
    ego_box_space = get_boxes(ego_box_idxs)
    
    # Rival can start/end any direction
    rival_box_idxs = collect(1:8)
    rival_box_space = get_boxes(rival_box_idxs)
end

### States ###

mutable struct Point
    x::Number
    y::Number
end

struct Box
    lower_left::Point
    upper_right::Point
end

function rival_intention(start, final)
    straight = [("south", "north"),
                ("east", "west"),
                ("north", "south"),
                ("west", "east")]

    right = [("east", "north"), 
             ("north", "west"), 
             ("west", "south"), 
             ("south", "east")]
                
    left = [("east", "south"), 
            ("south", "west"), 
            ("west", "north"), 
            ("north", "east")]
                
    return first([:straight, :right, :left][in.(Ref((start, final)), [straight, right, left])])
end

function get_boxes(idxs)
    dict = Dict(1 => Box(Point(55,65), Point(65,70)),
                2 => Box(Point(50,55), Point(55,65)),
                3 => Box(Point(65,55), Point(70,65)),
                4 => Box(Point(55,50), Point(65,55)),
                5 => Box(Point(55,60), Point(60,65)),
                6 => Box(Point(60,60), Point(65,65)),
                7 => Box(Point(55,55), Point(60,60)),
                8 => Box(Point(60,55), Point(65,60)))

    return getd(dict, idxs)
end

function is_point_in_box(box::Box, point::Point)
    return (box.lower_left.y <= point.y <= box.upper_right.y) && (box.lower_left.x <= point.x <= box.upper_right.x)
end

function distance_to_box(box::Box, p::Point)
    dist = 0.5 .* [(box.lower_left.x + box.upper_right.x), (box.lower_left.y + box.upper_right.y)] - [p.x, p.y]
    return L2_norm(dist)
end

function get_box_id(pomdp, p::Point; agent=:ego)
    if agent == :ego
        return pomdp.ego_box_idxs[argmin(distance_to_box.(pomdp.ego_box_space, Ref(p)))]
    elseif agent == :rival
        return pomdp.rival_box_idxs[argmin(distance_to_box.(pomdp.rival_box_space, Ref(p)))]
    end
end

function get_box_centroids(;bottom_left=[50,50], top_right=[70,70], spacing=5, discard_corners=0)
    (width, height) = top_right - bottom_left
    square_size = min(width, height) / floor(min(width, height) / spacing)
    rows = Int(floor(height ÷ square_size))
    cols = Int(floor(width ÷ square_size))
    x_values = bottom_left[1] .+ range(square_size/2; length=cols, stop=width-square_size/2)
    y_values = bottom_left[2] .+ range(square_size/2, length=rows, stop=height-square_size/2)
    centroids = [(x, y) for y in y_values, x in x_values]
    if discard_corners > 0
        to_discard = [CartesianIndex(i,j) for i in [(1:discard_corners)..., (rows-discard_corners+1:rows)...] for j in [(1:discard_corners)..., (cols-discard_corners+1:cols)...]]
        to_keep = setdiff(CartesianIndices((1:rows, 1:cols)), to_discard)
        return sort(centroids[to_keep])
    else
        return sort(vec(centroids))
    end
end

function get_box_neighbors(box_int::Int; ego=false)
    if ego
        dict = Dict(2 => [5],
                    4 => [8],
                    5 => [2,6],
                    6 => [5,8],
                    8 => [4,6])

    else
        dict = Dict(1 => [5,6],
                    2 => [5,7],
                    3 => [6,8],
                    4 => [7,8],
                    5 => [1,2,6,7,8],
                    6 => [1,3,5,7,8],
                    7 => [2,4,5,6,8],
                    8 => [3,4,5,6,7])

    end
    return dict[box_int]
end


### States ###

function POMDPs.states(pomdp::CarloPOMDP)
    ego = [4, 8, 6, 5, 2]
    rival = collect(1:8)
    rival_vel = [:within_limit]   # TODO: Change this back to [:within_limit, :above_limit]
    rival_itn = [:left, :right, :straight]
    results = Iterators.product([ego, rival, rival_vel, rival_itn]...) |> collect |> vec
    return [CarloDiscreteState(item...) for item in results]
end

POMDPs.stateindex(pomdp::CarloPOMDP, s::CarloDiscreteState) = findfirst(x->flatten(x)==flatten(s), POMDPs.states(pomdp))

flatten(t::CarloDiscreteState) = [t.ego_box, t.rival_box, t.rival_vel, t.rival_dir]

# Enforce equality of two CarloDiscreteState structs when their entries are identical.
Base.:(==)(t1::CarloDiscreteState, t2::CarloDiscreteState) = (t1.ego_box == t2.ego_box) && (t1.rival_box == t2.rival_box) && (t1.rival_vel == t2.rival_vel) && (t1.rival_dir == t2.rival_dir)
Base.isequal(t1::CarloDiscreteState, t2::CarloDiscreteState) = (t1.ego_box == t2.ego_box) && (t1.rival_box == t2.rival_box) && (t1.rival_vel == t2.rival_vel) && (t1.rival_dir == t2.rival_dir)
Base.hash(t::CarloDiscreteState) = Base.hash(flatten(t))



### Actions ###

POMDPs.actions(pomdp::CarloPOMDP) = (:brake, :coast, :go)

POMDPs.actionindex(pomdp::CarloPOMDP, a::Symbol) = findfirst(x->x==a, POMDPs.actions(pomdp))


### Transitions ###

function POMDPs.isterminal(p::CarloPOMDP, s::CarloDiscreteState) 
    return s.ego_box == s.rival_box || s.ego_box == 2
end

function POMDPs.transition(pomdp::CarloPOMDP, s::CarloDiscreteState, a::Symbol)
    @warn "Not supposed to have landed here (POMDPs.transition)"
    return nothing   # Will be overwritten from learned from data
end


### Observations ###

function POMDPs.observations(pomdp::CarloPOMDP)
    ego = [4, 8, 6, 5, 2]
    rival = collect(1:8)
    rival_vel = [:within_limit, :above_limit]
    rival_hdg = [:east, :north, :west, :south]
    results = Iterators.product([ego, rival, rival_vel, rival_hdg]...) |> collect |> vec
    return [CarloDiscreteState(item...) for item in results]
end


function POMDPs.observation(pomdp::CarloPOMDP, sp::CarloDiscreteState)
    @warn "Not supposed to have landed here (POMDPs.observation)"
    return nothing   # Will be overwritten from learned from data
end

POMDPs.obsindex(pomdp::CarloPOMDP, o::CarloDiscreteState) = findfirst(x->flatten(x)==flatten(o), POMDPs.observations(pomdp))


### Rewards ###


function POMDPs.reward(pomdp::CarloPOMDP, s::CarloDiscreteState)
    
    good_reward = +100
    bad_reward  = -1000
    
    if s.ego_box == s.rival_box
        return bad_reward
    elseif s.ego_box == 2
        return good_reward
    else
        return 0.0
    end
end
    
POMDPs.reward(pomdp::CarloPOMDP, s::CarloDiscreteState, a::Symbol) = reward(pomdp, s)


### Discount ###

POMDPs.discount(pomdp::CarloPOMDP) = pomdp.discount


# Get sink leaf belief, given its location in the grid world.
function get_leaf_belief(pomdp::CarloPOMDP, final_state)
    si = POMDPs.stateindex(pomdp, final_state)
    no_of_states = length(states(pomdp))
    β_final = zeros(no_of_states,)
    β_final[si] = 1.0
    return β_final
end
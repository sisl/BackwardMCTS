using POMDPs
using POMDPModels: SimpleGridWorld, GWPos, TabularPOMDP
using POMDPModelTools: SparseCat, Deterministic
using StaticArrays
using Parameters
using Random
using Distributions

"""
    SimpleGridWorldPOMDP(;kwargs...)

Create a simple grid world POMDP. This is the partially observable version of the `POMDPModels.SimpleGridWorld` MDP. We are observing the true state of the agent with an addition of observation noise. Options are specified with keyword arguments.

# States and Actions
The states are represented by 2-element static vectors of integers. Typically any Julia `AbstractVector` e.g. `[x,y]` can also be used for arguments. Actions are the symbols `:up`, `:left`, `:down`, and `:right`.

# Keyword Arguments
- `size::Tuple{Int, Int}`: Number of cells in the x and y direction [default: `(10,10)`]
- `rewards::Dict`: Dictionary mapping cells to the reward in that cell, e.g. `Dict([1,2]=>10.0)`. Default reward for unlisted cells is 0.0
- `terminate_from::Set`: Set of cells from which the problem will terminate. Note that these states are not themselves terminal, but from these states, the next transition will be to a terminal state. [default: `Set(keys(rewards))`]
- `tprob::Float64`: Probability of a successful transition in the direction specified by the action. The remaining probability is divided between the other neighbors. [default: `0.7`]
- `oprob::Float64`: Probability of a successful observation of the next state. The remaining probability is divided between the other neighbors. [default: `0.6`]
- `discount::Float64`: Discount factor [default: `0.95`]
"""
@with_kw struct SimpleGridWorldPOMDP <: POMDP{GWPos, Symbol, GWPos}
    size::Tuple{Int, Int}           = (10,10)
    rewards::Dict{GWPos, Float64}   = Dict(GWPos(4,3)=>-10.0, GWPos(4,6)=>-5.0, GWPos(9,3)=>10.0, GWPos(8,8)=>3.0)
    terminate_from::Set{GWPos}      = Set(keys(rewards))
    tprob::Float64                  = 0.7
    oprob::Float64                  = 0.6
    discount::Float64               = 0.95
end

"""
Converts a `SimpleGridWorldPOMDP` object to a `POMDPModels.TabularPOMDP` object.
Useful for offline policy creation, and policy validation.
"""
function tabulate(pomdp::SimpleGridWorldPOMDP)
    num_states = length(states(pomdp))
    num_actions = length(actions(pomdp))
    num_obs = length(observations(pomdp))
    
    Trans_Func = zeros(num_states, num_actions, num_states)  # |S|×|A|×|S|, T[s', a, s] = p(s'|a,s)
    Obs_Func = zeros(num_obs, num_actions, num_states)       # |O|×|A|×|S|, O[o, a, s'] = p(o|a,s')
    Reward_Func = zeros(num_states, num_actions)             # |S|×|A|,     R[s,a]

    for (ai, a) in enumerate(actions(pomdp))
        Reward_Func[:,ai] = reward.(Ref(pomdp), states(pomdp), Ref(a))
        for (si, s) in enumerate(states(pomdp))
            Trans_Func[:,ai,si] = pdf.(Ref(transition(pomdp, s, a)), states(pomdp))
        end
    end

    for (ai, _) in enumerate(actions(pomdp))
        for (spi, sp) in enumerate(states(pomdp))
            Obs_Func[:,ai,spi] = pdf.(Ref(observation(pomdp, sp)), observations(pomdp))
        end
    end

    return TabularPOMDP(Trans_Func, Reward_Func, Obs_Func, pomdp.discount);
end


### States ###

function POMDPs.states(pomdp::SimpleGridWorldPOMDP)
    ss = vec(GWPos[GWPos(x, y) for x in 1:pomdp.size[1], y in 1:pomdp.size[2]])
    push!(ss, GWPos(-1,-1))
    return ss
end

function POMDPs.stateindex(pomdp::SimpleGridWorldPOMDP, s::AbstractVector{Int})
    if all(s.>0)
        return LinearIndices(pomdp.size)[s...]
    else
        return prod(pomdp.size) + 1 
    end
end

struct GWUniform
    size::Tuple{Int, Int}
end
Base.rand(rng::AbstractRNG, d::GWUniform) = GWPos(rand(rng, 1:d.size[1]), rand(rng, 1:d.size[2]))
function POMDPs.pdf(d::GWUniform, s::GWPos)
    if all(1 .<= s[1] .<= d.size)
        return 1/prod(d.size)
    else
        return 0.0
    end
end
POMDPs.support(d::GWUniform) = (GWPos(x, y) for x in 1:d.size[1], y in 1:d.size[2])

POMDPs.initialstate(pomdp::SimpleGridWorldPOMDP) = GWUniform(pomdp.size)


### Actions ###

POMDPs.actions(pomdp::SimpleGridWorldPOMDP) = (:up, :down, :left, :right)
Base.rand(rng::AbstractRNG, t::NTuple{L,Symbol}) where L = t[rand(rng, 1:length(t))] # don't know why this doesn't work out of the box


const GWdir = Dict(:up=>GWPos(0,1), :down=>GWPos(0,-1), :left=>GWPos(-1,0), :right=>GWPos(1,0))
const GWaind = Dict(:up=>1, :down=>2, :left=>3, :right=>4)

POMDPs.actionindex(pomdp::SimpleGridWorldPOMDP, a::Symbol) = GWaind[a]


### Transitions ###

POMDPs.isterminal(p::SimpleGridWorldPOMDP, s::AbstractVector{Int}) = any(s.<0)

function POMDPs.transition(pomdp::SimpleGridWorldPOMDP, s::AbstractVector{Int}, a::Symbol)
    if s in pomdp.terminate_from || isterminal(pomdp, s)
        return Deterministic(GWPos(-1,-1))
    end

    destinations = MVector{length(actions(pomdp))+1, GWPos}(undef)
    destinations[1] = s

    probs = @MVector(zeros(length(actions(pomdp))+1))
    for (i, act) in enumerate(actions(pomdp))
        if act == a
            prob = pomdp.tprob # probability of transitioning to the desired cell
        else
            prob = (1.0 - pomdp.tprob)/(length(actions(pomdp)) - 1) # probability of transitioning to another cell
        end

        dest = s + GWdir[act]
        destinations[i+1] = dest

        if !inbounds(pomdp, dest) # hit an edge and come back
            probs[1] += prob
            destinations[i+1] = GWPos(-1, -1) # dest was out of bounds - this will have probability zero, but it should be a valid state
        else
            probs[i+1] += prob
        end
    end

    return SparseCat(destinations, probs)
end

function inbounds(p::SimpleGridWorldPOMDP, s::AbstractVector{Int})
    return 1 <= s[1] <= p.size[1] && 1 <= s[2] <= p.size[2]
end


### Observations ###

POMDPs.observations(pomdp::SimpleGridWorldPOMDP) = POMDPs.states(pomdp)

function POMDPs.observation(pomdp::SimpleGridWorldPOMDP, sp::AbstractVector{Int})
    destinations = MVector{length(actions(pomdp))+1, GWPos}(undef)
    destinations[1] = sp

    probs = @MVector(zeros(length(actions(pomdp))+1))
    probs[1] = pomdp.oprob # probability of observing correctly

    for (i, act) in enumerate(actions(pomdp))
        prob = (1.0 - pomdp.oprob)/(length(actions(pomdp))) # probability of observing a neighbor cell

        dest = sp + GWdir[act]
        destinations[i+1] = dest

        if !inbounds(pomdp, dest) # hit an edge and come back
            probs[1] += prob
            destinations[i+1] = GWPos(-1, -1) # dest was out of bounds - this will have probability zero, but it should be a valid state
        else
            probs[i+1] += prob
        end
    end

    return SparseCat(destinations, probs)
end


### Rewards ###

POMDPs.reward(pomdp::SimpleGridWorldPOMDP, s::AbstractVector{Int}) = get(pomdp.rewards, s, 0.0)
POMDPs.reward(pomdp::SimpleGridWorldPOMDP, s::AbstractVector{Int}, a::Symbol) = reward(pomdp, s)


### Discount ###

POMDPs.discount(pomdp::SimpleGridWorldPOMDP) = pomdp.discount


### Conversion ###

function POMDPs.convert_a(::Type{V}, a::Symbol, p::SimpleGridWorldPOMDP) where {V<:AbstractArray}
    convert(V, [GWaind[a]])
end
function POMDPs.convert_a(::Type{Symbol}, vec::V, pomdp::SimpleGridWorldPOMDP) where {V<:AbstractArray}
    actions(pomdp)[convert(Int, first(vec))]
end


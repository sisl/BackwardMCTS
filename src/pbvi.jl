"""
    PBVISolver <: Solver

Options dictionary for Point-Based Value Iteration for POMDPs.

# Fields
- `max_iterations::Int64` the maximal number of iterations the solver runs. Default: 10
- `improve_iterations::Int64` the maximal number of iterations to improve alpha vectors. Default: 100
- `ϵ::Float64` the maximal gap between alpha vector improve steps. Default = 0.01
- `verbose::Bool` switch for solver text output. Default: false
"""

using POMDPs
using POMDPModelTools
using BeliefUpdaters
# using POMDPLinter
using LinearAlgebra: dot
using Distributions
# using FiniteHorizonPOMDPs
using POMDPPolicies: AlphaVectorPolicy

import POMDPs: Solver, solve
import Base: ==, hash, convert
# import FiniteHorizonPOMDPs: InStageDistribution, FixedHorizonPOMDPWrapper


struct PBVISolver <: Solver
    max_iterations::Int64
    improve_iterations::Int64
    ϵ::Float64
    verbose::Bool
end

function PBVISolver(;max_iterations::Int64=10, improve_iterations::Int64=100, ϵ::Float64=0.01, verbose::Bool=false)
    return PBVISolver(max_iterations, improve_iterations, ϵ, verbose)
end

"""
    AlphaVec

Pair of alpha vector and corresponding action.

# Fields
- `alpha` α vector
- `action` action corresponding to α vector
"""
struct AlphaVec
    alpha::Vector{Float64}
    action::Any
end

==(a::AlphaVec, b::AlphaVec) = (a.alpha,a.action) == (b.alpha, b.action)
Base.hash(a::AlphaVec, h::UInt) = hash(a.alpha, hash(a.action, h))

function _argmax(f, X)
    return X[argmax(map(f, X))]
end

# adds probabilities of terminals in b to b′ and normalizes b′
function belief_norm(pomdp, b, b′, terminals, not_terminals)
    if sum(b′[not_terminals]) != 0.
        if !isempty(terminals)
            b′[not_terminals] = b′[not_terminals] / (sum(b′[not_terminals]) / (1. - sum(b[terminals]) - sum(b′[terminals])))
            b′[terminals] += b[terminals]
        else
            b′[not_terminals] /= sum(b′[not_terminals])
        end
    else
        b′[terminals] += b[terminals]
        b′[terminals] /= sum(b′[terminals])
    end
    return b′
end

# Backups belief with α vector maximizing dot product of itself with belief b
function backup_belief(pomdp::POMDP, Γ, b)
    S = ordered_states(pomdp)
    A = ordered_actions(pomdp)
    O = ordered_observations(pomdp)
    γ = discount(pomdp)
    r = StateActionReward(pomdp)

    Γa = Vector{Vector{Float64}}(undef, length(A))

    not_terminals = [stateindex(pomdp, s) for s in S if !isterminal(pomdp, s)]
    terminals = [stateindex(pomdp, s) for s in S if isterminal(pomdp, s)]
    for a in A
        Γao = Vector{Vector{Float64}}(undef, length(O))
        trans_probs = dropdims(sum([pdf(transition(pomdp, S[is], a), sp) * b.b[is] for sp in S, is in not_terminals], dims=2), dims=2)
        if !isempty(terminals) trans_probs[terminals] .+= b.b[terminals] end

        for o in O
            # update beliefs
            obs_probs = pdf.(map(sp -> observation(pomdp, a, sp), S), [o])
            b′ = obs_probs .* trans_probs

            if sum(b′) > 0.
                b′ = DiscreteBelief(pomdp, b.state_list, belief_norm(pomdp, b.b, b′, terminals, not_terminals))
            else
                b′ = DiscreteBelief(pomdp, b.state_list, zeros(length(S)))
            end

            # extract optimal alpha vector at resulting belief
            Γao[obsindex(pomdp, o)] = _argmax(α -> dot(α, b′.b), Γ)
        end

        # construct new alpha vectors
        Γa[actionindex(pomdp, a)] = [r(s, a) + (!isterminal(pomdp, s) ? (γ * sum(pdf(transition(pomdp, s, a), sp) * pdf(observation(pomdp, s, a, sp), o) * Γao[i][j]
                                        for (j, sp) in enumerate(S), (i, o) in enumerate(O))) : 0.)
                                        for s in S]
    end

    # find the optimal alpha vector
    idx = argmax(map(αa -> dot(αa, b.b), Γa))
    alphavec = AlphaVec(Γa[idx], A[idx])

    return alphavec
end

# Iteratively improves α vectors until the gap between steps is lesser than ϵ
function improve(pomdp, B, Γ, solver)
    alphavecs = nothing
    for _ in range(1, length=solver.improve_iterations)
        Γold = Γ
        alphavecs = [backup_belief(pomdp, Γold, b) for b in B]
        Γ = [alphavec.alpha for alphavec in alphavecs]
        # @show zip(Γold, Γ, B)
        prec = max([sum(abs.(dot(α1, b.b) .- dot(α2, b.b))) for (α1, α2, b) in zip(Γold, Γ, B)]...)
        
        # @show Γold
        # @show Γ
        # @show B
        # prec = max([sum(abs.(dot(Γold[i], B[i].b) .- dot(Γ[i], B[i].b))) for i=1:length(Γ)]...)
        if solver.verbose println("    Improving alphas, maximum gap between old and new α vector: $(prec)") end
        prec > solver.ϵ || break
    end

    return Γ, alphavecs
end

# Returns all possible, not yet visited successors of current belief b
function successors(pomdp, b, Bs)
    S = ordered_states(pomdp)
    not_terminals = [stateindex(pomdp, s) for s in S if !isterminal(pomdp, s)]
    terminals = [stateindex(pomdp, s) for s in S if isterminal(pomdp, s)]
    succs = []

    for a in actions(pomdp)
        trans_probs = dropdims(sum([pdf(transition(pomdp, S[is], a), sp) * b[is] for sp in S, is in not_terminals], dims=2), dims=2)
        if !isempty(terminals) trans_probs[terminals] .+= b[terminals] end

        for o in observations(pomdp)
            #update belief
            obs_probs = pdf.(map(sp -> observation(pomdp, a, sp), S), [o])
            b′ = obs_probs .* trans_probs


            if sum(b′) > 0.
                b′ = belief_norm(pomdp, b, b′, terminals, not_terminals)

                if !in(b′, Bs)
                    push!(succs, b′)
                end
            end
        end
    end

    return succs
end

# Computes distance of successor to the belief vectors in belief space
function succ_dist(pomdp, bp, B)
    dist = [norm(bp - b.b, 1) for b in B]
    return max(dist...)
end

# Expands the belief space with the most distant belief vector
# Returns new belief space, set of belifs and early termination flag
function expand(pomdp, B, Bs)
    B_new = copy(B)
    for b in B
        succs = successors(pomdp, b.b, Bs)
        if length(succs) > 0
            b′ = succs[argmax([succ_dist(pomdp, bp, B) for bp in succs])]
            push!(B_new, DiscreteBelief(pomdp, b′))
            push!(Bs, b′)
        end
    end

    return B_new, Bs, length(B) == length(B_new)
end

# 1: B ← {b0}
# 2: while V has not converged to V∗ do
# 3:    Improve(V, B)
# 4:    B ← Expand(B)
function solve(solver::PBVISolver, pomdp::POMDP, init_belief_set=nothing, init_Γ=nothing)
    S = ordered_states(pomdp)
    A = ordered_actions(pomdp)
    γ = discount(pomdp)
    r = StateActionReward(pomdp)

    # init belief, if given distribution, convert to vector
    if isnothing(init_belief_set)
        init = initialize_belief(DiscreteUpdater(pomdp), initialstate(pomdp))
        B = [init]
        Bs = Set([init.b])
        return (B, Bs)
    else
        Bs = init_belief_set
        B = [DiscreteBelief(pomdp, states(pomdp), bs) for bs in Bs]
    end
    
    # best action worst state lower bound
    if isnothing(init_Γ)
        α_init = 1 / (1 - γ) * maximum(minimum(r(s, a) for s in S) for a in A)
        Γ = [fill(α_init, length(S)) for _ in 1:length(Bs)]
    else
        Γ = init_Γ
    end

    if solver.verbose println("Running PBVI solver on $(typeof(pomdp)) problem with following settings:\n    max_iterations = $(solver.max_iterations), ϵ = $(solver.ϵ), verbose = $(solver.verbose)\n+----------------------------------------------------------+") end

    # original code should run until V converges to V*, this yet needs to be implemented
    # for example as: while max(@. abs(newV - oldV)...) > solver.ϵ
    # However this probably would not work, as newV and oldV have different number of elements (arrays of alphas)
    alphavecs = nothing
    Bs_old_len = length(Bs)
    for i in 1:solver.max_iterations
        # @show Bs_old_len
        append!(Γ, [fill(α_init, length(S)) for _ in range(1, length=length(Bs)-Bs_old_len)])
        Bs_old_len = length(Bs)
        
        # @show Bs
        # @show Γ
        @assert length(Bs)==length(Γ)
        Γ, alphavecs = improve(pomdp, B, Γ, solver)
        B, Bs, early_term = expand(pomdp, B, Bs)
        if solver.verbose println("Iteration $(i) executed, belief set contains $(length(Bs)) belief vectors.") end
        if early_term
            if solver.verbose println("Belief space did not expand. \nTerminating early.") end
            break
        end
    end

    if solver.verbose println("+----------------------------------------------------------+") end
    # acts = [actionindex(pomdp, alphavec.action) for alphavec in alphavecs]   # returns indices (Int) of actions.
    acts = [alphavec.action for alphavec in alphavecs]
    # return (Γ, acts, alphavecs)
    return AlphaVectorPolicy(pomdp, Γ, acts)
end

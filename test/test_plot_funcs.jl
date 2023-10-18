# using Plots


function get_optimal_actions_matrix(pomdp, policy)
    Γ = policy.alphas
    no_of_states = length(states(pomdp))

    optimal_actions = zeros(no_of_states,)
    for i in 1:no_of_states
        belief = zeros(no_of_states,)
        belief[i] = 1.0
    
        act = argmax([dot(α, belief) for α in Γ])
        optimal_actions[i] = act
    end
    
    N = Int(sqrt(no_of_states-1))
    # @show states_matrix = reshape_GW(states(pomdp))
    # @show rewards_matrix = reshape_GW(tab_pomdp.R[:,1])
    optimal_actions_matrix = map(x->actions(pomdp)[x], reshape_GW(optimal_actions))
    
    return optimal_actions_matrix
end
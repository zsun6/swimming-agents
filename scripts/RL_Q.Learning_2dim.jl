# """
# init work: Emad
# spruced:   Nate
# """
using POMDPs, POMDPModelTools, QuickPOMDPs
using POMDPSimulators
using DiscreteValueIteration, TabularTDLearning
using POMDPPolicies
using Random 
using Parameters


# define State space.
struct State
    x::Int
    y::Int
    #are we in a terminal state
    done::Bool
end
#Default state constructor
State(x::Int,y::Int) = State(x,y,false)

@with_kw struct GridWorldParameters
    size::Tuple{Int,Int} = (10, 10) #tuple with the grid size.
    null_state::State = (-1, -1) #state that outside the grid.
    p_transition::Real = 0.7 #probabilities of transition to the next state.
end

# Had defaults for the constructor, but we can force values too
Params = GridWorldParameters((20, 10),State(-1,-1), 0.7)

null = State(-1, -1)

# a Vector of all existing states, with one additional state(-1,-1) outside the GridWorld.
# S = [[State(x, y) for x = 1:10, y = 1:10]..., null]
# Now we can use our params to programmatically set up our state-space
S = [[State(x, y) for x=1:Params.size[1], y=1:Params.size[2]]...,null]

# compare two states based on their x and y.
Base.:(==)(s1::State, s2::State) = (s1.x == s2.x) && (s1.y == s2.y)

# define Action.
@enum Action UP DOWN LEFT RIGHT

# Action space.
A = [UP, DOWN, LEFT, RIGHT]

begin
    const MOVEMENTS = Dict(UP => State(0,1),
                    DOWN => State(0,-1),
                    LEFT => State(-1,0), 
                    RIGHT => State(1,0));
    # adding twe states.(not necessary Just for convenient)                                        
    Base.:+(s1::State, s2::State) = State(s1.x + s2.x, s1.y + s2.y) 
end
#helper function
function inbounds(p::GridWorldParameters,x::Int,y::Int)
    if 1 <= x <= p.size[1] && 1 <= y <= p.size[2]
        return true
    else
        return false
    end
end

function inbounds(p::GridWorldParameters,s::State)
    if 1 <= s.x <= p.size[1] && 1 <= s.y <= p.size[2]
        return true
    else
        return false
    end
end



# #transition function
function T(s::State, a::Action)
    # Deterministic() from POMDPModelTools.jl
    if R(s) != 0
        return Deterministic(null)# in this case we out of the environment.
    end

    len_a = length(A)
    next_states = Vector{State}(undef, len_a + 1)
    # calculating the probabilitie to the correct transition state.
    probabilities = zeros(len_a + 1) 

    for (index, a_prime) in enumerate(A)
        # prob = (a_prime == a) ? 0.7 : 0.1
        prob = (a_prime == a) ? Params.p_transition : 0.1
        dest = s + MOVEMENTS[a_prime]
        
        next_states[index + 1] = dest
        # the wall transition
        if !inbounds(Params,dest)
            probabilities[index + 1] = 0
        else #if 1 <= dest.x <= 10 && 1 <= dest.y <= 10 
            probabilities[index + 1] += prob
        end
    end
     # handle out-of-bounds transitions
     next_states[1] = s
     probabilities[1] = 1 - sum(probabilities)
     return SparseCat(next_states, probabilities)
 end

 #Reward Function
 function R(s,a=missing)
    if s == State(4,3)
        return -10
    elseif s == State(4,6)
        return -5
    elseif s == State(9,3)
        return 10
    elseif s == State(8,8)
        return 3
    elseif s == State(11,2) || s == State(11,3) || s == State(11,4) || s == State(11,5) || s == State(11,6) || s == State(11,7) ||
        s == State(11,8) || s == State(11,9)
        return -10
    elseif s == State( 17, 5)
        return 250
    elseif s == State( 16, 6) || s == State( 17, 6) || s == State( 18, 6) || s == State( 18, 5)||
        s == State( 18, 4) || s == State( 17, 4) || s== State(16,4)
        return -10
    else
        return 0 
    end   
 end
 
# set discount factor
gamma = 0.9

termination(s::State) = s == null
abstract type GridWorld <: MDP{State, Action} end

# Q_learning Algorithm in two dimensional 10x10 GridWorld.
q_mdp = QuickMDP(GridWorld,
    states = S,
    actions = A,
    transition = T,
    reward = R,
    discount = gamma,
    initialstate = S, # s_prime
    isterminal = termination
)

Random.seed!(101) # for reproduce the result
begin
    q_alpha = 0.8

    # number of episodes
    q_n_episodes = 1000 

    q_solver = QLearningSolver(
        n_episodes = q_n_episodes,
        learning_rate = q_alpha,
        exploration_policy = EpsGreedyPolicy(q_mdp, 0.5),
        verbose = false
    )
    # solve mdp
    q_policy = solve(q_solver, q_mdp)

    heatmap(reshape(mat,Params.size)',color=:turbo)
    for x in 1:Params.size[1], y in 1:Params.size[2]
        quiver!([x],[y],quiver=quivD[action(q_policy,State(x,y))], label="")        
    end
    plot!()
end

### state plotting stuff
using Plots

Params.size

mat = q_policy.value_table[1:end-1,4]
quivD = Dict(UP => ([0],[0.5]),
            DOWN => ([0],[-0.5]),
            LEFT => ([-0.5],[0]), 
            RIGHT => ([0.5],[0]));
begin
    heatmap(reshape(mat,Params.size)',color=:turbo)
    for x in 1:Params.size[1], y in 1:Params.size[2]
        quiver!([x],[y],quiver=quivD[action(q_policy,State(x,y))], label="")        
    end
    plot!()
end




a = DOWN
quivD[a]
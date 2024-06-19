
using POMDPs, POMDPModelTools, QuickPOMDPs
using POMDPSimulators
using DiscreteValueIteration, TabularTDLearning
using POMDPPolicies
using Random 
using Parameters
using LinearAlgebra
#make sure your path is correct
include("..\\src\\FiniteDipole.jl")
# using FiniteDipole # NATE: killed modules

# define State space _>>> EXPAND the boid definition to eat this
# or a distance function consume boid, target and output a State?
struct State
    r
    θ
    #are we in a terminal state
    done::Bool
end
#Default state constructor
State(r,θ) = State(r,θ,false)

@with_kw struct RadWorldParameters
    size::Tuple{Int,Int} #tuple with the grid size.
    null_state::State  #state that outside the grid.
    p_transition::Real #probabilities of transition to the next state.
    # Reward values 
    wd
    wa 
    # State Dimension metrics 
    D # max distance state ; \theta is boring - just cut up 2π
end
#Default constructor - needs for ℓ to be defined
ℓ = 5e-4
RadWorldParameters() = RadWorldParameters((30, 30), State(-1,-1), 0.7,
                                           0.1,0.9,
                                           2*ℓ*sqrt(2π))

# Had defaults for the constructor, but we can force values too
RW_params = RadWorldParameters()

#slap a boid into this thing and lets go
#SAMPLE constructorS

swim = Swimmer_params(ℓ)
b = FD_agent([0.0,0.0],[swim.Γ₀,-swim.Γ₀], π/2, [0.0,0.0],[0.0,0.0])

#start easy and put the target 1 to the right
target = (b.position[1],b.position[2] +5*swim.ℓ)

function dist_angle(agent::FD_agent, target)
    """ 
    Grab the distance and angle between an agent and the target

    ```jldoctest 
    julia> b = boid([1.0,0.5],[-1.0,1.0], π/2, [0.0,0.0],[0.0,0.0])
    julia> target = (b.position[1]-1.0,b.position[2] )
    julia> distance_angle_between(b,target)
    (1.0, 3.141592653589793)
    julia> target = (b.position[1]+1.0,b.position[2] )
    julia> distance_angle_between(b,target)
    (1.0, 0.0)
    
    """
    d =  target .- agent.position
    sqrt(sum(abs2,d)), atan(d[2],d[1])-agent.angle
    
end
function dist_angle(position, angle, target)
    """ 
    Grab the distance and angle between an agent and the target
    
    """
    d =  target .- position
    sqrt(sum(abs2,d)), atan(d[2],d[1])- angle
    
end

#testing
dn,θn =  dist_angle(b,target)
target = (b.position[1]+ 10*ℓ ,b.position[2] + 10*ℓ )
dn,θn =  dist_angle(b,target)

null = State(-1, -1)

function get_state(dn,θn)    
    """
    which state is the agent in? 
    State is in dists X thetas ordering
    #TODO: WRITE TEST CASES
    """
    delD = 10*RW_params.D
    delθ = 2π
    #mappings for which state a boid falls into
    rn = min(RW_params.size[1],max(1,floor(dn*RW_params.size[1]/delD)))
    tn = min(RW_params.size[2], max(1,floor(θn*RW_params.size[2]/delθ)))
    Int(tn*RW_params.size[1] +rn)
end

# have it take an agent as arg, use dot notation for a vector of agents
get_state(b::FD_agent) = get_state(dist_angle(b,target)...)


# The state defaults to 30 distances between 0 -> 10D
# and 30 angles between 0 -> 2π
S = [[State(r, θ) for r=LinRange(0,10*RW_params.D,RW_params.size[1]), 
                      θ=LinRange(0,2π,RW_params.size[2])]...,null]

#test to make sure that the swimmer will actually change states if moved
begin 
    swim = Swimmer_params(ℓ)
    b = FD_agent([0.0,0.0],[swim.Γ₀,-swim.Γ₀], π/2, [0.0,0.0],[0.0,0.0])

    #start easy and put the target 5 swimmers above
    target = (b.position[1],b.position[2] +5*swim.ℓ)    
    @show get_state(b)
    for i = 1:10

        move_swimmers!([b])
        @show get_state(b)
    end    
end
# define Actions
@enum Action CRUISE FASTER SLOWER LEFT RIGHT

# Action space.
A = [CRUISE, FASTER, SLOWER, LEFT, RIGHT]

#gamma increments for accel and turning
ga,gt = change_bearing_speed(swim)

#test it 
b + (gt,0) ### doesn't change the state of the swimmer
#this assumes we are changing the circulation on a swimmer
# -> how does accel or deccel effect State? or do we just care about pos?
# -> have the action change circ -> advect system with changes -> find new dist, θ
CIRC_CHANGE = Dict(
    1 => (0.0, 0.0),
    2 => (ga, ga),
    3 => (-ga, -ga),
    4 => (gt, -gt),
    5  => (-gt, gt)
) 
Base.:+(b::FD_agent,a::Tuple) = b.gamma .+ a

#helper function
function inbounds(r::Int, θ::Int)
    if 1 <= r <= RW_params.size[1] && 1 <= θ <=RW_params.size[2]
        return true
    else
        return false
    end
end
#this is depcrated now, as we test on the swimmer, not on the state
function inbounds(s::State)
    if 1 <= s.x <= RW_params.size[1] && 1 <= s.y <= RW_params.size[2]
        return true
    else
        return false
    end
end



# #transition function
# if dn == 0  -> achieved goal
# this can also be viewed as being in the states 
# S[1:30:end] #assuming RW_params.size[1] == 30
"""
The simulation here is different than GridWorld
Here, we assign a motion, 
      change circulation for each swimmer (as if no other swimmers existed)
      update the entire system

"""
function T(s::State, a::Action)
    # Deterministic() from POMDPModelTools.jl
    if R(s,a) != 0
        return Deterministic(null)# in this case we out of the environment.
    end

    len_a = length(A)
    next_states = Vector{State}(undef, len_a + 1)
    # calculating the probabilitie to the correct transition state.
    probabilities = zeros(len_a + 1) 
    
    #DOESN'T CHANGE IF GAMMA CHANGES
    v2v = vortex_to_vortex_velocity(boids;ℓ)    
    avg_v = add_lr(v2v)
    # siv = self_induced_velocity(boids;ℓ)
    # angles = angle_projection(boids,v2v)
    # ind_v = siv + avg_v #eqn 2.a
    
    position = zeros(2,(length(boids)))
    angles = zeros(length(boids))
    for (i,b) in enumerate(boids)
        # b.v = ind_v[:,i]
        position[:,i] = b.position + ind_v[:,i] .* Δt
        angles[i] = b.angle + angles[i]  #eqn 2.b
    end

    for (index, a_prime) in enumerate(A)
        # prob = (a_prime == a) ? 0.7 : 0.1
        prob = (a_prime == a) ? params.p_transition : 0.1
        # dest = s + CIRC_CHANGE[a_prime]
        #find the new circulation for the action
        new_gammas = boid + CIRC_CHANGE[a_prime] 
        #calculate the self-induced velocity
        siv = diff(new_gammas).*[cos(boid.angle), sin(boid.angle)]/(2π*swim.ℓ)
        #add the self-induced velocity to velocity induced by other swimmers 
        #NATE TODO : (indexing issues here)
        ind_v = siv + avg_v 
        # Update the positon 
        position[:,i] = b.position + ind_v[:,i] .* Δt
        

        next_states[index + 1] = get_state(RW_params,dist_angle(position,boid.angle,target))
        # the wall transition
        if !inbounds(params,dest)
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
 begin
    for (index, a_prime) in enumerate(A)
        @show a_prime
        @show CIRC_CHANGE[a_prime]
    end
 end

 #Reward Function
 function R(s, a)
    """
    eqn on pg 7 para 1
    TODO: look into how the solver scopes these variables: dn, S
    TODO: instead of dn use s.r?
    """
    # there is a better way to accululate being in the terminal state
    η = 0
    if s in S[1:RW_params.size[1]:end]
        return 100         
    elseif a == CRUISE 
        η =  0
    elseif a == FASTER
        η = -1
    elseif a == SLOWER
        η = 1
    elseif a == RIGHT
        η = -1
    elseif a == LEFT
        η = -1
    end
    RW_params.wa*(1- dn/ℓ) + η*RW_params.wd 
end
# set discount factor
gamma = 0.9


# termination(s::State) = s == null
termination(s::State) = s in S[1:30:end]
abstract type RadWorld <: MDP{State, Action} end

# Q_learning Algorithm in two dimensional 10x10 GridWorld.
q_mdp = QuickMDP(RadWorld,
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


end


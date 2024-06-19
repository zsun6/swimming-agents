export SwimmerEnv
using ReinforcementLearning
using Random
using StableRNGs
using IntervalSets
using Plots
using LinearAlgebra
"""
Minimum to setup for the SwimmerEnv
action_space(env::YourEnv)
state(env::YourEnv)
state_space(env::YourEnv)
reward(env::YourEnv)
is_terminated(env::YourEnv)
reset!(env::YourEnv)
(env::YourEnv)(action)
"""

# DQN
#A2C
#PPO
#transformers!!!!!
include("../src/FiniteDipole.jl")
struct SwimmerParams{T}
    #parameters for specific set of swimming agents
    ℓ::T #dipole length - vortex to vortex
    Γ0::T #init circulation
    Γa::T #incremental circulation
    v0::T #cruise velocity
    va::T #incremental velocity
    ρ::T  #turn radius    
    Γt::T#turning circ change
    wa::T    # Reward values 
    wd::T
    D::2 * π # max distance state ; \theta is boring - just cut up 2π
    Δt::T

end

#Default constructor uses
SwimmerParams(ℓ, T) = SwimmerParams(ℓ / 2,   #ℓ
    convert(T, 10 * π * ℓ^2), #Γ0
    π * ℓ^2,    #Γa
    5 * ℓ,      #v0
    ℓ / 2,    #va
    10ℓ,      #ρ
    convert(T, sqrt(2π^3) * ℓ^2), #Γt,
    convert(T, 0.1), #wa
    convert(T, 0.9), #wd
    convert(T, 2 * ℓ * sqrt(2π)),
    convert(T, 0.1))

mutable struct SwimmingEnv{A,T,R} <: AbstractEnv
    params::SwimmerParams{T}
    action_space::A
    action::T
    observation_space::Space{Vector{ClosedInterval{T}}}
    state::Vector{T}
    done::Bool
    t::Int
    max_steps::Int
    rng::R
    reward::T
    rewards
    n_actions::Int
    swimmer::FD_agent{T}
    target::Vector{T}
    target_path::Tuple{Vector{T},Vector{T}}
    freestream::Vector{T}
    # swimmers::Vector{FD_agent{T}} #multiple swimmers 
end

function _target_path(; ω=5; T=3; N=25)
    omega = ω #rad/s # ω is a default value passed into the function, you can override it if you like
    ll = 0.5
    # these values should scale with values ripped from SwimmerParams and the SwimmerEnv
    t = 0 # <-- SwimmerEnv.t
    T  # periods
    N  # frequency? same as above
    # path should be a function of the period and frequency above, you just probe it at an explicit time
    # the function below is close 

    #infinity ring
    path(ind) = omega * ll * 5.0 .* [cos(omega * ind), sin(omega * 2 * ind) / 2]

    #circle
    #path(ind) = ll* [cos(omega*ind) , sin(omega*ind)]

    #Line
    #path(ind) = [ll*ind, 0.0]
end

"""
    SwimmerEnv(; T = Float32,  max_steps = 500, n_actions::Int = 5, rng = Random.GLOBAL_RNG,
    termation_type=1, ℓ=5e-4, =[5.0,0.0],freestream = [1,0,3])

T = Type, max_steps is the number of steps per episode, n_actions is actions of the swimmer locked to five for now 
rng = Random Num Gen, termation_type selects differenct criteria for ending an episode,ℓ dist to vortex from mdpt
 is location of  scaled by d
freestream = [x_dir, y_dir, Uinf*v0]
"""
function SwimmerEnv(; T=Float32, max_steps=500, n_actions::Int=5, rng=Random.GLOBAL_RNG,
    ℓ=5e-4, dir=[5.0, 0.0], freestream=[1, 0, 3])
    # high = T.([1, 1, max_speed])
    ℓ = convert(T, ℓ)
    action_space = Base.OneTo(n_actions) # A = [CRUISE, FASTER, SLOWER, LEFT, RIGHT]
    ηs = [1, -1, 1, -1, -1] # [CRUISE, FASTER, SLOWER, LEFT, RIGHT]
    swim = SwimmerParams(ℓ, T)
    #the freestream tuple is [x,y,scale] -> scale*v0*[x,y]
    env = SwimmingEnv(swim,
        action_space,
        0 |> T,
        Space(ClosedInterval{T}.([0, 0], [20 * ℓ * sqrt(2π), 2π])), #obs(radius, angle) = [0->10D][0->2π]
        zeros(T, 2), #r,θ to 
        false,
        0,
        max_steps,
        rng,
        zero(T),
        ηs,
        n_actions, FD_agent(Vector{T}([0.0, 0.0]), Vector{T}([swim.Γ0, -swim.Γ0]), T(π / 2), Vector{T}([0.0, 0.0]), Vector{T}([0.0, 0.0])),
        # Vector{T}([-5.0 * swim.D, 0.0])
        # Vector{T}([5.0 * swim.D, 0.0])
        convert(Vector{T}, swim.D),
        _target_path,
        #the freestream tuple is [x,y,scale] -> scale*v0*[x,y]
        convert(Vector{T}, freestream[3] * swim.v0 * [freestream[1], freestream[2]] / norm([freestream[1], freestream[2]])))
    reset!(env)
    env
end

Random.seed!(env::SwimmingEnv, seed) = Random.seed!(env.rng, seed)

RLBase.action_space(env::SwimmingEnv) = env.action_space
RLBase.state_space(env::SwimmingEnv) = env.observation_space
RLBase.reward(env::SwimmingEnv) = env.reward
#three options for is_terminated
RLBase.is_terminated(env::SwimmingEnv) = env.done

function RLBase.state(env::SwimmingEnv{A,T}) where {A,T}
    dn, θn = dist_angle(env.swimmer, env.target)
    [clamp(dn, 0, env.observation_space[1].right) |> T, mod2pi(θn) |> T]
end

function RLBase.reset!(env::SwimmingEnv{A,T}) where {A,T}
    #go back to the original state, not random?  
    θ = rand() * 2π  #π/2 old
    r = rand() * env.observation_space[1].right #0.0 old

    env.swimmer = FD_agent(Vector{T}([state2xy(r, θ)...]), Vector{T}([env.params.Γ0, -env.params.Γ0]),
        T(θ), Vector{T}([0.0, 0.0]), Vector{T}([0.0, 0.0]))
    dn, θn = dist_angle(env.swimmer, env.target_path)
    env.state[1] = clamp(dn, 0, env.observation_space[1].right)
    env.state[2] = mod2pi(θn)
    env.action = zero(T)
    env.t = 0
    env.done = false
    env.reward = zero(T)
    #if we add a dynamic  throw code here too!
    nothing
end

function (env::SwimmingEnv)(a::Union{Int,AbstractFloat})
    @assert a in env.action_space
    env.action = change_circulation!(env, a)
    _step!(env, env.action)
end



function _step!(env::SwimmingEnv, a)
    env.t += 1

    v2v = vortex_to_vortex_velocity([env.swimmer]; env.params.ℓ)
    avg_v = sum(v2v, dims=2) #add_lr(v2v) 

    siv = self_induced_velocity([env.swimmer]; env.params.ℓ)
    ind_v = siv + avg_v #eqn 2.a

    angles = angle_projection([env.swimmer], v2v .+ env.freestream)
    # @show norm(ind_v) #siv,avg_v
    for (i, b) in enumerate([env.swimmer])
        # log these values and look at them, are they physical? 
        b.position += (ind_v[:, i] + env.freestream) .* env.params.Δt
        b.angle = mod2pi(b.angle + angles[i] .* env.params.Δt) #eqn 2.b
    end

    dn, θn = dist_angle(env.swimmer, env.target[1], env.target[2])




    path(omega, t) = omega * env.params.ℓ * 5.0 .* [-sin(omega * t), (cos(omega * 2 * t) / 2)]
    env.path += path(2π, env.t / env.max_steps) .* env.params.Δt

    env.state[1] = clamp(dn, 0, env.observation_space[1].right) #make sure it is in the domain
    env.state[2] = mod2pi(θn)

    #NATE: changed up costs to allow wildly negative costs if outside of observation_space
    costs = env.params.wd * (1 - dn / env.observation_space[1].right) + env.rewards[Int(a)] * env.params.wa
    #NATE : hits the , gets a big bonus!
    @assert costs <= 1.0
    if dn <= env.params.D / 2.0
        costs += 10.0
    end

    env.done = env.t >= env.max_steps
    env.reward = costs
    nothing
end

function dist_angle(agent::FD_agent, target_x, target_y)
    """
        Grab the distance and angle between an agent and the 

        ```jldoctest 
        julia> b = boid([1.0,0.5],[-1.0,1.0], π/2, [0.0,0.0],[0.0,0.0])
        julia>  = (b.position[1]-1.0,b.position[2] )
        julia> distance_angle_between(b,)
        (1.0, 3.141592653589793)
        julia>  = (b.position[1]+1.0,b.position[2] )
        julia> distance_angle_between(b,)
        (1.0, 0.0)"""

    d = [target_x, target_y] - agent.position
    (sqrt(sum(abs2, d)), mod2pi(mod2pi(atan(d[2], d[1])) - agent.angle))
end

dist_angle(env::SwimmingEnv) = dist_angle(env.swimmer, env.target[1], env.target[2])


function change_circulation!(env::SwimmingEnv{<:Base.OneTo}, a::Int)
    """ Agent starts with circulation of from
        [-swim.Γ0, swim.Γ0]
        so we have to inc/dec to change the magnitude not absolutes
        [CRUISE, FASTER, SLOWER, LEFT, RIGHT]"""
    # TODO: Add sign(enb.swimmer.gamma)?

    if a == 1
        # nothing
        env.swimmer.gamma = [-env.params.Γ0, env.params.Γ0]
    elseif a == 2
        env.swimmer.gamma = [-env.params.Γ0 - env.params.Γa, env.params.Γ0 + env.params.Γa]
    elseif a == 3
        env.swimmer.gamma = [-env.params.Γ0 + env.params.Γa, env.params.Γ0 - env.params.Γa]
    elseif a == 4
        env.swimmer.gamma = [-env.params.Γ0 - env.params.Γt, env.params.Γ0 - env.params.Γt]
    elseif a == 5
        env.swimmer.gamma = [-env.params.Γ0 + env.params.Γt, env.params.Γ0 + env.params.Γt]
    else
        @error "unknown action of $action"
    end

    a
end

function change_circulation_incremental!(env::SwimmingEnv{<:Base.OneTo}, a::Int)
    """ Agent starts with circulation of from
        broken
        [-swim.Γ0, swim.Γ0]
        so we have to inc/dec to change the magnitude not absolutes
        [CRUISE, FASTER, SLOWER, LEFT, RIGHT]"""
    # TODO: Add sign(enb.swimmer.gamma)?

    if a == 1
        # nothing
        env.swimmer.gamma = [-env.params.Γ0, env.params.Γ0]
    elseif a == 2
        env.swimmer.gamma += [-env.params.Γa, +env.params.Γa]
    elseif a == 3
        env.swimmer.gamma += [+env.params.Γa, -env.params.Γa]
    elseif a == 4
        env.swimmer.gamma += [-env.params.Γt, -env.params.Γt]
    elseif a == 5
        env.swimmer.gamma += [+env.params.Γt, +env.params.Γt]
    else
        @error "unknown action of $action"
    end

    a
end


env = SwimmerEnv()
RLBase.test_runnable!(env)

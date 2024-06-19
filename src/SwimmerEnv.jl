export SwimmerEnv
using ReinforcementLearning
using Random
using StableRNGs
using IntervalSets
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

include("../src/FiniteDipole.jl")
"""
The SwimmerParams are immutable and do not change during the episode(s) for a given study
"""
struct SwimmerParams{T}
    #parameters for specific set of swimming agents
    ℓ ::T #dipole length - vortex to vortex
    Γ0::T #init circulation
    Γa::T #incremental circulation
    v0::T #cruise velocity
    va::T #incremental velocity
    ρ::T  #turn radius    
    Γt ::T#turning circ change
    wa ::T    # Reward values 
    wd ::T
    D ::T # max distance state ; \theta is boring - just cut up 2π
    Δt :: T
    
end 
#Default constructor uses
SwimmerParams(ℓ,T) = SwimmerParams(ℓ/2,   #ℓ
                                convert(T,10*π*ℓ^2), #Γ0
                                π*ℓ^2,    #Γa
                                5*ℓ,      #v0
                                ℓ/2,    #va
                                10ℓ,      #ρ
                                convert(T,sqrt(2π^3)*ℓ^2), #Γt,
                                convert(T,0.1), #wa
                                convert(T,0.9), #wd
                                convert(T,2*ℓ*sqrt(2π)),
                                convert(T, 0.1))

mutable struct SwimmingEnv{A,T,R} <: AbstractEnv
    params::SwimmerParams{T}
    action_space::A
    action::Vector{T}
    observation_space::Space{Vector{ClosedInterval{T}}}
    # state::Vector{Vector{T}}
    state::Matrix{T}
    done::Bool
    t::Int
    max_steps::Int
    rng::R
    reward::Vector{T}
    rewards
    n_actions::Int
    swimmers::Vector{FD_agent{T}} # multiple swimmers
    target::Vector{T}       #x,y pair
    freestream::Vector{T}   #x,y vector with magnitude 
end
#Default constructor FD_agent given SwimmerParams
#TODO: Zhongrui - add in sp and rng to constructors

function FD_agent(swim::SwimmerParams) 
    T = typeof(swim.ℓ)
    FD_agent(Vector{T}([0.0,0.0]), Vector{T}([swim.Γ0,-swim.Γ0]), T(π/2), Vector{T}([0.0,0.0]), Vector{T}([0.0,0.0]))
end

function FD_agent(swim::SwimmerParams, xy, angle ) 
    T = typeof(swim.ℓ)
    FD_agent(Vector{T}([xy[1],xy[2]]), Vector{T}([swim.Γ0,-swim.Γ0]), T(angle), Vector{T}([0.0,0.0]), Vector{T}([0.0,0.0]))
end
FD_agent(swim::SwimmerParams, xy) = FD_agent(swim, xy, π/2.0) 
"""
    SwimmerEnv(; T = Float32,  max_steps = 500, n_actions::Int = 5, rng = Random.GLOBAL_RNG,
    termation_type=1, ℓ=5e-4, target=[5.0,0.0],freestream = [1,0,3])

T = Type, max_steps is the number of steps per episode, n_actions is actions of the swimmer locked to five for now 
rng = Random Num Gen, termation_type selects differenct criteria for ending an episode,ℓ dist to vortex from mdpt
target is location of target scaled by d
freestream = [x_dir, y_dir, Uinf*v0]
"""
function SwimmingEnv(;n_swimmers = 2, T = Float32, max_steps = 500, n_actions::Int = 5, rng = Random.GLOBAL_RNG,
    ℓ=5e-4, target=[5.0,5.0],freestream = [0,0,1])
    # high = T.([1, 1, max_speed])
    ℓ = convert(T,ℓ)
    actions = zeros(T, n_swimmers) #As many actions as swimmers
    action_space = Base.OneTo(n_actions) # A = [CRUISE, FASTER, SLOWER, LEFT, RIGHT]
    ηs = [1, -1, 1, -1, -1] # [CRUISE, FASTER, SLOWER, LEFT, RIGHT]
    swim =  SwimmerParams(ℓ,T)
    #the freestream tuple is [x,y,scale] -> scale*v0*[x,y]
    swimmers = [FD_agent(swim) for _ in 1:n_swimmers]
    freestream = convert(Vector{T}, freestream[3]*swim.v0*[freestream[1],freestream[2]]/norm([freestream[1],freestream[2]]))
    target = convert(Vector{T}, target.*swim.D)
    maxD = maximum(dist_angle(swimmers,target)[1])
    env = SwimmingEnv(swim,
        action_space,
        actions, 
        # Space(ClosedInterval{T}.([0..20*ℓ*sqrt(2π), 0..2π])), #obs(radius, angle) = [0->10D, 0->2π]
        Space(ClosedInterval{T}.([0..maxD, 0..2π])), #obs(radius, angle) = [0->10D, 0->2π]
        zeros(T, n_swimmers, 2), #r,θ to target
        false,
        0,
        max_steps,
        rng,
        zeros(T, n_swimmers),
        ηs,
        n_actions,
        swimmers,
        # Vector{T}([-5.0 * swim.D, 0.0])
        # Vector{T}([5.0 * swim.D, 0.0])
        target,
        #the freestream tuple is [x,y,scale] -> scale*v0*[x,y]        
        all(x->isnan(x), freestream) ? [T(0.0), T(0.0)] : freestream
    )
    reset!(env)
    env
end

Random.seed!(env::SwimmingEnv, seed) = Random.seed!(env.rng, seed)
# Defining the `action_space` of each independent player can help to transform
# this SIMULTANEOUS environment into a SEQUENTIAL environment with
# [`simultaneous2sequential`](@ref).
RLBase.action_space(env::SwimmingEnv, ::Int) = env.action_space

#TODO: Reduce this to be a lazy eval -> reduce it down to a sample space
RLBase.action_space(env::SwimmingEnv, ::SimultaneousPlayer) = env.action_space
#wrong
    # Tuple(Iterators.product(fill(env.action_space, axes(env.swimmers))...))
#right
    # env.action_space
RLBase.action_space(env::SwimmingEnv) = action_space(env, SIMULTANEOUS_PLAYER)

# RLBase.action_space(env::SwimmingEnv) = env.action_space
#TODO: expand state_space to accomodate N swimmers
function RLBase.state_space(env::SwimmingEnv) 
    out = []
    for i in 1:length(env.swimmers)
        push!(out,env.observation_space[1])
        push!(out,env.observation_space[2])
    end
    Space(out)
end

RLBase.state_space(env::SwimmingEnv, p) = Space(state_space(env::SwimmingEnv)[(p-1)*2+1:(p-1)*2+2])

# RLBase.state_space(env::SwimmingEnv) = env.observation_space
#TODO: Zhongrui - do we use the best reward? or the current reward? or the average reward? .... or a vector
RLBase.reward(env::SwimmingEnv) = env.reward
RLBase.reward(env::SwimmingEnv, p) = env.reward[p]
RLBase.is_terminated(env::SwimmingEnv) = env.done
RLBase.NumAgentStyle(env::SwimmingEnv) = MultiAgent(length(env.swimmers))
RLBase.DynamicStyle(env::SwimmingEnv) = SIMULTANEOUS
RLBase.players(env::SwimmingEnv) = 1:length(env.swimmers)
RLBase.current_player(env::SwimmingEnv) = SIMULTANEOUS_PLAYER

function RLBase.state(env::SwimmingEnv{A,T}) where {A,T} 
    dn,θn =  dist_angle(env.swimmers,env.target)
    dn = clamp.(dn, 0, env.observation_space[1].right - eps(Float32)).|>T 
    θn = mod2pi.(θn).|>T   
    out = [] 
    [push!(out,dn[i], θn[i]) for i in eachindex(dn)]
    out
end

RLBase.state(env::SwimmingEnv, p) = state(env::SwimmingEnv)[(p-1)*2+1:(p-1)*2+2]
    


function RLBase.reset!(env::SwimmingEnv{A,T}) where {A,T}
    #go back to the original state, not random?  
    n_swimmers = length(env.swimmers)

    θs = ones(T, n_swimmers) .* 2π
    rs = (collect(1:n_swimmers) .- 1) .* 4.0 *env.params.ℓ 
    # env.swimmers = [FD_agent(env.params, pos) for pos in state2xy.(rs,θs)]
    env.swimmers = [FD_agent(env.params, pos) for pos in diamond_dist(n_swimmers, env.params.D*4)]
    dθs = map(x->dist_angle(x, env.target), env.swimmers)
    for i in 1:n_swimmers
        env.state[i, 1] = dθs[i][1]
        env.state[i, 2] = dθs[i][2]
    end
    env.action = zeros(T, n_swimmers)
    env.t = 0
    env.done = false
    env.reward = zeros(T, n_swimmers)
    nothing    
end

function diamond_dist(n, d; theta = π/4)
    N = √n|>Int
    x = LinRange(0,(N-1)*d,N)
    pos = [[x1 x2] for x1 in x for x2 in x]     
    rot(θ) = [cos(θ) sin(θ)
             -sin(θ) cos(θ)]
    pos = [p*rot(theta) for p in pos]
end

function (env::SwimmingEnv)(a::Union{Int, AbstractFloat})
    @assert a in env.action_space
    env.action = change_circulation!(env, a)
    _step!(env, env.action)
end

function (env::SwimmingEnv)(as::Tuple)   
    for (i,a) in enumerate(as)
        @assert a in env.action_space         
        env.action[i] = change_circulation!(env, a; index = i)               
    end            
    _step!(env, env.action)
end

function _step!(env::SwimmingEnv{A,T}, as) where {A,T}
    env.t += 1
    
    v2v = vortex_to_vortex_velocity(env.swimmers; env.params.ℓ)   
    avg_v = add_lr(v2v) #sum(v2v,dims=2)
    
    siv = self_induced_velocity(env.swimmers;env.params.ℓ)
    ind_v = siv + avg_v #eqn 2.a

    angles = angle_projection(env.swimmers,v2v .+ env.freestream)
    # @show norm(ind_v) #siv,avg_v
    for (i,b) in enumerate(env.swimmers)
        # log these values and look at them, are they physical? 
        b.position += (ind_v[:,i] + env.freestream) .* env.params.Δt
        b.angle = mod2pi(b.angle + angles[i] .* env.params.Δt) #eqn 2.b
    end
    
    dn,θn = dist_angle(env.swimmers, env.target)
    
    
    env.state[:,1] = clamp.(dn, 0, env.observation_space[1].right-eps(T) ) #make sure it is in the domain
    env.state[:,2] = mod2pi.(θn)       

    
    costs = zeros(size(env.swimmers) )

    @. costs = env.params.wd*(1 - dn /env.observation_space[1].right) + [env.rewards[Int(i)] for i in as]*env.params.wa 

    @assert all(costs .<= 1.0)

    if sum(dn)./length(env.swimmers) <  env.params.D * sqrt(length(env.swimmers))
        costs .+= 100.0
    end
    
    env.done = env.t >= env.max_steps || env.done
    #### DO THE AGENTS OVERLAP? If yes, kill it
    n_swimmers = length(env.swimmers)
    xs = Vector{T}(undef, n_swimmers)
    ys = Vector{T}(undef, n_swimmers)
    for i in axes(env.swimmers,1)
        xs[i],ys[i] = env.swimmers[i].position
    end
    dist = Matrix{T}(undef, n_swimmers, n_swimmers)
    @. dist = sqrt((xs - xs')^2 + (ys - ys')^2)
    dist[diagind(dist)] .= 1.0
    env.done = env.done || length(findall(x-> x < env.params.D/4.0, dist)) > 0 

    env.reward = costs
    nothing
end

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
    sqrt(sum(abs2,d)), mod2pi(mod2pi(atan(d[2],d[1])) - agent.angle)
    
end

function dist_angle(agents::Vector{FD_agent{T}}, target ) where T
    dns = []
    θs  = []
    for agent in agents
        dn, θn = dist_angle(agent, target)
        push!(dns, dn)
        push!(θs, θn)
    end
    dns, θs
end

dist_angle(env::SwimmingEnv) = dist_angle(env.swimmers, env.target)   


function change_circulation!(env::SwimmingEnv{<:Base.OneTo}, a::Int ; index = 1)
    """ Agent starts with circulation of from
        [-swim.Γ0, swim.Γ0]
        so we have to inc/dec to change the magnitude not absolutes
        [CRUISE, FASTER, SLOWER, LEFT, RIGHT]"""

    
    if a == 1
        # nothing
        env.swimmers[index].gamma = [-env.params.Γ0, env.params.Γ0]
    elseif a == 2   
        env.swimmers[index].gamma = [-env.params.Γ0 - env.params.Γa, env.params.Γ0 + env.params.Γa]
    elseif a == 3  
        env.swimmers[index].gamma = [-env.params.Γ0 + env.params.Γa, env.params.Γ0 - env.params.Γa]
    elseif a == 4  
        env.swimmers[index].gamma = [-env.params.Γ0 - env.params.Γt, env.params.Γ0 - env.params.Γt]
    elseif a == 5   
        env.swimmers[index].gamma = [-env.params.Γ0 + env.params.Γt, env.params.Γ0  + env.params.Γt]
    else 
        @error "unknown action of $action"
    end 
    
    a
end


function change_circulation!(env::SwimmingEnv{<:Base.OneTo}, as::Vector{Int})
    """ Agent starts with circulation of from
        [-swim.Γ0, swim.Γ0]
        so we have to inc/dec to change the magnitude not absolutes
        [CRUISE, FASTER, SLOWER, LEFT, RIGHT]"""
    
    for (i,a) in enumerate(as)
        if a == 1
            # nothing
            env.swimmers[i].gamma = [-env.params.Γ0, env.params.Γ0]
        elseif a == 2   
            env.swimmers[i].gamma = [-env.params.Γ0 - env.params.Γa, env.params.Γ0 + env.params.Γa]
        elseif a == 3  
            env.swimmers[i].gamma = [-env.params.Γ0 + env.params.Γa, env.params.Γ0 - env.params.Γa]
        elseif a == 4  
            env.swimmers[i].gamma = [-env.params.Γ0 - env.params.Γt, env.params.Γ0 - env.params.Γt]
        elseif a == 5   
            env.swimmers[i].gamma = [-env.params.Γ0 + env.params.Γt, env.params.Γ0  + env.params.Γt]
        else 
            @error "unknown action of $action"
        end
    end     
    as
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
        env.swimmers.gamma = [-env.params.Γ0, env.params.Γ0]
    elseif a == 2   
        env.swimmers.gamma += [ - env.params.Γa,  + env.params.Γa]
    elseif a == 3  
        env.swimmers.gamma += [ + env.params.Γa,  - env.params.Γa]
    elseif a == 4  
        env.swimmers.gamma += [ - env.params.Γt, - env.params.Γt]
    elseif a == 5   
        env.swimmers.gamma += [ + env.params.Γt,  + env.params.Γt]
    else 
        @error "unknown action of $action"
    end 
    
    a
end

state2xy(r,θ) = [r*cos(θ),r*sin(θ)]
state2xy(r,θ, target) = [r*cos(θ+π)+target[1],r*sin(θ+π)+target[2]]
Base.@kwdef mutable struct DistRewardPerEpisode <: AbstractHook    
    dists::Vector = []
    position::Vector = []
    positions:: Vector = []
    rewards::Vector = []
    gammas::Vector = []
    reward = 0
    is_display_on_exit::Bool = true
end

function (h::DistRewardPerEpisode)(::PostEpisodeStage, policy, env) 
    push!(h.dists, dist_angle(env))
    push!(h.positions, h.position)
    # push!(h.gammas, env.swimmers.gamma)    
    push!(h.rewards, h.reward)
    h.reward = 0.0
    h.position = []
end
function (h::DistRewardPerEpisode)(::PostActStage, policy, env)
    h.reward += reward(env)
    push!(h.position, env.swimmers.position)
end



env = SwimmingEnv(;n_swimmers = 4)
RLBase.test_interfaces!(env)
RLBase.test_runnable!(env)

# New tests to see if our environment actually does what we want
using ReinforcementLearning
using Random
using StableRNGs
using Flux
using Flux.Losses
using Plots


##ACTUAL SCRIPTING##
include("..\\src\\SwimmerEnv.jl")
##Modified functions
function (learner::DQNLearner)(env::SwimmingEnv)   
    env |>
    state |>
    x ->
        Flux.unsqueeze(x, ndims(x) + 1) |>
        x ->
            send_to_device(device(learner), x) |>
            learner.approximator |>
            vec |>
            send_to_host
end
function (learner::DQNLearner)(env::SwimmingEnv, player)
    state(env,player) |>
    x ->
        Flux.unsqueeze(x, ndims(x) + 1) |>
        x ->
            send_to_device(device(learner), x) |>
            learner.approximator |>
            vec |>
            send_to_host
end
function (hook::TotalRewardPerEpisode)(::PostActStage, agent, env::SwimmingEnv)
    hook.reward += sum(reward(env))/length(env.swimmers)
end


##REWRITING OF RUN FOR THE ENV##
import Base: run
function run(
    policy::AbstractPolicy,
    env::SwimmingEnv,
    stop_condition = StopAfterEpisode(1),
    hook = EmptyHook(),
)
    check(policy, env)
    _run(policy, env, stop_condition, hook)
end

function check(policy, env) end

function _run(policy::AbstractPolicy, env::SwimmingEnv, stop_condition, hook::AbstractHook)

    hook(PRE_EXPERIMENT_STAGE, policy, env)
    policy(PRE_EXPERIMENT_STAGE, env)
    is_stop = false
    while !is_stop
        reset!(env)
        policy(PRE_EPISODE_STAGE, env)
        hook(PRE_EPISODE_STAGE, policy, env)

        while !is_terminated(env) # one episode
            action = policy(env) 
          
            policy(PRE_ACT_STAGE, env, action)
            hook(PRE_ACT_STAGE, policy, env, action)

            env(action)

            policy(POST_ACT_STAGE, env)
            hook(POST_ACT_STAGE, policy, env)

            if stop_condition(policy, env)
                is_stop = true
                break
            end
        end # end of an episode

        if is_terminated(env)
            policy(POST_EPISODE_STAGE, env)  # let the policy see the last observation
            hook(POST_EPISODE_STAGE, policy, env)
        end
    end
    hook(POST_EXPERIMENT_STAGE, policy, env)
    hook
end
## REWRITING OF AGENT.JL ##
function RLBase.update!(
    trajectory::AbstractTrajectory,
    policy::AbstractPolicy,
    env::SwimmingEnv,
    ::PreActStage,
    action,
)
    # s = policy isa NamedPolicy ? state(env, nameof(policy)) : state(env)
    for i in 1:length(env.swimmers)
        push!(trajectory[:state], state(env,i))
        push!(trajectory[:action], action[i])
    end
    if haskey(trajectory, :legal_actions_mask)
        lasm =
            policy isa NamedPolicy ? legal_action_space_mask(env, nameof(policy)) :
            legal_action_space_mask(env)
        push!(trajectory[:legal_actions_mask], lasm)
    end
end
function RLBase.update!(
    trajectory::AbstractTrajectory,
    policy::AbstractPolicy,
    env::SwimmingEnv,
    ::PostActStage,
)
    # r = policy isa NamedPolicy ? reward(env, nameof(policy)) : reward(env)
    for i in 1:length(env.swimmers)
        push!(trajectory[:reward], reward(env,i))
    end

    push!(trajectory[:terminal], is_terminated(env))
end
function get_dummy_action(action_space::AbstractVector)
    # For discrete action spaces, we select the first action as dummy action.
    return action_space[1]
end
function RLBase.update!(
    trajectory::AbstractTrajectory,
    policy::AbstractPolicy,
    env::SwimmingEnv,
    ::PostEpisodeStage,
)
    # Note that for trajectories like `CircularArraySARTTrajectory`, data are
    # stored in a SARSA format, which means we still need to generate a dummy
    # action at the end of an episode.

    s = policy isa NamedPolicy ? state(env, nameof(policy)) : state(env,1)

    A = policy isa NamedPolicy ? action_space(env, nameof(policy)) : action_space(env)
    a = get_dummy_action(A)

    push!(trajectory[:state], s)
    push!(trajectory[:action], a)
    if haskey(trajectory, :legal_actions_mask)
        lasm =
            policy isa NamedPolicy ? legal_action_space_mask(env, nameof(policy)) :
            legal_action_space_mask(env)
        push!(trajectory[:legal_actions_mask], lasm)
    end
end
##REWRITE q_based_policy.jl ##
function (π::QBasedPolicy)(env::SwimmingEnv, ::MinimalActionSet, ::Base.OneTo) 
    actions = []
    for player in players(env) 
        push!(actions, π.explorer(π.learner(env,player)))
    end
    actions |> Tuple
end

##A new HOOK##
Base.@kwdef mutable struct PotentialHook <: AbstractHook    
    path::Vector = []
    states:: Vector = []
    angle::Vector = []
    angles::Vector = []
    reward = 0.0
    rewards::Vector = []
    actions::Vector = []
    action::Vector = []
    is_display_on_exit::Bool = true
end
function (h::PotentialHook)(::PostEpisodeStage, policy, env) 
    push!(h.states, h.path)
    push!(h.actions, h.action)
    push!(h.rewards,h.reward)
    push!(h.angles, h.angle)
    h.reward = 0
    h.path = []
    h.action =[]
    h.angle = []
end
function (h::PotentialHook)(::PostActStage, policy, env)
    push!(h.path, vcat([sw.position for sw in env.swimmers]...))
    push!(h.action, env.action)
    push!(h.angle, vcat([sw.angle for sw in env.swimmers]...))
    h.reward += sum(reward(env))
end

n_swimmers = 4
env = SwimmingEnv(;n_swimmers = n_swimmers, max_steps = 200, target=[0,75],freestream=[0, -1, 2])
# wrappedenv = discrete2standard_discrete(env)
player = 1 # or 2
ns, na = length(state(env, player)), length(action_space(env,  player))
rng = StableRNG(abs(rand(Int)))
agent = Agent(
    policy =           
        QBasedPolicy(
        learner = DQNLearner(
            approximator = NeuralNetworkApproximator(
                model = Chain(
                    Dense(ns, 64, relu; init = glorot_uniform(rng)),
                    Dense(64, 64, relu; init = glorot_uniform(rng)),
                    Dense(64, na; init = glorot_uniform(rng)),
                ) |> gpu,
                optimizer = ADAM(),
            ),
            target_approximator = NeuralNetworkApproximator(
                model = Chain(
                    Dense(ns, 64, relu; init = glorot_uniform(rng)),
                    Dense(64, 64, relu; init = glorot_uniform(rng)),
                    Dense(64, na; init = glorot_uniform(rng)),
                ) |> gpu,
                optimizer = ADAM(),
            ),
            loss_func = huber_loss,
            stack_size = nothing,
            batch_size = 32,
            update_horizon = 1,
            min_replay_history = 100,
            update_freq = 1,
            target_update_freq = 100,
            rng = rng,
        ),
        explorer = EpsilonGreedyExplorer(
            kind = :exp,
            ϵ_stable = 0.01,
            decay_steps = 5000,
            rng = rng,
        ),
    ),
    trajectory = CircularArraySARTTrajectory(
        capacity = 500_000,
        state = Vector{Float32} => (ns,),
    )
)

stop_condition =StopAfterEpisode(5_000; cur = 0, is_show_progress = true) 
# StopAfterStep(500_000, is_show_progress=!haskey(ENV, "CI")) # 
# hook = TotalRewardPerEpisode()
hook = PotentialHook()

ex = Experiment(agent, env, stop_condition, hook, "")

# run(ex)
#work on MD for this, typing issue at this level
run(ex)

#look at the rewards of all swimmers
plot([ex.hook.rewards[i:n_swimmers:end] for i=1:n_swimmers])
plot(ex.hook.rewards./n_swimmers, st=:scatter)

plot(ex.hook.rewards[1:n_swimmers:end])
ex.hook.rewards[1:n_swimmers:end]
rew = zeros(length(ex.hook.rewards)÷n_swimmers)
for i= 1:n_swimmers
	rew[:] += ex.hook.rewards[i:n_swimmers:end]
end
rew ./= n_swimmers
plot(rew)
#NEW HOOK PLOTS
begin    
    rs = []
    thetas = []
    a = plot([env.target[1]],[env.target[2]], st=:scatter, marker=:star, ms=4,label="")
    # for pos in ex.hook.states[9666]
    for pos in ex.hook.states[argmax(ex.hook.rewards)]
        push!(rs, pos[1:2:end]...)
        push!(thetas,pos[2:2:end]...)        
    end
    
    for i = 1:n_swimmers        
        plot!(a, rs[i:n_swimmers:end], thetas[i:n_swimmers:end], lw= 1, label="")        
    end
    plot!(a,[env.target[1]],[env.target[2]], st=:scatter, marker=:star, ms=4,label="Target")
    a
end
begin    
    a = plot([env.target[1]],[env.target[2]], st=:scatter, marker=:star, ms=4)   
    anim = @animate for (frame, pos) in enumerate(ex.hook.states[argmax(ex.hook.rewards)])
        a = plot([env.target[1]],[env.target[2]], st=:scatter, marker=:star, ms=4, ylims=(-0.02,0.3))
        for i = 1:n_swimmers
     
            plot!(a,[pos[(i-1)*2+1]],[pos[(i-1)*2+2]], st=:scatter, label="")
            plot!(title="$frame", aspect_ratio=:equal)
        end    
        a
    end
    gif(anim)
    
end

argmax(ex.hook.rewards)



begin
    #create sample to test
    ep = argmax(ex.hook.rewards)
    # ep = 750
    angles = ex.hook.angles[ep]
    states = ex.hook.states[ep]
    actions = ex.hook.actions[ep]

 
    #first time step
    angle = angles[1]
    coms = states[1]
    angs = angles[1]

    #a swimmer going up from origin
    xs = LinRange(-0.15,0.15,101)
    ys = LinRange(-0.05,0.25,101)
    X = repeat(reshape(xs, 1, :), length(ys), 1)
    Y = repeat(ys, 1, length(xs))
    targets = [X,Y]
    #climits
    g = env.params.Γ0
    dx = minimum( diff(xs))
    cl = g*dx*(2π)
    anim = @animate for i = 1:length(angles)
        pot = potential(states[i], angles[i], targets, actions[i]; env.params.ℓ)
        f = plot(collect(xs),collect(ys), pot, st=:contour)#, clim=(-cl, cl),levels=40)
        plot!(f, states[i][1:2:end], states[i][2:2:end], st=:scatter)
        f
    end
    gif(anim)
end




N = 4
D = 7.5

x = LinRange(0,(N-1)*D,N)
pos = [[x1 x2] for x1 in x for x2 in x] 
a = plot()
theta = deg2rad(45)
rot(θ) = [cos(θ) sin(θ)
         -sin(θ) cos(θ)]
pos = [p*rot(theta) for p in pos]
for p in pos
    plot!(a, [p[1]],[p[2]], st=:scatter, label="")
end
a

function diamond_dist(n, d; theta = π/4)
    N = √N
    x = LinRange(0,(N-1)*D,N)
    pos = [[x1 x2] for x1 in x for x2 in x]     
    rot(θ) = [cos(θ) sin(θ)
             -sin(θ) cos(θ)]
    pos = [p*rot(theta) for p in pos]
end

xs = Vector{T}(undef,16)
ys = Vector{T}(undef,16)
for i in axes(env.swimmers,1)
    xs[i],ys[i] = env.swimmers[i].position
end
dist = Matrix{T}(undef, n_swimmers, n_swimmers)
@. dist = sqrt((xs - xs')^2 + (ys - ys')^2)
dist[diagind(dist)] .= 1.0
length(findall(x-> x < env.params.D/2, dist)) > 0

plot()
for sw in env.swimmers
    plot!([sw.position[1]],[sw.position[2]], st=:scatter)
end
plot!
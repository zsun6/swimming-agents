using ReinforcementLearning
using Random
using StableRNGs
using Flux
using Flux.Losses
using Plots
using Zygote
using Statistics
"""
    A2CLearner(;kwargs...)

# Keyword arguments

- `approximator`::[`ActorCritic`](@ref)
- `γ::Float32`, reward discount rate.
- `actor_loss_weight::Float32`
- `critic_loss_weight::Float32`
- `entropy_loss_weight::Float32`
- `update_freq::Int`, usually set to the same with the length of trajectory.
"""

include("..\\src\\SwimmerEnv.jl")

function (learner::A2CLearner)(env::SwimmingEnv)
    learner.approximator.actor(send_to_device(device(learner), state(env))) |> send_to_host
end
function (learner::A2CLearner)(env::SwimmingEnv, player)
    learner.approximator.actor(send_to_device(device(learner), state(env, player))) |> send_to_host
end
function (learner::A2CLearner)(env::SwimmingEnv)
    s = state(env)
    s = Flux.unsqueeze(s, ndims(s) + 1)
    s = send_to_device(device(learner), s)
    learner.approximator.actor(s) |> vec |> send_to_host
end
function (learner::A2CLearner)(env::SwimmingEnv, player)
    s = state(env,player)
    s = Flux.unsqueeze(s, ndims(s) + 1)
    s = send_to_device(device(learner), s)
    learner.approximator.actor(s) |> vec |> send_to_host
end

##run function
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

function RLBase.update!(learner::A2CLearner, t::CircularArraySARTTrajectory)
    length(t) == 0 && return  # in the first update, only state & action is inserted into trajectory
    learner.update_step += 1
    if learner.update_step % learner.update_freq == 0
        _update!(learner, t)
    end
end

function _update!(learner::A2CLearner, t::CircularArraySARTTrajectory)
    n = length(t)

    AC = learner.approximator
    γ = learner.γ
    w₁ = learner.actor_loss_weight
    w₂ = learner.critic_loss_weight
    w₃ = learner.entropy_loss_weight
    to_device = x -> send_to_device(device(AC), x)

    S = t[:state] |> to_device
    states = select_last_dim(S, 1:n)
    states_flattened = flatten_batch(states) # (state_size..., n_thread * update_freq)

    actions = select_last_dim(t[:action], 1:n)
    actions = flatten_batch(actions)
    actions = CartesianIndex.(actions, 1:length(actions))

    next_state_values = S |> select_last_frame |> AC.critic |> send_to_host

    gains =
        discount_rewards(
            t[:reward],
            γ;
            dims = 2,
            init = send_to_host(next_state_values),
            terminal = t[:terminal],
        ) |> to_device

    ps = Flux.params(AC)
    gs = gradient(ps) do
        logits = AC.actor(states_flattened)
        probs = softmax(logits)
        log_probs = logsoftmax(logits)
        log_probs_select = log_probs[actions]
        values = AC.critic(states_flattened)
        advantage = vec(gains) .- vec(values)
        actor_loss = -mean(log_probs_select .* Zygote.dropgrad(advantage))
        critic_loss = mean(advantage .^ 2)
        entropy_loss = -sum(probs .* log_probs) * 1 // size(probs, 2)
        loss = w₁ * actor_loss + w₂ * critic_loss - w₃ * entropy_loss
        ignore() do
            learner.actor_loss = actor_loss
            learner.critic_loss = critic_loss
            learner.entropy_loss = entropy_loss
            learner.loss = loss
        end
        loss
    end
    if !isnothing(learner.max_grad_norm)
        learner.norm = clip_by_global_norm!(gs, ps, learner.max_grad_norm)
    end
    update!(AC, gs)
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




RLCore.check(::QBasedPolicy{<:A2CLearner}, ::MultiThreadEnv) = nothing
#trial below
N_ENV = 16
UPDATE_FREQ = 10
n_swimmers=4
# wrappedenv = discrete2standard_discrete(env)
player = 1 # or 2
ns, na = length(state(env, player)), length(action_space(env,  player))
rng = StableRNG(abs(rand(Int)))
env = SwimmingEnv(;n_swimmers = n_swimmers, max_steps = 200, target=[0,70],freestream=[0,-1,2])
# RLBase.reset!(env, is_force = true)
agent = Agent(
    policy = QBasedPolicy(
        learner = A2CLearner(
            approximator = ActorCritic(
                actor = Chain(
                    Dense(ns, 256, relu; init = glorot_uniform(rng)),
                    Dense(256, na; init = glorot_uniform(rng)),
                ),
                critic = Chain(
                    Dense(ns, 256, relu; init = glorot_uniform(rng)),
                    Dense(256, 1; init = glorot_uniform(rng)),
                ),
                optimizer = ADAM(1e-3),
            ) |> gpu,
            γ = 0.99f0,
            actor_loss_weight = 1.0f0,
            critic_loss_weight = 0.5f0,
            entropy_loss_weight = 0.001f0,
            update_freq = UPDATE_FREQ,
        ),
        explorer = BatchExplorer(GumbelSoftmaxExplorer()),
    ),
    trajectory = CircularArraySARTTrajectory(;
        capacity = UPDATE_FREQ,
        state = Matrix{Float32} => (ns, N_ENV),
        action = Vector{Int} => (N_ENV,),
        reward = Vector{Float32} => (N_ENV,),
        terminal = Vector{Bool} => (N_ENV,),
    ),
)

action_space(env)
(env)((1,1,1,1))

env |> state # state(env)
env |> state |> agent.policy.learner.approximator.actor
state(env,1) |> agent.policy.learner.approximator.actor #!good, do it n_swimmers times


##REWRITE q_based_policy.jl ##
function (π::QBasedPolicy)(env::SwimmingEnv, ::MinimalActionSet, ::Base.OneTo) 
    actions = []
    for player in players(env) 
        push!(actions, π.explorer(π.learner(env,player)))
    end
    actions |> Tuple
end


Base.@kwdef mutable struct NewHook <: AbstractHook    
    path::Vector = []
    states:: Vector = []
    reward = 0.0
    rewards::Vector = []
    is_display_on_exit::Bool = true
end
function (h::NewHook)(::PostEpisodeStage, policy, env) 
    push!(h.states, h.path)
    push!(h.rewards,h.reward)
    h.reward = 0
    h.path = []
end
function (h::NewHook)(::PostActStage, policy, env)
    push!(h.path,vcat([sw.position for sw in env.swimmers]...))
    h.reward += sum(reward(env))
end
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
stop_condition =StopAfterEpisode(10_000; cur = 0, is_show_progress = true) 
# StopAfterStep(500_000, is_show_progress=!haskey(ENV, "CI")) # 
# hook = TotalRewardPerEpisode()
hook = NewHook()
hook = PotentialHook()

ex = Experiment(agent, env, stop_condition, hook, "")
run(ex)
if !isempty(ex.hook.rewards)
    for pos in ex.hook.states[argmax(ex.hook.rewards)]
        # Rest of your code here...
    end
else
    println("ex.hook.rewards is empty!")
end
rew = zeros(length(ex.hook.rewards)÷n_swimmers)
for i= 1:n_swimmers
	rew[:] += ex.hook.rewards[i:n_swimmers:end]
end
rew ./= n_swimmers
plot(rew)
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
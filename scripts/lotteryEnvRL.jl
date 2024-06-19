using ReinforcementLearning

Base.@kwdef mutable struct LotteryEnv <: AbstractEnv
    reward::Union{Nothing, Int} = nothing
end

RLBase.action_space(env::LotteryEnv) = (:PowerRich, :MegaHaul, nothing)

RLBase.reward(env::LotteryEnv) = env.reward
RLBase.state(env::LotteryEnv) = !isnothing(env.reward)
RLBase.state_space(env::LotteryEnv) = [false, true]
RLBase.is_terminated(env::LotteryEnv) = !isnothing(env.reward)
RLBase.reset!(env::LotteryEnv) = env.reward = nothing

function (x::LotteryEnv)(action)
    if action == :PowerRich
        x.reward = rand() < 0.01 ? 100_000_000 : -10
    elseif action == :MegaHaul
        x.reward = rand() < 0.05 ? 1_000_000 : -10
    elseif isnothing(action) x.reward = 0
    else
        @error "unknown action of $action"
    end
end

env = LotteryEnv()
RLBase.test_runnable!(env)

n_episode = 10
#testing
for _ in 1:n_episode
    reset!(env)
    while !is_terminated(env)
        #grab the env's action_space \> pick a random value \> push that value back into the env (reward will kick out a value)
        env |> action_space |> rand |> env
    end
end

run(RandomPolicy(action_space(env)), env, StopAfterEpisode(1_000))
hook = TotalRewardPerEpisode()
run(RandomPolicy(action_space(env)), env, StopAfterEpisode(1_000), hook)
hook

using Flux: InvDecay

p = QBasedPolicy(
    learner = MonteCarloLearner(;
        approximator=TabularQApproximator(
            ;n_state = length(state_space(env)),
            n_action = length(action_space(env)),
            opt = InvDecay(1.0)
        )
    ),
    explorer = EpsilonGreedyExplorer(0.1)
)

# p(env)
Q(s,a ) = p.learner.approximator(s,a)
Q(1,1)
p.learner.approximator(1)    #Q(s,:)

# We need the env to return an integer for accessing state! We can wrap to do this
wrapped_env = ActionTransformedEnv(
        StateTransformedEnv(
            env;
            state_mapping= s -> s ? 1 : 2,
            state_space_mapping = _ -> Base.OneTo(2)
        );
        action_mapping = i -> action_space(env)[i],
        action_space_mapping = _ -> Base.OneTo(3),
    )

p(wrapped_env)

h = TotalRewardPerEpisode()
run(p, wrapped_env, StopAfterEpisode(1_000), h)
h.rewards

using InteractiveUtils
subtypes(RLBase.AbstractStateStyle)
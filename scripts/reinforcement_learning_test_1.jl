using ReinforcementLearning
using StableRNGs
using Flux
using Flux.Losses
using Plots

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
    push!(h.path, state(env))
    h.reward += sum(reward(env))
end
function RL.Experiment(
    ::Val{:JuliaRL},
    ::Val{:BasicDQN},
    ::Val{:CartPole},
    ::Nothing;
    seed = 123,
)
    rng = StableRNG(seed)
    env = CartPoleEnv(; T = Float32, rng = rng)
    ns, na = length(state(env)), length(action_space(env))

    policy = Agent(
        policy = QBasedPolicy(
            learner = BasicDQNLearner(
                approximator = NeuralNetworkApproximator(
                    model = Chain(
                        Dense(ns, 128, relu; init = glorot_uniform(rng)),
                        Dense(128, 128, relu; init = glorot_uniform(rng)),
                        Dense(128, na; init = glorot_uniform(rng)),
                    ) |> gpu,
                    optimizer = ADAM(),
                ),
                batch_size = 32,
                min_replay_history = 100,
                loss_func = huber_loss,
                rng = rng,
            ),
            explorer = EpsilonGreedyExplorer(
                kind = :exp,
                ϵ_stable = 0.01,
                decay_steps = 500,
                rng = rng,
            ),
        ),
        trajectory = CircularArraySARTTrajectory(
            capacity = 1000,
            state = Vector{Float32} => (ns,),
        ),
    )
    stop_condition = StopAfterStep(10_000, is_show_progress=!haskey(ENV, "CI"))
    # hook = TotalRewardPerEpisode()
    hook = NewHook()
    Experiment(policy, env, stop_condition, hook, "# BasicDQN <-> CartPole")
end

# pyplot() #hide
ex = E`JuliaRL_BasicDQN_CartPole`
run(ex)
plot(ex.hook.rewards)
begin
plot()
anim = @animate for pos in ex.hook.states[1]
    plot!([0,pos[1]],[0,pos[2]],aspect_ratio=:equal,label="")
    plot!([pos[1]],[pos[2]],st=:scatter,aspect_ratio=:equal,label="")
end
gif(anim)
end
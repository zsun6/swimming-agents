using ReinforcementLearning
using Random
using StableRNGs
using Flux
using Flux.Losses
using Plots


Base.@kwdef mutable struct DistRewardPerEpisode <: AbstractHook    
    dists::Vector = []
    position::Vector = []
    positions:: Vector = []
    rewards::Vector = []
    gammas::Vector = []
    reward = 0.0
    is_display_on_exit::Bool = true
end

function (h::DistRewardPerEpisode)(::PostEpisodeStage, policy, env) 
    push!(h.dists, dist_angle(env))
    push!(h.positions, h.position)
    push!(h.gammas, env.swimmer.gamma)    
    push!(h.rewards,h.reward)
    h.reward = 0
    h.position = []
end
function (h::DistRewardPerEpisode)(::PostActStage, policy, env)
    # push!(h.reward ,reward(env))
    h.reward += reward(env)
    push!(h.position, env.swimmer.position)
end

include(".\\SwimmerEnv.jl")
function RL.Experiment(
    ::Val{:JuliaRL},
    ::Val{:BasicDQN},
    # ::Val{:PendulumDiscrete},
    ::Val{:LTSDiscrete},
    ::Nothing;
    seed = 23,
)
    rng = StableRNG(seed)
    # env2 = PendulumEnv(continuous = false, max_steps = 5000, rng = rng)
    env = SwimmerEnv(max_steps = 100, target=[0,0])
    ns, na = length(state(env)), length(action_space(env))
    agent = Agent(
        policy = 
            # RandomPolicy(),
            QBasedPolicy(
            # learner = BasicDQNLearner(
            #     approximator = NeuralNetworkApproximator(
            #         model = Chain(
            #             Dense(ns, 64, relu; init = glorot_uniform(rng)),
            #             Dense(64, 64, relu; init = glorot_uniform(rng)),
            #             Dense(64, na; init = glorot_uniform(rng)),
            #         ) |> gpu,
            #         optimizer = ADAM(),
            #     ),
            #     batch_size = 32,
            #     min_replay_history = 100,
            #     loss_func = huber_loss,
            #     rng = rng,
            # ),
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
            capacity = 50_000,
            state = Vector{Float32} => (ns,),
        ),
    )

    stop_condition = StopAfterStep(50_000, is_show_progress=!haskey(ENV, "CI"))
    hook = DistRewardPerEpisode() #TotalRewardPerEpisode()
    # hook = StepsPerEpisode()
    Experiment(agent, env, stop_condition, hook, "LTS")
end

ex = E`JuliaRL_BasicDQN_LTSDiscrete`

run(ex)
plot(ex.hook.rewards)
plot(ex.hook.rewards,marker=:dot)
plot(ex.hook.gammas ,label="")

rs = [r for (r,t) in ex.hook.dists]#./(ex.env.observation_space[1].right)
ts = [t for (r,t) in ex.hook.dists]
plot(rs)
plot!(ts)


begin 
    #assume the policy is good, run it through the motions
    env = SwimmerEnv(max_steps = 500, target=[1,1])
    h = DistRewardPerEpisode()
    run(ex.policy,env,StopAfterEpisode(50),h)
end

begin
    #OUTPUT with a max of eps
    S = state_space(ex.env)
    R = range(S[1].left, stop=S[1].right, length= 51)
    Θ = range(S[2].left, stop=S[2].right, length= 73)
    function show_approx(n)
        run(ex.policy,ex.env, StopAfterEpisode(n))
        [ex.policy.policy.learner.target_approximator([p, v]) |> maximum for p in R, v in Θ]
    end

    n = 1
    run(ex.policy,ex.env, StopAfterEpisode(n))
    field = -[ex.policy.policy.learner.approximator([p, v]) |> maximum for p in R, v in Θ]
    ofield = -[ex.policy.policy.learner.target_approximator([p, v]) |> maximum for p in R, v in Θ]
    
    # plot(R,Θ, field , linetype=:wireframe,
        # xlabel="R", ylabel="θ", zlabel="cost", title="Episode $n")
    plot(R,Θ, field' , linetype=:contourf,
    ylabel="θ", xlabel="R",  title="Episode $n",c=:thermal)

    hm = heatmap(field, aspect_ratio=:equal, proj=:polar,yaxis=false,c=:coolwarm)

end



begin 
plot([ex.env.target[1]],[ex.env.target[2]],st=:scatter,marker=:star,color=:green,label="target")
anim = @animate for pos in ex.hook.positions[266]
        plot!([pos[1]],[pos[2]],st=:scatter,aspect_ration=:equal,label="")
end
gif(anim)

end

begin
    i = 99#argmax(ex.hook.rewards)
xs = []
ys = []
for pos in ex.hook.positions[i]
    push!(xs,pos[1])
    push!(ys,pos[2])
end
plot([ex.env.target[1]],[ex.env.target[2]],st=:scatter,marker=:star,color=:green,label="target")
plot!([xs[1]],[ys[1]],marker=:circle,st=:scatter,color=:green,label="start")
plot!(xs,ys,label=ex.hook.rewards[i])
plot!([xs[end]],[ys[end]],marker=:circle,st=:scatter,color=:red, label="end")
end

begin
"""
Bullshit for testing 
"""
    T = Float32
    act = env |> ex.policy
    plot([0],[0],st=:scatter,marker=:star,label="start")
    for a = 1:5
        env.swimmer = FD_agent(Vector{T}([0.0,0.0]),Vector{T}([env.params.Γ0,-env.params.Γ0]), T(π/2), Vector{T}([0.0,0.0]),Vector{T}([0.0,0.0]))        
        (env)(a)
        # @show env.swimmer
        plot!([env.swimmer.position[1]],[env.swimmer.position[2]],st=:scatter,label=a)
    end
    plot!(aspect_ratio=:equal)
end

begin
    """
    look at turning radius
    """
        T = Float32
        act = env |> ex.policy
        plot([0],[0],st=:scatter,marker=:star,label="start")
        env.swimmer = FD_agent(Vector{T}([0.0,0.0]),Vector{T}([-env.params.Γ0, env.params.Γ0]), T(π/2), Vector{T}([0.0,0.0]),Vector{T}([0.0,0.0]))        
        for t = 1:1000
            
            (env)(4)
            # @show env.swimmer
            plot!([env.swimmer.position[1]],[env.swimmer.position[2]],st=:scatter,label="")
        end
        plot!(aspect_ratio=:equal)
end
    

begin
type = Float32
boids = [ex.env.swimmer]
xs = LinRange{type}(-0.025,0.01,31)
ys = LinRange{type}(-0.025,0.01,31)
#do it a different way
X = repeat(reshape(xs, 1, :), length(ys), 1)
Y = repeat(ys, 1, length(xs))
targets = [X,Y]
stream = streamfunction(boids, targets;ℓ=ex.env.params.ℓ)
plot(collect(xs),collect(ys), stream, st=:contourf)

end
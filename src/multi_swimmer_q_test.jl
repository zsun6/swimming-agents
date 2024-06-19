using ReinforcementLearning
using Random
using StableRNGs
using Flux
using Flux.Losses
using Plots



include(".\\SwimmerEnv.jl")
n_swimmers = 16
env = SwimmingEnv(;n_swimmers = n_swimmers, max_steps = 500, target=[0,30])
# wrappedenv = discrete2standard_discrete(env)
player = 1 # or 2
ns, na = length(state(env, player)), length(action_space(env,  player))
rng = StableRNG(17)
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
        #Nate: changed the trajectory to hold all states for now -> probably 
        # change the push!(trajectory[:state], s) in agent.jl
        state = Vector{Float32} => (ns,),
    )
)

stop_condition = StopAfterStep(500_000, is_show_progress=!haskey(ENV, "CI"))
# hook = TotalRewardPerEpisode()
hook = NewHook()

ex = Experiment(agent, env, stop_condition, hook, "")

# run(ex)
#work on MD for this, typing issue at this level
run(ex)


#LEAVE THIS BE AND COMPOSE AFTER APPLICATION OF explorer per action
function (learner::DQNLearner)(env::SwimmingEnv)
    #SHARED Q between all agents
    # grab all actions per each agents position
    # actions = []
    # for player in players(env)
    #     # @show state(env,player) 
    #     state(env,player) |>   x ->
    #         Flux.unsqueeze(x, ndims(x) + 1) |>     
    #         x ->
    #             send_to_device(device(learner), x) |>       
    #             learner.approximator |>
    #             vec |> findmin |> x -> push!(actions, x[2])
    # end
    # actions|>Tuple
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
    hook.reward += sum(reward(env))
end
""" the below is achieved by running the internals of the above function
 
learner = agent.policy.learner
actions = []
for player in players(env)
    state(env,player) |>   x ->
        Flux.unsqueeze(x, ndims(x) + 1) |>            
                learner.approximator |>
                vec |> findmax |> x -> push!(actions, x[2])
end
actions|>Tuple
"""
(env)(actions)
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
           
            # if action !=(1,1,1,1,1,1)
            #     println(action)
            # end
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
    for player in players(en) 
        push!(actions, π.explorer(π.learner(env,player)))
    end
    actions |> Tuple
end





# run(ex)
s1,s2,s3 = ex.hook.rewards[1:3:end],ex.hook.rewards[2:3:end],ex.hook.rewards[3:3:end]
plot(s1)
plot!(s2)
plot!(s3)


#NEW HOOK PLOTS
begin    
    rs = []
    thetas = []
    a = plot([env.target[1]],[env.target[2]], st=:scatter, marker=:star, ms=4)
    for pos in ex.hook.states[140]
        push!(rs, pos[1:n_swimmers:end]...)
        push!(thetas,pos[2:n_swimmers:end]...)        
    end
    
    for i = 1:length(rs)
        rs[i],thetas[i] = state2xy(rs[i],thetas[i])
    end

    for i = 1:n_swimmers        
        plot!(a, rs[i:n_swimmers:end], thetas[i:n_swimmers:end], lw= 1, label="")        
    end
    a
end
plot(xs[i:n_swimmers:end], ys[i:n_swimmers:end], label="")
plot([ex.hook.rewards[i:n_swimmers:end] for i=1:n_swimmers])

argmax(ex.hook.rewards)


####SCRATCH BELOW CODE#####

agent.policy.learner(env)
agent.policy(env)    
""" it fails with error 
 DimensionMismatch: A has dimensions (64,2) but B has dimensions (6,1)
 Lets think about what we want for our Learner -> Q 
    1. Q is the same network for all swimmers to start so it should have a shape
       2 -> Magic# ->...->Magic# -> 5
       where 2 is the r,θ of a current swimmer
       Magic#  is some dimension of the NN 
       5 is the number of actions
    2. in the future we can expand on this to have competing Qs, but for now just one

 We need to make a swimmer manager in order to have each swimmer -> query Q for its actions individually and 
 then compose those actions into a set to update the environment, then the environment costs are used to update
 the Q approximator on the other end. 
"""
"""
We can probably pull this off by changing the guts of the _run function
    https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/master/src/ReinforcementLearningCore/src/core/run.jl
 and make it into 
 function Base.run(
    policy::AbstractPolicy,
    env::SwimmingEnv,
    stop_condition=StopAfterEpisode(1),
    hook=EmptyHook(),
    reset_condition=ResetAtTerminal()
)

or in all honesty, it might be handled in how we treat the policy itself
https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/main/src/ReinforcementLearningCore/src/policies/q_based_policy.jl

once again, rewrite the functions to operate on our environment not the generic
and make sure it iterates through the swimmers here
"""


pen = PendulumEnv(continuous = false, max_steps = 5000, rng = rng)
nns, nna = length(state(pen)), length(action_space(pen))
pagent = Agent(
        policy = QBasedPolicy(
            learner = BasicDQNLearner(
                approximator = NeuralNetworkApproximator(
                    model = Chain(
                        Dense(nns, 64, relu; init = glorot_uniform(rng)),
                        Dense(64, 64, relu; init = glorot_uniform(rng)),
                        Dense(64, nna; init = glorot_uniform(rng)),
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
            capacity = 5_000,
            state = Vector{Float32} => (ns,),
        ),
    )

pen |> pagent.policy

state(pen) |>   x ->
        Flux.unsqueeze(x, ndims(x) + 1) |>            
                agent.policy.learner.approximator |>vec
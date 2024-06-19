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
    push!(h.action, env.action[:])
    push!(h.angle, vcat([sw.angle for sw in env.swimmers]...))
    h.reward += sum(reward(env))
end

#modified functions for plotting
function get_circulation(as::Vector{Int})
    """ Agent starts with circulation of from
        [-swim.Γ0, swim.Γ0]
        so we have to inc/dec to change the magnitude not absolutes
        [CRUISE, FASTER, SLOWER, LEFT, RIGHT]"""
    gammas = zeros(2, length(as))
    for (i,a) in enumerate(as)
        if a == 1
            gammas[:,i] = [-env.params.Γ0, env.params.Γ0]
        elseif a == 2  
            gammas[:,i] = [-env.params.Γ0 - env.params.Γa, env.params.Γ0 + env.params.Γa]
        elseif a == 3  
            gammas[:,i] = [-env.params.Γ0 + env.params.Γa, env.params.Γ0 - env.params.Γa]
        elseif a == 4  
            gammas[:,i] = [-env.params.Γ0 - env.params.Γt, env.params.Γ0 - env.params.Γt]
        elseif a == 5  
            gammas[:,i] = [-env.params.Γ0 + env.params.Γt, env.params.Γ0  + env.params.Γt]
        else 
            @error "unknown action of $a"
        end        
    end    
    gammas
end
get_circulation(as::Vector{T}) where T<: Any = get_circulation(as.|>Int)

dlx(pos, angle; ℓ=2.5e-4) = pos[1] + ℓ/2.0*cos(angle+π/2)
dly(pos, angle; ℓ=2.5e-4) = pos[2] + ℓ/2.0*sin(angle+π/2)
drx(pos, angle; ℓ=2.5e-4) = pos[1] + ℓ/2.0*cos(angle-π/2)
dry(pos, angle; ℓ=2.5e-4) = pos[2] + ℓ/2.0*sin(angle-π/2)

function potential(coms::Vector{T}, angles, targets, actions; ℓ=2.5e-4) where T<: Real
    """
    find vortex streamlines from sources to a LazyGrids of targets
    mainly for plotting, but might be worth examining wtih Autodiff
    """
    # ℓ = env.params.ℓ
    # T = typeof(ℓ)
    n_swimmers = length(angles)
    pot = zeros(T, (size(targets[1])...))
    gamma = get_circulation(actions)
    for i=1:n_swimmers
        com = coms[(i-1)*2+1:(i-1)*2+2]
        #left vortex
        dx = targets[1] .- dlx(com, angles[i] ;ℓ)
        dy = targets[2] .- dly(com, angles[i] ;ℓ)
        @. pot += -gamma[1,i] *log(sqrt(dx^2+dy^2))
        #right vortex
        dx = targets[1] .- drx(com, angles[i];ℓ)
        dy = targets[2] .- dry(com, angles[i];ℓ)
        @. pot += -gamma[2,i] *log(sqrt(dx^2+dy^2))
    end
    pot./(2π)
end


begin
    #create sample to test
    ep = 101# argmax(ex.hook.rewards)
    angles = ex.hook.angles[ep]
    states = ex.hook.states[ep]
    actions = ex.hook.actions[ep]

    #a swimmer going up from origin
    xs = LinRange(-0.15,0.15,201)
    ys = LinRange(-0.05,0.25,201)
    X = repeat(reshape(xs, 1, :), length(ys), 1)
    Y = repeat(ys, 1, length(xs))
    targets = [X,Y]
    #climits
    g = env.params.Γ0
    dx = minimum(diff(xs))
    cl = [-1,1].*(g*dx) |>Tuple
    anim = @animate for i = 1:length(angles)
        pot = potential(states[i], angles[i], targets, actions[i]; env.params.ℓ)
        f = plot(collect(xs),collect(ys), pot, st=:contour)#, clim=cl)
        plot!(f, states[i][1:2:end], states[i][2:2:end], st=:scatter)
        f
    end
    gif(anim)
end

function streamlines_gif(ex, ep, xs, ys ; cl = nothing)
    angles = ex.hook.angles[ep]
    states = ex.hook.states[ep]
    actions = ex.hook.actions[ep]

    X = repeat(reshape(xs, 1, :), length(ys), 1)
    Y = repeat(ys, 1, length(xs))
    targets = [X,Y]
    g = ex.env.params.Γ0
    dx = minimum(diff(xs))
    # cl = g*dx.*10
    anim = @animate for i = 1:length(angles)
        pot = potential(states[i], angles[i], targets, actions[i]; env.params.ℓ)
        f = plot(collect(xs),collect(ys), pot, st=:contour, levels=40)
        if !isnothing(cl)
            plot!(f, clim=(-cl,cl))
        end
        plot!(f, states[i][1:2:end], states[i][2:2:end], st=:scatter)
        f
    end
    gif(anim)
end

streamlines_gif(ex, argmax(ex.hook.rewards), LinRange(-0.15,0.15,101),LinRange(0,0.3,101);cl=ex.env.params.Γ0/100.)

p = plot()
for act in actions
    plot!(p, act, st=:scatter,label="")
end
p
function streamlines_gif(ex, ep, xs, ys ; cl = nothing)
    angles = ex.hook.angles[ep]
    states = ex.hook.states[ep]
    actions = ex.hook.actions[ep]
    
    X = repeat(reshape(xs, 1, :), length(ys), 1)
    Y = repeat(ys, 1, length(xs))
    targets = [X,Y]
    g = ex.env.params.Γ0
    dx = minimum(diff(xs))
    # cl = g*dx.*10

    anim = @animate for i = 1:length(angles)
        pot = potential(states[i], angles[i], targets, actions[i]; env.params.ℓ)
        f = plot(collect(xs),collect(ys), pot, st=:contour, levels=40)
        plot!([ex.env.target[1]],[ex.env.target[2]], st=:scatter, label="Target")
        
        if !isnothing(cl)
            plot!(f, clim=(-cl,cl))
        end

        plot!(f, states[i][1:2:end], states[i][2:2:end], st=:scatter, label="Swimmers")
        f
    end
    gif(anim)
end


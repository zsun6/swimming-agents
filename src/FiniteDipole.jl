#NATE: GAVE UP on modules
# module FiniteDipole

using LinearAlgebra
Base.@kwdef mutable struct FD_agent{T<:Real}
    position::Vector{T} #2-D to start
    gamma::Vector{T} #Γ_l Γ_r
    angle::T #α
    v::Vector{T} #velocity
    a::Vector{T} #acceleration
end 

Base.@kwdef mutable struct Swimmer_params
    #parameters for specific set of swimming agents
    ℓ #dipole length - vortex to vortex
    Γ₀ #init circulation
    Γₐ #incremental circulation
    v₀ #cruise velocity
    vₐ #incremental velocity
    ρ  #turn radius    
end 
#Default constructor uses
Swimmer_params(ℓ) = Swimmer_params(ℓ/2.0, 10*π*ℓ^2, π*ℓ^2, 5*ℓ, ℓ/2.0, 10ℓ)

Base.show(io::IO,b::FD_agent) = print(io,"FD_agent (x,y)=$(b.position),
\t (θ) = $(b.angle),
\t (Γl,Γr) = $(b.gamma),
\t (u,v) = $(b.v))")

#dipole left and right gamma x and Y
#dipole <left/right> <x/y> ==>d[l/r][x/y]
dlx(b::FD_agent;ℓ=5e-4) = b.position[1] + ℓ/2.0*cos(b.angle+π/2)
dly(b::FD_agent;ℓ=5e-4) = b.position[2] + ℓ/2.0*sin(b.angle+π/2)
drx(b::FD_agent;ℓ=5e-4) = b.position[1] + ℓ/2.0*cos(b.angle-π/2)
dry(b::FD_agent;ℓ=5e-4) = b.position[2] + ℓ/2.0*sin(b.angle-π/2)


function agent_to_target(boids::Vector{FD_agent{T}},targets; ℓ=5e-4) where T<:Real
    n = size(boids)[1]
    vels = zeros(T, (2,n))
    vel = zeros(T, n)
    for b in boids            
        dx = targets[1,:] .-  dlx(b; ℓ)
        dy = targets[2,:] .-  dly(b; ℓ)
        @. vel = b.gamma[1]  / (2π *(dx^2 + dy^2 ))
        @. vels[1,:,:] += dy * vel
        @. vels[2,:,:] -= dx * vel
        dx = targets[1,:] .- drx(b; ℓ)
        dy = targets[2,:] .- dry(b; ℓ)
        @. vel = b.gamma[2] / (2π *(dx^2 + dy^2 ))
        @. vels[1,:,:] += dy * vel
        @. vels[2,:,:] -= dx * vel
    end
    vels
end

function vortex_to_swimmer_midpoint_velocity(boids::Vector{FD_agent{T}} ;ℓ=5e-4) where T<:Real
    """
    find vortex velocity from sources to the midpoint of the swimmer
    sources - vector of vortex_particle type
    DOES NOT UPDATE STATE of sources
    returns velocity induced field
    """  

    for (i,b) in enumerate(boids)
        targets[:,i] = b.position
    end
    agent_to_target(boids,targets; ℓ)
end

function vortex_to_vortex_velocity(boids::Vector{FD_agent{T}} ;ℓ=5e-4) where T<:Real
    """
    find vortex to vortex velocity interactions    
    DOES NOT UPDATE STATE of sources
    returns velocity induced field split corresponding to left and right vortices
    """  
    n =size(boids)[1]
    vels    = zeros(T, (2,2*n))
    vel     = zeros(T, 2*n) 

    targets = [[dlx.(boids;ℓ) dly.(boids;ℓ)]' [drx.(boids;ℓ) dry.(boids;ℓ)]']

    # dx = [(targets[1,i] - targets[1,j]) for i in 1:n, j in 1:n]
    # dy = [(targets[2,i] - targets[2,j]) for i in 1:n, j in 1:n]
    # d2 = (dx.^2 + dy.^2)
    # d2inv = 1.0./d2
    # foreach(i -> @inbounds(d2inv[i, i] = 0.0), 1:n)
    # for (i,b) in enumerate(boids)
    #     vels[1,:] += sum(b.gamma[1]/(2π) *dy.*d2inv, dims = 2)
    #     vels[2,:] += sum(b.gamma[1]/(2π) *dx.*d2inv, dims = 2)
    # end

    for i in 1:n            
        dx = targets[1,:] .- targets[1,i] #lefts[1,i] 
        dy = targets[2,:] .- targets[2,i] #lefts[2,i] 
        @. vel = boids[i].gamma[1]  / (2π *(dx^2 + dy^2 ))   
        vel[i] = 0.0 #singularity following the left vortex
        # vel[i+n] = 0.0 #NATE TODO
        @. vels[1,:] += dy * vel
        @. vels[2,:] -= dx * vel
        dx = targets[1,:] .- targets[1, n+i] #rights[1,i] 
        dy = targets[2,:] .- targets[2, n+i] #rights[2,i] 
        @. vel = boids[i].gamma[2] / (2π *(dx^2 + dy^2 ))
        vel[i+n] = 0.0 # singularity follows the right vortex position
        # vel[i] = 0.0   # #NATE TODO self interactions are found in another function
        @. vels[1,:] += dy * vel
        @. vels[2,:] -= dx * vel
    end
    vels #the first n are for the left and second n are for the right
end

function angle_projection(boids::Vector{FD_agent{T}},lrvel::Matrix{T}; ℓ=5e-4) where T<:Real
    """
        find the change it angle for a swimmer based on the velocity difference 
        found across left and right vortices
    """
    n = length(boids)[1]
    #TODO : add in another parameter for background flow 
    left,right = lrvel[:,1:n],lrvel[:,n+1:end]
    #project the velocity difference onto the induced velocity of the swimmer     
    alphadot = zeros(T,n)
    for (i,b) in enumerate(boids)
        alphadot[i]  =  (right[:,i] - left[:,i])⋅[cos(b.angle), sin(b.angle)]/(ℓ) - sum(b.gamma)/(2π*ℓ^2)  
    end
    alphadot 
end

function self_induced_velocity(boids::Vector{FD_agent{T}}; ℓ=5e-4) where T<:Real
    """
    Find velocity that an agent has onto itself
    """
    vel =  zeros(T, (2,length(boids)[1]))
    for (i,b) in enumerate(boids)
        vel[:,i] = diff(b.gamma).*[cos(b.angle), sin(b.angle)]/(2π*ℓ)
    end
    vel
end

function vortex_to_grid_velocity(boids ::Vector{FD_agent{T}}, targets  ;ℓ=5e-4) where T<:Real
    """
    find vortex velocity from sources to a LazyGrids of targets
    """

    n = size(boids)[1]
    vels = zeros(T, (2,size(targets[1])...))
    vel = zeros(T, size(targets[1]))
    for i in 1:n            
        #left vortex
        dx = targets[1] .- dlx(b;ℓ)
        dy = targets[2] .- dly(b;ℓ)
        @. vel = boids[i].gamma[1]  / (2π *(dx^2 + dy^2 ))
        @. vels[1,:,:] += dy * vel
        @. vels[2,:,:] -= dx * vel
        #right vortex
        dx = targets[1] .- drx(b;ℓ)
        dy = targets[2] .- dry(b;ℓ)
        @. vel = boids[i].gamma[2] / (2π *(dx^2 + dy^2 ))
        @. vels[1,:,:] += dy * vel
        @. vels[2,:,:] -= dx * vel
    end
    vels
end

function potential(boids::Vector{FD_agent{T}},targets; ℓ=5e-4) where T<: Real
    """
    find vortex potential from sources to a LazyGrids of targets
    mainly for plotting, but might be worth examining wtih Autodiff
    """
    pot = zeros(T, (size(targets[1])...))

    for b in boids            
        #left vortex
        dx = targets[1] .- dlx(b;ℓ)
        dy = targets[2] .- dly(b;ℓ)
        @. pot += -b.gamma[1] *atan(dx,dy)
        #right vortex
        dx = targets[1] .- drx(b;ℓ)
        dy = targets[2] .- dry(b;ℓ)
        @. pot += -b.gamma[2] *atan(dx,dy)
    end
    pot./(2π)
end

function streamfunction(boids::Vector{FD_agent{T}},targets; ℓ=5e-4) where T<: Real
    """
    find vortex streamlines from sources to a LazyGrids of targets
    mainly for plotting, but might be worth examining wtih Autodiff
    """
    pot = zeros(T, (size(targets[1])...))

    for b in boids            
        #left vortex
        dx = targets[1] .- dlx(b;ℓ)
        dy = targets[2] .- dly(b;ℓ)
        @. pot += -b.gamma[1] *log(sqrt(dx^2+dy^2))
        #right vortex
        dx = targets[1] .- drx(b;ℓ)
        dy = targets[2] .- dry(b;ℓ)
        @. pot += -b.gamma[2] *log(sqrt(dx^2+dy^2))

    end
    pot./(2π)
end

function add_lr(v2v)
    """
    add together and average the velocity contributions at the left and right vortices
    """
    n = size(v2v)[2]÷2
    (v2v[:,1:n] + v2v[:,n+1:end])/2.
end

function move_swimmers!(boids::Vector{FD_agent{T}}; Δt=0.1, ℓ=5e-4) where T<: Real
    """ 
    Find the velocity induced from each swimmer onto the others and 
    update position via Euler's method 
    """    
    v2v = vortex_to_vortex_velocity(boids;ℓ)    
    avg_v = add_lr(v2v)
    siv = self_induced_velocity(boids;ℓ)
    angles = angle_projection(boids,v2v)
    ind_v = siv + avg_v #eqn 2.a
    

    for (i,b) in enumerate(boids)
        b.v = ind_v[:,i]
        b.position +=  ind_v[:,i] .* Δt
        b.angle += angles[i]  #eqn 2.b
    end

end

function change_bearing_speed(swim::Swimmer_params)
    """
    eqns 5.a,b 
    sim is a params struct
    """
    Γadd = swim.vₐ/swim.v₀ *swim.Γ₀
    ΓT   = swim.Γ₀/(swim.ρ/(sqrt(2*π)*swim.ℓ))
    Γadd, ΓT    
end

function test_moves(boids::Vector{FD_agent{T}}; Δt=0.1, ℓ=5e-4) where T<: Real
    """ 
    Find the velocity induced from each swimmer onto the others and 
    update position via Euler's method 
    """    
    v2v = vortex_to_vortex_velocity(boids;ℓ)    
    avg_v = add_lr(v2v)
    siv = self_induced_velocity(boids;ℓ)
    angles = angle_projection(boids,v2v)
    ind_v = siv + avg_v #eqn 2.a
    
    position = zeros(2,(length(boids)))
    angles = zeros(length(boids))
    for (i,b) in enumerate(boids)
        # b.v = ind_v[:,i]
        position[:,i] = b.position + ind_v[:,i] .* Δt
        angles[i] = b.angle + angles[i]  #eqn 2.b
    end

end
# end # module #NATE: GAVE UP on modules


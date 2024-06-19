module swimming_agents

export boid, params
export vortex_to_grid_velocity, vortex_to_swimmer_midpoint_velocity, vortex_to_vortex_velocity
export angle_projection, self_induced_velocity, potential, streamfunction
export add_lr, move_swimmers!, change_bearing_speed

Base.@kwdef mutable struct boid{T<:Real}
    position::Vector{T} #2-D to start
    gamma::Vector{T} #Γ_l Γ_r
    angle::T #α
    v::Vector{T} #velocity
    a::Vector{T} #acceleration
end 

Base.@kwdef mutable struct params
    ℓ #dipole length
    Γ₀ #init circulation
    Γₐ #incremental circulation
    v₀ #cruise velocity
    vₐ #incremental velocity
    ρ  #turn radius
    function params(ℓ)
        v0 = 5*ℓ
        g0 = 2π*ℓ*v0 
        ga = 0.1 *g0    
        new(ℓ,g0,ga,v0,v0*0.1,10ℓ)
    end 
end 
# Print function for a boid
Base.show(io::IO,b::boid) = print(io,"Boid (x,y,α,Γlr,v)=($(b.position),$(b.angle),$(b.gamma),$(b.v))")


function vortex_to_swimmer_midpoint_velocity(boids::Vector{boid{T}} ;ℓ=0.001) where T<:Real
    """
    find vortex velocity from sources to the midpoint of the swimmer
    sources - vector of vortex_particle type
    DOES NOT UPDATE STATE of sources
    returns velocity induced field
    """  

    n =size(boids)[1]
    vels = zeros(T, (2,n))
    vel = zeros(T, n)
    targets = zeros(T,(2,n))
    for (i,b) in enumerate(boids)
        targets[:,i] = b.position
    end
    # targets = [b.position[:] for b in boids]
    for i in 1:n            
        dx = targets[1,:] .- (boids[i].position[1] .+ ℓ*cos(boids[i].angle+π/2))
        dy = targets[2,:] .- (boids[i].position[2] .+ ℓ*sin(boids[i].angle+π/2))
        @. vel = boids[i].gamma[1]  / (2π *(dx^2 + dy^2 ))
        @. vels[1,:,:] += dy * vel
        @. vels[2,:,:] -= dx * vel
        dx = targets[1,:] .- (boids[i].position[1] .+ ℓ*cos(boids[i].angle-π/2))
        dy = targets[2,:] .- (boids[i].position[2] .+ ℓ*sin(boids[i].angle-π/2))
        @. vel = boids[i].gamma[2] / (2π *(dx^2 + dy^2 ))
        @. vels[1,:,:] += dy * vel
        @. vels[2,:,:] -= dx * vel
    end
    vels
end

function vortex_to_vortex_velocity(boids::Vector{boid{T}} ;ℓ=0.001) where T<:Real
    """
    find vortex to vortex velocity interactions    
    DOES NOT UPDATE STATE of sources
    returns velocity induced field split corresponding to left and right vortices
    """  
    n =size(boids)[1]
    vels    = zeros(T, (2,2*n))
    lefts   = zeros(T, (2,n))
    rights  = zeros(T, (2,n))
    # targets = zeros(T,(2,2n)) #(u,v) at (Γl and Γr)
    vel     = zeros(T, 2*n) 
    for (i,b) in enumerate(boids)
        lefts[:,i]  .= (b.position[1] .+ ℓ*cos(b.angle+π/2)),
                    (b.position[2] .+ ℓ*sin(b.angle+π/2))
        rights[:,i] .= (b.position[1] .+ ℓ*cos(b.angle-π/2)),
                    (b.position[2] .+ ℓ*sin(b.angle-π/2))        
    end
    #stack up the left and right vortices as the targets
    targets = [lefts rights]
    for i in 1:n            
        dx = targets[1,:] .- lefts[1,i] 
        dy = targets[2,:] .- lefts[2,i] 
        @. vel = boids[i].gamma[1]  / (2π *(dx^2 + dy^2 ))   
        vel[i] = 0.0 #singularity following the left vortex
        vel[i+n] = 0.0
        @. vels[1,:] += dy * vel
        @. vels[2,:] -= dx * vel
        dx = targets[1,:] .- rights[1,i] 
        dy = targets[2,:] .- rights[2,i] 
        @. vel = boids[i].gamma[2] / (2π *(dx^2 + dy^2 ))
        vel[i+n] = 0.0 # singularity follows the right vortex position
        vel[i] = 0.0   # self interactions are found in another function
        @. vels[1,:] += dy * vel
        @. vels[2,:] -= dx * vel
    end
    vels #the first n are for the left and second n are for the right
end



function angle_projection(boids::Vector{boid{T}},lrvel::Matrix{T};ℓ=0.001) where T<:Real
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
        alphadot[i]  = sum(b.gamma)/(2π*ℓ^2) +  (right[:,i] - left[:,i])⋅[cos(b.angle), sin(b.angle)]        
    end
    alphadot 
end



function self_induced_velocity(boids::Vector{boid{T}}; ℓ=0.001) where T<:Real
    """
    Find velocity that an agent has onto itself
    """
    vel =  zeros(T, (2,length(boids)[1]))
    for (i,b) in enumerate(boids)
        vel[:,i] = -diff(b.gamma).*[cos(b.angle), -sin(b.angle)]/(2π*ℓ)
    end
    vel
# [-diff(b.gamma).*[cos(b.angle), -sin(b.angle)]'/(2π*ℓ) for b in boids]
end



function vortex_to_grid_velocity(boids ::Vector{boid{T}}, targets  ;ℓ=0.001) where T<:Real
    """
    find vortex velocity from sources to a LazyGrids of targets
    """

    n = size(boids)[1]
    vels = zeros(T, (2,size(targets[1])...))
    vel = zeros(T, size(targets[1]))
    for i in 1:n            
        #left vortex
        dx = targets[1] .- (boids[i].position[1] .+ ℓ*cos(boids[i].angle+π/2))
        dy = targets[2] .- (boids[i].position[2] .+ ℓ*sin(boids[i].angle+π/2))
        @. vel = boids[i].gamma[1]  / (2π *(dx^2 + dy^2 ))
        @. vels[1,:,:] += dy * vel
        @. vels[2,:,:] -= dx * vel
        #right vortex
        dx = targets[1] .- (boids[i].position[1] .+ ℓ*cos(boids[i].angle-π/2))
        dy = targets[2] .- (boids[i].position[2] .+ ℓ*sin(boids[i].angle-π/2))
        @. vel = boids[i].gamma[2] / (2π *(dx^2 + dy^2 ))
        @. vels[1,:,:] += dy * vel
        @. vels[2,:,:] -= dx * vel
    end
    vels
end


function potential(boids::Vector{boid{T}},targets; ℓ= 0.001) where T<: Real
    """
    find vortex potential from sources to a LazyGrids of targets
    mainly for plotting, but might be worth examining wtih Autodiff
    """
    pot = zeros(T, (size(targets[1])...))

    for b in boids            
        #left vortex
        dx = targets[1] .- (b.position[1] .+ ℓ*cos(b.angle+π/2))
        dy = targets[2] .- (b.position[2] .+ ℓ*sin(b.angle+π/2))
        @. pot += -b.gamma[1] *atan(dx,dy)
        #right vortex
        dx = targets[1] .- (b.position[1] .+ ℓ*cos(b.angle-π/2))
        dy = targets[2] .- (b.position[2] .+ ℓ*sin(b.angle-π/2))
        @. pot += -b.gamma[2] *atan(dx,dy)
    end
    pot./(2π)
end

function streamfunction(boids::Vector{boid{T}},targets; ℓ= 0.001) where T<: Real
    """
    find vortex streamlines from sources to a LazyGrids of targets
    mainly for plotting, but might be worth examining wtih Autodiff
    """
    pot = zeros(T, (size(targets[1])...))

    for b in boids            
        #left vortex
        dx = targets[1] .- (b.position[1] .+ ℓ*cos(b.angle+π/2))
        dy = targets[2] .- (b.position[2] .+ ℓ*sin(b.angle+π/2))
        @. pot += -b.gamma[1] *log(sqrt(dx^2+dy^2))
        #right vortex
        dx = targets[1] .- (b.position[1] .+ ℓ*cos(b.angle-π/2))
        dy = targets[2] .- (b.position[2] .+ ℓ*sin(b.angle-π/2))
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

function move_swimmers!(boids::Vector{boid{T}}; Δt=0.1, ℓ= 0.001) where T<: Real
    """ 
    Find the velocity induced from each swimmer onto the others and 
    update position via Euler's method 
    """

    target = (10,0)
    v2v = vortex_to_vortex_velocity(boids;ℓ)
    siv = self_induced_velocity(boids;ℓ)
    avg_v = add_lr(v2v)
    ind_v = siv + avg_v #eqn 2.a
    angles = angle_projection(boids,v2v)
    a_desired = [atan(y,x) for (x,y) in [target .- b.position for b in boids]]
    adotdes = (a_desired .- [b.angle for b in boids])/Δt

    for (i,b) in enumerate(boids)
        b.v = ind_v[:,i]
        b.position +=  ind_v[:,i] .* Δt
        b.angle += angles[i]  +adotdes[i] #eqn 2.b
    end

    #Aokin- model

    # Γadd = π*ℓ^2*(adotdes - angles)
    # for (i,b) in enumerate(boids)
    #     b.gamma[1] += Γadd[i]  
    #     b.gamma[2] -= Γadd[i]
    # end
end


###### <---- transition state codes for pompds -----> #########
function change_bearing_speed(sim)
    """
    eqns 5.a,b 
    sim is a params struct
    """
    Γadd = sim.vₐ/sim.v₀ *sim.Γ₀
    ΓT   = sim.ρ/(2*π*sim.ℓ)
    Γadd,ΓT    
end


end # module

using LazyGrids
using Plots
using SparseArrays
# using PlotlyJS
using LinearAlgebra
include("..\\src\\FiniteDipole.jl")
using FiniteDipole
# TODO : Nate add in freestream velocity field calcs for determining if the 
# angle_projection methods work -add tests
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
Base.show(io::IO,b::boid) = print(io,"Boid (x,y,α)=($(b.position),$(b.angle)), ̂v = $(b.v)")
   
#SAMPLE constructorS
b = boid([1.0,0.5],[-1.0,1.0], π/2, [0.0,0.0],[0.0,0.0])
sim = params(5e-4)


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
    find vortex velocity from sources to themselves
    sources - vector of vortex_particle type
    DOES NOT UPDATE STATE of sources
    returns velocity induced field
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
        vel[i] = 0.0
        @. vels[1,:] += dy * vel
        @. vels[2,:] -= dx * vel
    end
    
    vels #the first n are for the left and second n are for the right
end


function angle_projection(boids::Vector{boid{T}};ℓ=0.001) where T<:Real
    """
        find the change it angle for a swimmer based on the velocity difference 
        found across left and right vortices
    """
    n = length(boids)[1]
    vel = vortex_to_vortex_velocity(boids;ℓ)
    left,right = vel[:,1:n],vel[:,n+1:end]
    #project the velocity difference onto the induced velocity of the swimmer 
    dir = zeros(T,(2,n))
    for (i,b) in enumerate(boids)
        dir[:,i]  .= [cos(b.angle), sin(b.angle)]
    end
    alphadot = (right - left) ⋅ dir 
end

function angle_projection(boids::Vector{boid{T}},lrvel::Matrix{T};ℓ=0.001) where T<:Real
    """
        find the change it angle for a swimmer based on the velocity difference 
        found across left and right vortices
    """
    n = length(boids)[1]

    left,right = lrvel[:,1:n],lrvel[:,n+1:end]
    #project the velocity difference onto the induced velocity of the swimmer     
    alphadot = zeros(T,n)
    for (i,b) in enumerate(boids)
        alphadot[i]  = sum(b.gamma)/(2π*ℓ^2) +  (right[:,i] - left[:,i])⋅[cos(b.angle), sin(b.angle)]        
    end
    alphadot 
end



function self_induced_velocity(boids::Vector{boid{T}}; ℓ=0.001) where T<:Real
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
    n = size(v2v)[2]÷2
    (v2v[:,1:n] + v2v[:,n+1:end])/2.
end

function move_swimmers!(boids::Vector{boid{T}}; Δt=0.1, ℓ= 5e-4,target = (10,0)) where T<: Real
    """ Find the velocity induced from each swimmer onto the others and 
        update position via a simple Euler's method """
    
    v2v = vortex_to_vortex_velocity(boids;ℓ)
    avg_v = add_lr(v2v)
    siv = self_induced_velocity(boids;ℓ)    
    ind_v = siv + avg_v #eqn 2.a
    angles = angle_projection(boids,v2v)

    
    for (i,b) in enumerate(boids)l
        b.v = ind_v[:,i]
        b.position +=  ind_v[:,i] .* Δt
        b.angle += angles[i]  #eqn 2.b
    end
end

begin 
    n = length(boids)[1]
    lrvel = vortex_to_vortex_velocity(boids)
    left,right = lrvel[:,1:n],lrvel[:,n+1:end]
    #project the velocity difference onto the induced velocity of the swimmer     
    alphadot = zeros(typeof(lrvel[1]), n)
    #define a freestream and background flow bgf
    U = 0.0025
    flowθ = π/4.0 
    bgf = U* [cos(flowθ), sin(flowθ)]
    for (i,b) in enumerate(boids)
        alphadot[i]  = sum(b.gamma)/(2π*ℓ^2) +  ((right[:,i] - left[:,i]).+ bgf)⋅[cos(b.angle), sin(b.angle)]        
    end
    alphadot 

end

###### <---- transition state codes for pompds -----> #########
function change_bearing_speed(sim)
    """
    eqns 5.a,b _call once to set values for the sim
    """
    Γadd = sim.vₐ/sim.v₀ *sim.Γ₀
    ΓT   = sim.ρ/(2*π*sim.ℓ)
    Γadd, ΓT    
end

#ENUMERATE THE ACTIONS  eqns 4
@enum ΔΓ CRUISE FASTER SLOWER LEFT RIGHT

ga,gt = change_bearing_speed(sim)
CIRC_CHANGE = Dict(
    CRUISE => (0.0, 0.0),
    FASTER => (ga, ga),
    SLOWER => (-ga, -ga),
    LEFT   => (gt, -gt),
    RIGHT  => (-gt, gt)
) 
# Possible actions
A = [CRUISE, FASTER, SLOWER, LEFT, RIGHT]
@show CIRC_CHANGE[CRUISE] 

#overload + for changing the circulation of a boid
function Base.:+(b::boid,circ::Tuple) 
    b.gamma[1] += circ[1]
    b.gamma[2] += circ[2]
end

function transition(boids::Vector{boid{T}}, gamma::ΔΓ)
    #define a null state --.
    # ?!
    if any(rewards.(boids) != 0)
        return Deterministic(null)
    end

    num_a = length(A)
    next_states = Vector
    for b in boids

    end
end

function rewards(b::boid, a = missing)
    if norm(b.position - target) < 0.1
        return 10 #get close to the target
    elseif  norm(b.position - target) > 10.0
        return -50
    end
end
###### <---- transition state codes for pompds -----> #########


#We are using 32-bits throughout
type =T= Base.Float32

# Make a grid - strictly for visualization (so far)
xs = LinRange{type}(0,1,31)
ys = LinRange{type}(0,1,31)
# targets = ndgrid(xs,ys)
#do it a different way
X = repeat(reshape(xs, 1, :), length(ys), 1)
Y = repeat(ys, 1, length(xs))
targets = [X,Y]


#Velocity induced from swimmer to swimmer
ind_v = vortex_to_swimmer_midpoint_velocity(boids)
field_v = vortex_to_grid_velocity(boids, targets;ℓ=5e-4)

field_pot = potential(boids, targets;ℓ=5e-4)
stream = streamfunction(boids, targets;ℓ=5e-4)

begin 
    boids = [boid([0.51, 0.4],  [-swim.Γ₀*2.,swim.Γ₀*2.], π/2.0, [0.0,swim.v₀], [0.0,0.0]),
             boid([0.49, 0.6],  [-swim.Γ₀,swim.Γ₀], -π/2.0, [0.0,-sim.v₀], [0.0,0.0]),]
    # for i in -1
    #     # boids = [boid([cos(i), sin(i)],  [-sim.Γ₀,sim.Γ₀],  i, [0.0,0.0], [0.0,0.0])]
    #     push!(boids, boid([0.5, 0.5],  [-sim.Γ₀,sim.Γ₀], π/2.0, [0.0,sim.v₀], [0.0,0.0]))
    #     # push!(vals,vortex_vel(boids;sim.ℓ)[2])
    # end

    # boids = boids|>Vector{boid{Float64}}
    Δt =0.1f0
    n = 100
    anim = @animate for i ∈ 1:n
                move_swimmers!(boids; Δt,sim.ℓ)
                
                # @show sqrt(sum(abs2,boids[1].position .- boids[2].position))
                @show boids[1].position[2],boids[2].position[2]
                f_vels = vortex_to_grid_velocity(boids, targets;sim.ℓ)         
                stream = streamfunction(boids, targets;sim.ℓ)
                plot(collect(xs),collect(ys), stream, st=:contourf)#,clim=(clim,-clim))
                # quiver!(targets[1]|>vec,targets[2]|>vec,
                #        quiver = (f_vels[1,:,:]|>vec,f_vels[2,:,:]|>vec),
                #        aspect_ratio= :equal,
                #        xlim=(xs[1],xs[end]),ylim=(ys[1],ys[end]));
                plot!( xlim=(0,1),ylim=(0,1))
                for b in boids        
                    scatter!([b.position[1]],[b.position[2]],markersize=4,color=:red,label="",markershape=:utriangle)
                end
                plot
    end
    gif(anim, "simple_swimmers.gif", fps = 50)

end
#MAKE a few different plotting routines to verify what we would hope to anticipate
clim =  -sim.Γ₀*(xs[2]-xs[1])*(ys[2]-ys[1]) #color limit for the contour plots
begin 
    #a swimmer going up from origin
    boids = [boid([0.5, 0.5],  [-sim.Γ₀,sim.Γ₀], π/2, [0.0,0.0], [0.0,0.0])]
    field_v = vortex_to_grid_velocity(boids, targets)
    stream = streamfunction(boids, targets)
    clim =  -sim.Γ₀*(xs[2]-xs[1])*(ys[2]-ys[1])
    plot(collect(xs),collect(ys), stream, st=:contourf,clim=(clim,-clim))
    quiver!(targets[1]|>vec,targets[2]|>vec, quiver = (field_v[1,:,:]|>vec,field_v[2,:,:]|>vec),
       xlims=(xs[1],xs[end]),ylims=(ys[1],ys[end]), aspect_ratio= :equal)

end

#test case #validate eqn 2a for a sinlge swimmer
begin
    boids = [boid([0.45, 0.5],  [-1.0,1.0], π/2, [0.0,0.0], [0.0,0.0]),
             boid([0.55, 0.5],  [-1.0,1.0], π/2, [0.0,0.0], [0.0,0.0])]
    ind_v = vortex_to_swimmer_midpoint_velocity(boids)
    self_v = self_induced_velocity(boids)
    v2v = vortex_to_vortex_velocity(boids)
    n = length(boids)[1]
    left,right = v2v[:,1:n],v2v[:,n+1:end]
    # vel = self_v 
    lr =  (left+right)/2.
    # @assert ind_v[1] ≈ vel[1]
    field_v = vortex_to_grid_velocity(boids, targets)
    stream = streamfunction(boids, targets)
    clim =  -sim.Γ₀*(xs[2]-xs[1])*(ys[2]-ys[1])
    a = plot(collect(xs),collect(ys), stream, st=:contourf)
    quiver!(a,targets[1]|>vec,targets[2]|>vec, quiver = (field_v[1,:,:]|>vec,field_v[2,:,:]|>vec),
       xlims=(xs[1],xs[end]),ylims=(ys[1],ys[end]), aspect_ratio= :equal)
    a
    @show ind_v
    @show self_v+lr #this is the perferred decomposition; allows for angle angle_projection
end
a
begin 
    #test scripts for updating the vorticity and angle based on eqns 6
    #set a (x,y) pair as a goal to swim to 
    ℓ = 0.001
    target = (100,100)
    a_desired = [atan(y,x) for (x,y) in [target .- b.position for b in boids]]
    adotdes = (a_desired .- [b.angle for b in boids])/Δt
    Γadd = π*ℓ^2*(adotdes - angles)
    for (i,b) in enumerate(boids)
        b.gamma[1] +=Γadd[i]
        b.gamma[2] -=Γadd[i]
    end
end


begin 
    #a diamond of swimmers going up from origin
    boids = [boid([1.0, 0.0],  [-1.0,1.0], π/2, [0.0,0.0], [0.0,0.0]),
             boid([-1.0, 0.0],  [-1.0,1.0], π/2, [0.0,0.0], [0.0,0.0]),
             boid([0.0, 1.0],  [-1.0,1.0], π/2, [0.0,0.0], [0.0,0.0]),
             boid([0.0, -1.0],  [-1.0,1.0], π/2, [0.0,0.0], [0.0,0.0])]
    field_v = vortex_to_grid_velocity(boids, targets)
    stream = streamfunction(boids, targets)
    plot(collect(xs),collect(ys), stream, st=:contourf)
    quiver!(targets[1]|>vec,targets[2]|>vec, quiver = (field_v[1,:,:]|>vec,field_v[2,:,:]|>vec),
       xlims=(xs[1],xs[end]),ylims=(ys[1],ys[end]),color=:green, aspect_ratio= :equal)
end

begin 
    #swimmers going to the origin
    boids = [boid([1.0, 0.0],  [-1.0,1.0], π/1.0, [0.0,0.0], [0.0,0.0]),
             boid([-1.0, 0.0],  [-1.0,1.0], 0.0, [0.0,0.0], [0.0,0.0]),
             boid([0.0, 1.0],  [-1.0,1.0], -π/2, [0.0,0.0], [0.0,0.0]),
             boid([0.0, -1.0],  [-1.0,1.0], π/2, [0.0,0.0], [0.0,0.0])]
    field_v = vortex_to_grid_velocity(boids, targets)
    stream = streamfunction(boids, targets)
    plot(collect(xs),collect(ys), stream, st=:contourf)
    quiver!(targets[1]|>vec,targets[2]|>vec, quiver = (field_v[1,:,:]|>vec,field_v[2,:,:]|>vec),
       xlims=(xs[1],xs[end]),ylims=(ys[1],ys[end]),color=:green, aspect_ratio= :equal)
end


#The below is busted
begin
    plot(collect(xs),collect(ys), stream, st=:contourf)
    for b in boids
        @show (cos(b.angle),sin(b.angle),b.angle)
        scatter!([b.position[1]],[b.position[2]],markersize=4,color=:red,label="",markershape=:dtriangle)        
    end
    plot!()
end





quiver([1.0],[2.0],quiver=[0.5,0.5],arrow=true,linewidth=0)
plot([0.0,1.0],[0.0,2.0],marker=(:utriangle,10))
begin
plot()
d=0.1
for b in boids
    # plot!([b.position[1]-d*cos(b.angle)],
    #     [b.position[2]-d*sin(b.angle)],
    #     label="",color=:black,seriestype=:scatter)
    plot!([b.position[1],b.position[1]+d*cos(b.angle)],
          [b.position[2],b.position[2]+d*sin(b.angle)],
          arrow = arrow(:closed),label="",color=:blue,linewidth=.1)
end
plot!()
end


#eq 2a is encoded in the function vortex_vel, but angle change is missing
for b in boids
    @show (cos(b.angle)*b.v[1] - sin(b.angle)*b.v[2])

end




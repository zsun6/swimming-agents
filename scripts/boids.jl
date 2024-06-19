#IMPORT other libraries

#OLD BOIDS CODE ))) NO BUENO
using Statistics
using LinearAlgebra
using Plots
# define a data structure for an individual boid
Base.@kwdef mutable struct boid
    position
    velocity
    radius
    length    
    axis
    accell
    function boid(p,v,r,l)
        # constructor of a new boid <- we don't pass in the axis that it is orientated along
        # but calculate it based on velocity and body length
        new(p,v,r,l,l*norm(v),[0.,0.])
    end 
end
# Print function for a boid 
Base.show(io::IO,b::boid) = print(io,"pos  = $(b.position), vel = $(b.velocity)")

# After we update the boids position, set its new axis of motion
function reorient(boid)
    boid.axis = boid.l*norm(boid.velocity)
end

Base.@kwdef mutable struct rules
    # radius for avoidance
    r_avoid
    r_center
    r_copy
    # angles - radians
    a_avoid
    a_center
    a_copy 
    #weightings
    w_avoid
    w_align
    w_center
    w_love
end


function get_neighbors(boids, rule)
    # Use an empirical value from rule to determine which other boids are in the 
    # neighborhood of the current boid
    n = length(boids)
    x = repeat([b.position[1] for b in boids], 1, n)
    y = repeat([b.position[2] for b in boids], 1, n)
    radius = sqrt.((x' .- x).^2 + (y' .- y).^2) #.+ Diagonal([rule.r_avoid for i= 1:n])
    #check: within radius
    #column is the nth boid and row is the neighbor index
    neighbors = radius .< rule.r_avoid
    #check: within angle(cone)
    # TODO :add this in            
end

# Two functions below, but with different input parameters
# this is a very simple case of multiple dispatch
function center(boids, rule )
    """Find the center of mass of other boids in range"""
    n = length(boids)
    close = get_neighbors(boids, rule)
    vec = []
    for i in 1:n
        xs = [boids[i].position[1]]
        ys = [boids[i].position[2]]
        for b in boids[close[i,:].==1]
            push!(xs, b.position[1])
            push!(ys, b.position[2]) 
        end
        push!(vec, [mean(xs), mean(ys)] )
    end
    vec
end
function center_of_mass(boids, rule,close )
    """Find the center of mass of other boids in range"""    
    n = length(boids)
    vec = []
    for i in 1:n
        xs = []
        ys = []
        for b in boids[close[i,:].==1]
            push!(xs, b.position[1])
            push!(ys, b.position[2]) 
        end
        push!(vec, [mean(xs), mean(ys)] )
    end
    vec
end

function to_center(boids, com)
    vec = []
    for (index,center) in enumerate(com)
        push!(vec,center .- [boids[index].position[1],boids[index].position[2]])
    end
    vec
end
function vel_mean(boids, rule, close)
    """Find the center of mass of other boids in range"""    
    n = length(boids)
    vec = []
    for i in 1:n
        u = []
        v = []
        for b in boids[close[i,:].==1]
            push!(u, b.velocity[1])
            push!(v, b.velocity[2]) 
        end
        push!(vec, [mean(u), mean(u)] )
    end
    vec
end

function align(boids, rule, close)
        # neighbors = self.get_neighbors(boids, radius, angle)
        vel =  vel_mean(boids, rule, close)
        to_center(boids, vel)
end

function avoid(boids, carrot, rule)    
    all = [boids; carrot]
    close = get_neighbors(all,rule)
    com = center_of_mass(all, rule, close)
    .- to_center(all, com)
end

function move(boids, goals, μ = 0.1, dt = 0.1)
    for (i,b) in enumerate(boids)
        b.velocity = b.velocity * (1 - μ) + goals[i] * μ
        b.position += b.velocity * dt
        b.axis = b.length * b.velocity
    end
end

function love(boids, carrots)
    m = length(carrots)
    n = length(boids)
    cx = repeat([c.position[1] for c in carrots], 1, n)
    cy = repeat([c.position[2] for c in carrots], 1, n)
    bx = repeat([b.position[1] for b in boids], 1, m)
    by = repeat([b.position[2] for b in boids], 1, m)
    rad = (cx.-bx').^2 .+ (cy.-by').^2
    nearest_lovers = findmin(rad,dims=1)[2]
    toward = []
    for (x,y) in zip((cx.-bx')[nearest_lovers], (cy.-by')[nearest_lovers])
        push!(toward,[x,y])
    end
    toward
end

function set_goal(avoid_v, center_v,align_v,love_v, rule)
    goal = rule.w_avoid*avoid_v + rule.w_center*center_v + 
           rule.w_align*align_v + rule.w_love * love_v 
end
opposite_dir(vec) = .- vec

# Define a set of rules - named after author work is taken from
# downey = rules(0.3, 1.0, 0.5, 2π, 2, 2, 4, 3, 2, 10) 
downey = rules(sqrt(2.1), 1.0, 0.5, 2π, 2, 2, 4, 3, 2, 10) 
# number of boids
# n_boids = 32
# boids == flock = Array of boid structs
# boids = [boid(rand(2).*2, rand(2).-0.5, 0.03, 0.1) for i in 1:n_boids]
boids = [boid([0,0], [1,1], 0.03, 0.1),
         boid([0,1], [1,-1], 0.03, 0.1),
         boid([1,1], [-1,-1], 0.03, 0.1),
         boid([1,0], [-1,1], 0.03, 0.1)]
n_boids = length(boids)
# which boids are close by?
close = get_neighbors(boids, downey)
# find the center of mass
com = center_of_mass(boids,downey,close)
boid_to_com = to_center(boids, com)
com_to_boid = opposite_dir(boid_to_com)
#obstacle to avoid or go towards
# carrots = [boid([1.25,1.25],   [0,0], 1., 1.),
#            boid([0.5,0.5], [0,0], 1., 1.), 
#            boid([0.9,0.1], [0,0], 1., 1.)]
carrots = [boid([0.5,0.5],   [0,0], 1., 1.)]

aligning = align(boids, downey, close)
toward = love(boids,carrots)
away = avoid(boids,carrots,downey)[1:n_boids,:]

accel = set_goal(away, boid_to_com, aligning, toward, downey)

move(boids,accel)

quiver([b.position[1] for b in boids],[b.position[2] for b in boids],
    quiver=([b.velocity[1] for b in boids],[b.velocity[2] for b in boids]))
quiver!([b.position[1] for b in boids],[b.position[2] for b in boids],
    quiver=aligning')
plot!([c.position[1] for c in carrots],
    [c.position[2] for c in carrots], seriestype=:scatter)
quiver!([b.position[1] for b in boids],[b.position[2] for b in boids],
    quiver=avoid(boids,carrots,downey)')
quiver!([b.position[1] for b in boids],[b.position[2] for b in boids],
    quiver=toward)



@gif for i ∈1:100
    plot([b.position[1] for b in boids],[b.position[2] for b in boids],seriestype=:scatter)
    toward = love(boids,carrots)
    away = avoid(boids,carrots,downey)[1:n_boids,:]
    aligning = align(boids, downey, close)
    accel = set_goal(away, boid_to_com, aligning, toward, downey)
    move(boids,accel)
end

plot!([b.position[1] for b in boids],[b.position[2] for b in boids],
      seriestype=:scatter)
toward = love(boids,carrots)
away = avoid(boids,carrots,downey)[1:n_boids,:]
aligning = align(boids, downey, close)
accel = set_goal(away, boid_to_com, aligning, toward, downey)
move(boids,accel)

include("..//src//FiniteDipole.jl")
begin
    # How is the self-induced vel for turning boids? validate against Barba code--- its good
    ℓ = 1.0
    swim = Swimmer_params(ℓ)
    boids = [FD_agent([0.0, 0.0],  [1.0,-1.0], π/2.0, [0.0,swim.v₀], [0.0,0.0])]
    v2v = vortex_to_vortex_velocity(boids;ℓ)
    siv = self_induced_velocity(boids;ℓ)
    # avg_v = add_lr(v2v)
    avg_v = sum(v2v,dims=2)
    ind_v = siv + avg_v #eqn 2.a
    angles = angle_projection(boids,v2v)
    raw_vel = vortex_to_swimmer_midpoint_velocity(boids;ℓ)
    # @show siv, avg_v, raw_vel
    @assert ind_v ≈ raw_vel
    @assert avg_v ≈ [0.0, -0.3183098861837907] #pulled from Barba AeroPython
end
begin
    # How is the self-induced vel for turning boids? 
    ℓ = 1.0
    boids = [FD_agent([0.0, 0.0],  [1.0,-1.0], 3π/4.0, [0.0,swim.v₀], [0.0,0.0])]
    v2v = vortex_to_vortex_velocity(boids;ℓ)
    siv = self_induced_velocity(boids;ℓ)
    avg_v = add_lr(v2v)
    ind_v = siv + avg_v #eqn 2.a
    ind_v = siv + sum(v2v,dims=2)

    angles = angle_projection(boids,v2v;ℓ)
    @assert ind_v  ≈ vortex_to_swimmer_midpoint_velocity(boids;ℓ)
    @assert sum(v2v,dims=2) ≈ [0.2250790790392765, -0.22507907903927654] #pulled from Barba AeroPython
end

begin 
    #a swimmer going up from origin
    xs = LinRange(0,1,31)
    ys = LinRange(0,1,31)

    X = repeat(reshape(xs, 1, :), length(ys), 1)
    Y = repeat(ys, 1, length(xs))
    targets = [X,Y]
    boids = [FD_agent([0.0, 1.0],  [1.0,-1.0], π/2, [0.0,0.0], [0.0,0.0]),
    FD_agent([2.0, 3.0],  [1.0,-1.0], π/2, [0.0,0.0], [0.0,0.0]),
    FD_agent([4.5, 5.5],  [1.0,-1.0], π/2, [0.0,0.0], [0.0,0.0])]
    field_v = vortex_to_grid_velocity(boids, targets;ℓ=0.1)
    stream = streamfunction(boids, targets;ℓ=0.1)
    clim =  -swim.Γ₀*(xs[2]-xs[1])*(ys[2]-ys[1])
    plot(collect(xs),collect(ys), stream, st=:contourf,clim=(clim,-clim))
    quiver!(targets[1]|>vec,targets[2]|>vec, quiver = (field_v[1,:,:]|>vec,field_v[2,:,:]|>vec),
        aspect_ratio= :equal, xlims=(0,1),ylims=(0,1))

end

#TODO: add suite to tests against PotentialFlow.jl 
#    : probably should transition to using that library to represent any ROMS
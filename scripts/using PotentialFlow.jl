using PotentialFlow
t = 0.0
v1 = Vortex.Point(-1.0 + 0im, π+π/10)
v2 = Vortex.Point(1.0 + 0im, -π+π/10)

induce_velocity(1.0+0im, v1,0.0)
induce_velocity(-1.0+0im, v2,0.0)
lr = induce_velocity([-1.0 + 0im,1.0 + 0im],(v1,v2),0.0)
induce_velocity(0.0+0im, (v1,v2), t)
induce_velocity(1.0+0im, v1, t)

x = range(-5, 5, length=100)
y = range(-5, 5, length=100)
# streamfunction((x,y), (v1,v2),0.0)
streamlines(x, y, (v1,v2),color=:thermal,st=:contourf)
# plot!([0;0],[0; 1], color=:red)
plot!([0;cos(π/10+π/2)],[0; sin(π/10+π/2)],color=:green)
streamlines(x,y,(Doublet(0.0+0.0im,π)),color=:reds)

boids = [FD_agent([0.0, 0.0],  [π+π/10,-π+π/10], π/2.0, [0.0,swim.v₀], [0.0,0.0])]
X = repeat(reshape(x, 1, :), length(y), 1)
Y = repeat(y, 1, length(x))
targets = [X,Y]
stream = streamfunction(boids, targets;ℓ)
field_v = vortex_to_grid_velocity(boids, targets;ℓ)
plot(x,y, stream, st=:contourf,color=:thermal)
quiver!(targets[1]|>vec,targets[2]|>vec,
        quiver=(field_v[1,:,:]|>vec,field_v[2,:,:]|>vec)./10,
        aspect_ratio= :equal, xlims=(-10,10),ylims=(-10,10),color=:red)
siv = self_induced_velocity(boids;ℓ)
v2v = vortex_to_vortex_velocity(boids;ℓ)
b=boids[1]
plot()

using LinearAlgebra
using Plots


########################################################################
"""
Reduced order model swimmer
requirements: 
1. vortex based
2. encodes transient traveling wave motion
3. includes a pressure solver -> finds efficiency, thrust, power, lift, etc …
    a. Neural Net pressure solver, the remainder of the system is linear per time step and
       this can be avoided and sped up sig if done right
4. easy to setup discrete motions -> faster, slower, left, right
5. invertable? or just attach a FMM code for acceleration
6. Write a version of Tianjuan's BEM to scrape inviscid data from 
7. POD/PCA or DMD on the vortex distro to define a rom
8. ROM -> P(NN) -> Hydrodynamic performance metrics  
9. Emit a vortex at reg interval

P(Γ((x,y)(t))) we want to reduce Γ while still getting pressure along the body

"""
#rows left right
vort = [-1 1 
         0.5  -0.5]
# cols i, rows x y                  
pos  = [0.0 0.0
        0.5 0.0]  

δ = 0.25
θ = π/2.0
θr = θ - π/2.0
θl = θ + π/2.0  
left = [(pos[1,:] .+ δ*cos(θl))'
        (pos[2,:] .+ δ*sin(θl))']
right = [(pos[1,:] .+ δ*cos(θr))'
         (pos[2,:] .+ δ*sin(θr))']        

sources = [left right]

function streamfunction(sources,gammas,targets)
    """
    """
    pot = zeros((size(targets[1])...))

    for i in 1:size(sources)[2]

        dx = targets[1] .- sources[1,i]
        dy = targets[2] .- sources[2,i]
        @. pot += gammas[i] *log(sqrt(dx^2+dy^2))            
    end
    pot./(2π)
end


xs = LinRange(-2,2,31)
ys = LinRange(-2,2,31)

X = repeat(reshape(xs, 1, :), length(ys), 1)
Y = repeat(ys, 1, length(xs))
targets = [X,Y]

stream = streamfunction(sources,vort, targets)



plot(collect(xs),collect(ys), stream, st=:contourf,c=:redsblues)
plot!(pos[1,:],pos[2,:],seriestype=:scatter, label="top")
plot!(left[1,:],left[2,:],seriestype=:scatter,label="left")
plot!(right[1,:],right[2,:],seriestype=:scatter,label="right")   

#TAVELING WAVES
a0 = 0.1
a = [0.367,0.323,0.310]
f = k = 1
h(x,t) = a0*amp(x,a)*sin(2π*(k*x - f*t))
amp(x,a) = a[1] + a[2]*x + a[3]*x^2


ang = x -> h(x,a)
T = 0.12
an = [0.2969, -0.126, -0.3516, 0.2843, -0.1036]
@. yt(x) = T/0.2 *(an[1]*x^0.5 + an[2]*x + an[3]*x^2 + an[4]*x^3 + an[5]*x^4 )
plot(x, x->h.(x,0.0))

th = h.(x,0)
plot(x, h.(x,0),aspect_ration=:equal)
plot!(x[end:-1:1], (yt(x) +th)[end:-1:1])
plot!(x, -yt(x) + th,aspect_ratio= :equal)
foil = [ x[end:-1:1]   yt(x[end:-1:1]) ;
         x     -yt(x)]
plot!(foil[:,1],foil[:,2],aspect_ratio=:equal)

plot!(foil[21:32,1],foil[21:32,2],marker=:star)
th = h.(x,0)
foil_y(x,δ) = [ yt(x[end:-1:1]) + δ[end:-1:1];  -yt(x[2:end])+δ[2:end]]
plot([x[end:-1:1]; x[2:end]], foil_y(x,th),aspect_ratio=:equal)
begin 
    n = 25
    T = LinRange(0, 1, n)
    # x = LinRange(0,1,25)
    anim = @animate for t in T
  
        plot([x[end:-1:1]; x[2:end]], foil_y(x,h.(x,t)), aspect_ratio=:equal)
        # plot(x,h.(x,t), xlim=(-0.1,1.1), ylim=(-2*a0,2*a0),aspect_ratio=:equal)
    end
    gif(anim,"ang.gif", fps=12)
end

begin
    plot()
    for t in LinRange(0, 1, 10)
        plot!(x,h.(x,t), 
            xlim=(-0.1,1.1), ylim=(-2*a0,2*a0),
            aspect_ratio=:equal, label="")
    end
    plot!()
end



begin 
    n = 25
    T = LinRange(0, 1, n)
    
    anim = @animate for t in T
        vorts = vort .* (sin(2π*t))/2.
        @show vort
        stream = streamfunction(sources,vorts,targets)
        plot(collect(xs),collect(ys), stream, st=:contourf,c=:redsblues)        
        plot!(left[1,:],left[2,:],seriestype=:scatter,label="")
        plot!(right[1,:],right[2,:],seriestype=:scatter,label="")  
    end
    gif(anim,"simple.gif", fps=12)
end


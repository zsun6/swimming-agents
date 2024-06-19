########## SHIT SHOW -> first pass, read in time varying data of body circulation to reduce it down

# using CSV
# using DataFrames
using DataDrivenDiffEq 
using DelimitedFiles
using Plots
using FFTW

file = "C:\\Users\\nate\\Desktop\\biofluids-coupled-bems\\offset_y0.5_x0.5\\gammas0.csv"
data = readdlm(file, ',', Float64)

size(data) #451, 132 -> Δt, 128 Body + 4 edge
begin 
    n = 451
    # T = LinRange(0, 1, n)
    x = LinRange(0,1,132)
    anim = @animate for t in 1:n
        plot(x,data[t,:], aspect_ratio=:equal,xlims=(-0.1,1.1),ylims=(-0.5,0.5))
        title!( "$(t%150)")
    end
    gif(anim,"circulation.gif", fps=12)
end

@show size(data[:,1:128])
sum(data[:,1:128],dims=2)
sum(data[:,129:end],dims=2)
sum(data[:,:],dims=2)

#trim up that data
clean  = data[4*150+2:end,1:128]
edge  = data[4*150+2:end,129:end]
plot(
plot(clean,st=:contourf,color=:turbo,xlabel="Body Panel"),
plot(edge,st=:contourf,color=:turbo,xlabel="Edge Panel" ),
layout = grid(1,2,widths =[ 0.7,0.3]), size=(1000,800),
sharey=:true
)
sum(clean,dims=2)

trans = fft(clean,2)
plot(real(trans)[:,2:50], st=:contourf,color =:turbo)
plot(imag(trans)[:,2:50], st=:contourf,color =:turbo)

plot(real(ifft(clean[:,1:10])),st=:contourf,color=:turbo)
ifft(clean[:,1:10])

heatmap(clean[1:2:end,2:end])
heatmap(edge[1:10:300,2:end])
begin 
    n = 451
    # T = LinRange(0, 1, n)
    xt = LinRange(0,1,63)
    xb = LinRange(1,0,64)
    anim = @animate for t in 1:n
        plot(xt,clean[t,2:64],color=:red,
             xlims=(-0.1,1.1),ylims=(-0.05,0.1), label="")
        plot!(xb,clean[t,65:128],color=:blue,
        xlims=(-0.1,1.1),ylims=(-0.05,0.1), label="")
        title!( "$(t%150)")
    end
    gif(anim,"circulation.gif", fps=30)
end
begin
    plot()
    for i in 1:10:600
        plot!(clean[i,2:end],label="")
    end
    plot!()
end
begin
    plot()
    for i in 1:10:600
        plot!(edge[i,:],label="")
    end
    plot!()
end
controller = ? 
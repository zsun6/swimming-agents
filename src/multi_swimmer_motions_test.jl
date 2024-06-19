using Plots
include(".\\SwimmerEnv.jl")

n_swimmers = 16
env = SwimmingEnv(;n_swimmers = n_swimmers, max_steps = 200, target=[1,1])
env.swimmers
begin
    """
    look at turning radius
    """
    T = Float32
    # act = env |> ex.policy
    plt = plot(aspect_ratio=:equal)
    reset!(env)
    
    for (i,sw) in enumerate(env.swimmers)
        # sw.position[1] += env.params.D*i*2
        plot!(plt, [sw.position[1]],[sw.position[2]],st=:scatter, marker=:star,label="")
    end
    xs = []
    ys = []
    for sw in env.swimmers
        # sw = env.swimmers[1]
            push!(xs, sw.position[1])
            push!(ys, sw.position[2])
            # plot!(plt, [sw.position[1]],[sw.position[2]],st=:scatter, marker=:star,label="")
    end
    for t = 1:250
        (env)((ones(Int,n_swimmers).*4)|>Tuple)
        # (env)(rand(1:5, n_swimmers)|>Tuple)
        # @show env.swimmer
        for sw in env.swimmers
        # sw = env.swimmers[1]
            push!(xs, sw.position[1])
            push!(ys, sw.position[2])
            # plot!(plt, [sw.position[1]],[sw.position[2]],st=:scatter, marker=:star,label="")
        end
    end
    plot!(aspect_ratio=:equal)
    for i = 1:n_swimmers
        plot!(xs[i:n_swimmers:end] ,ys[i:n_swimmers:end],lw=0.5, label="")
    end
    plt
end
xs
using Flux, Plots, Distributions, Random, LinearAlgebra
x = hcat(collect(Float32, -3:0.1:3)...)
f(x) = @. 3x + 2;
y = f(x)

x = x .* reshape(rand(Float32, 61), (1, 61));
f1 = plot(vec(x), vec(y), lw = 3, seriestype = :scatter, label = "", title = "Generated data", xlabel = "x", ylabel= "y")
# model = Dense(1 => 1)
model = Chain(Dense(1=>10, tanh), Dense(10=>10, relu), Dense(10=>1)) 
plot!(vec(x), model(x)[:])
# params, reconstruct = Flux.destructure(model)
# model(x)
# params .= randn(2)
# model = reconstruct(params)
loss(m, x, y) = Flux.mse(m(x), y)




normsq(x) = dot(x,x)


function eventzz(tmax, λs, λref)
    τ0 = tmax
    k0 = 0
    ev = 0
    for k in eachindex(λs)
        τ1 = λs[k] <= 0 ? Inf : -log(rand())/λs[k]
        τ2 = λref <= 0 ? Inf : -log(rand())/λref
        if τ0 < min(τ1,τ2)
            continue
        elseif τ1 < τ2
             τ0, k0, ev = τ1, k, 1
        else
            τ0, k0, ev = τ2, k, 2
        end
    end
    return τ0, k0, ev
end


function zz_flux((x, y, model, loss), λref, Niter, h, thin)
    xx = [loss(model, x, y),]
    nobs = size(x, 2)
    θ0, reconstruct = Flux.destructure(model)
    p = length(θ0)
    v = rand([-1.0, +1.0], p) 
    i = 1
    dt = h
    flips = 0
    t = 0.0
    while(i < Niter)
        # i = 1:61
        j = rand(1:nobs)
        input, output = x[:, j], y[:,j]
        grads = Flux.gradient(model) do m 
                loss(m, input, output)
            end
        ∇Uθ, _ = Flux.destructure(grads)
        θ, _ = Flux.destructure(model)

        τ0, k0, ev  = eventzz(dt, ∇Uθ.*v, λref)
        t += τ0
        θ .+= v*τ0
        if ev != 0 # either reflection or refreshment, can also check with if k0 = 0
            dt = dt - τ0
            v[k0] *= -1
            flips += 1 
        else
            i += 1
            dt=h
            if i % thin == 0
                push!(xx, loss(model, x, y))
            end   
        end 
        # @. params = params - 0.001 * ∇Uparams
        model = reconstruct(θ)
        # plot!(vec(x), model(x)[:], alpha = 0.1)
    end
    model, xx
end

model1, trace =zz_flux((x, y, model, loss),  0.0, 100_000, 0.00005, 1000)
scatter!(f1, vec(x), model1(x)[:])
f2 = plot(trace)
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

function bounce!(∇Ux, v)
    v .-= (2*dot(∇Ux, v)/normsq(∇Ux))*∇Ux
    v
end


function eventbps(tmax, λ, λref)
    τ1 = λ <= 0 ? Inf : -log(rand())/λ
    τ2 = -log(rand())/λref
    if tmax < min(τ1, τ2)
      return 0, tmax
    elseif τ1 < τ2
      return 1, τ1
    else
      return 2, τ2
    end
end


function bps_flux((x, y, model, loss), λref, Niter, h, thin)
    xx = [loss(model, x, y),]
    nobs = size(x, 2)
    θ0, reconstruct = Flux.destructure(model)
    p = length(θ0)
    v = randn(p)
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

        ev, τ  = eventbps(dt, dot(∇Uθ, v), λref)
        t += τ
        θ .+= v*τ
        if ev == 1
            dt = dt - τ
            # right now I am not able to reuse the same jth observation 
            # ∇Ux = ∇U!(∇Ux, x, j, args...)
            v = bounce!(∇Uθ, v)
            flips += 1
        elseif ev == 2
            dt = dt - τ
            v = randn(p)
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

model1, trace = bps_flux((x, y, model, loss),  1.0, 100_000, 0.00005, 1000)
scatter!(f1, vec(x), model1(x)[:])
f2 = plot(trace)
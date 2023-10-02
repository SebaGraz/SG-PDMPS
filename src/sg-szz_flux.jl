using Flux, Plots, Distributions, Random, LinearAlgebra
# x = hcat(collect(Float32, -3:0.1:3)...)
# f(x) = @. 3x + 2;
# y = f(x)

# x = x .* reshape(rand(Float32, 61), (1, 61));
# f1 = plot(vec(x), vec(y), lw = 3, seriestype = :scatter, label = "", title = "Generated data", xlabel = "x", ylabel= "y")
# # model = Dense(1 => 1)
# model = Chain(Dense(1=>10, tanh), Dense(10=>10, relu), Dense(10=>1)) 
# plot!(vec(x), model(x)[:])
# # params, reconstruct = Flux.destructure(model)
# # model(x)
# # params .= randn(2)
# # model = reconstruct(params)
# loss(m, x, y) = Flux.mse(m(x), y)




# normsq(x) = dot(x,x)

function hitting_time(x,v)
    if x*v>= 0.0 
        return Inf
    else
        return -x/v
    end  
end

function eventszz(x, v, vc, s, tmax, λs, λref, κs)
    τ0 = tmax
    k0 = 0
    ev = 0
    for k in eachindex(λs)
        if s[k] == 0
            τ1 = λs[k] <= 0 ? Inf : -log(rand())/λs[k]
            τ2 = λref <= 0 ? Inf : -log(rand())/λref
            τ3 = hitting_time(x[k],v[k])
        else
            τ1 = Inf
            τ2 = λref <= 0 ? Inf : -log(rand())/λref
            τ3 = -log(rand())/(κs[k].*abs(vc[k]))
        end
        if τ0 < min(τ1,τ2, τ3)
            continue
        elseif τ1 < min(τ2, τ3)
             τ0, k0, ev = τ1, k, 1
        elseif τ2 < τ3
            τ0, k0, ev = τ2, k, 2
        else
            τ0, k0, ev = τ3, k, 3
        end
    end
    return τ0, k0, ev
end

function sgszz_flux((x, y, model, loss), kappa, λref, Niter, h, thin)
    xx = [loss(model, x, y),]
    model0 = deepcopy(model)
    fullgrads0 = Flux.gradient(model0) do m 
        loss(m, x, y)
    end
    ∇Uθ0_full, _ = Flux.destructure(fullgrads0)
    nobs = size(x, 2)
    θ0, reconstruct = Flux.destructure(model)
    p = length(θ0)
    κs = ones(p)*kappa
    v = rand([-1, +1], p) 
    vc = copy(v)
    s = zeros(Bool, p)
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
        grads0 = Flux.gradient(model0) do m 
            loss(m, input, output)
        end    
        ∇Uθ, _ = Flux.destructure(grads)
        ∇Uθ0, _ = Flux.destructure(grads0)
        θ, _ = Flux.destructure(model)
        ∇Uθcv = (∇Uθ - ∇Uθ0)*nobs + ∇Uθ0_full
        τ0, k0, ev  = eventszz(θ, v, vc, s, dt, ∇Uθcv.*v, λref, κs)
        t += τ0
        θ .+= v*τ0
        if ev == 1 || ev == 2 # either reflection or refreshment, can also check with if k0 = 0
            dt = dt - τ0
            v[k0] *= -1
            vc[k0] *= -1 
            flips += 1 
        elseif ev == 3
            dt = dt - τ0
            if s[k0] == 1
                θ[k0] = 0.0
                v[k0] = vc[k0]
                s[k0] = 0
            else
                θ[k0] = 0.0
                v[k0] = 0.0
                s[k0] = 1
            end 
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

# model1, trace =sgszz_flux((x, y, model, loss),  1.0, 0.0, 100_000, 0.00005, 1000)

# scatter!(f1, vec(x), model1(x)[:])
# f2 = plot(trace)
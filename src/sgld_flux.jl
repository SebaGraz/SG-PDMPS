using Flux, Plots, Distributions, Random
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


function sgld_flux((x, y, model, loss), Niter, h, thin, (x_test, y_test))
    xx = [loss(model, x_test, y_test),]
    model0 = deepcopy(model)
    fullgrads0 = Flux.gradient(model0) do m 
        loss(m, x, y)
    end
    ∇Uθ0_full, _ = Flux.destructure(fullgrads0)
    nobs = size(x, 2)
    for j in 1:Niter
        # i = 1:61
        i = rand(1:nobs)
        input, output = x[:, i], y[:,i]
        grads = Flux.gradient(model) do m 
                loss(m, input, output)
            end
        grads0 = Flux.gradient(model0) do m 
                loss(m, input, output)
            end    
        ∇Uθ, _ = Flux.destructure(grads)
        ∇Uθ0, _ = Flux.destructure(grads0)
        θ, reconstruct = Flux.destructure(model)
        
        ∇Uθcv = (∇Uθ - ∇Uθ0)*nobs + ∇Uθ0_full
        θ .= θ .- h/2 * ∇Uθcv .+ sqrt(h)*randn(length(θ)) 

        # @. params = params - 0.001 * ∇Uparams
        model = reconstruct(θ)
        # plot!(vec(x), model(x)[:], alpha = 0.1)
        if j % thin == 0 
            push!(xx, loss(model, x_test, y_test))
        end
    end
    model, xx
end

# model1, trace = sgld_flux((x, y, model, loss), 30_000, 0.000001, 100)
# scatter!(f1, vec(x), model1(x)[:])
# f2 = plot(trace)
# scatter!(f1, vec(x), model(x)[:])

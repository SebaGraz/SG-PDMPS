using Flux, Plots, Distributions, Random
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

function sgld_flux((x, y, model, loss), Niter, h, thin)
    xx = [loss(model, x, y),]
    nobs = size(x, 2)
    for j in 1:Niter
        # i = 1:61
        i = rand(1:nobs)
        input, output = x[:, i], y[:,i]
        grads = Flux.gradient(model) do m 
                loss(m, input, output)
            end
        ∇Uparams, _ = Flux.destructure(grads)
        params, reconstruct = Flux.destructure(model)
        params .= params .- h/2 * ∇Uparams*nobs .+ sqrt(h)*randn(length(params)) 
        # @. params = params - 0.001 * ∇Uparams
        model = reconstruct(params)
        # plot!(vec(x), model(x)[:], alpha = 0.1)
        if j % thin == 0 
            push!(xx, loss(model, x, y))
        end
    end
    model, xx
end

model1, trace = sgld_flux((x, y, model, loss), 10_000, 0.0001, 100)
scatter!(f1, vec(x), model1(x)[:])
f2 = plot(trace)
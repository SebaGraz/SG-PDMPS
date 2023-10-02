using Flux, Plots
function runall()
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

    println("loss at the beginning $(loss(model, x, y))")
    function run_sampler(x, y, model, loss, N)
        xx = []
        for j in 1:N
            # i = 1:61
            i = rand(1:61)
            input, output = x[:, i], y[:,i]
            grads = Flux.gradient(model) do m 
                    loss(m, input, output)
                end
            ∇Uparams, _ = Flux.destructure(grads)
            params, reconstruct = Flux.destructure(model)
            @. params = params - 0.00001 * ∇Uparams*61 
            # @. params = params - 0.001 * ∇Uparams
            model = reconstruct(params)
            # plot!(vec(x), model(x)[:], alpha = 0.1)
            if j %100 == 0 
                push!(xx, loss(model, x, y))
            end
        end
        model, xx
    end

    model, ll = run_sampler(x, y, model, loss, 10000)
    println("loss at the  end $(loss(model, x, y))")
    scatter!(f1, vec(x), model(x)[:])
    f2 = plot(ll)
    f1, f2
end


f1, f2 = runall()
f1
f2
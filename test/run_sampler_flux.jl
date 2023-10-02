include("./../src/sgld_flux.jl")
include("./../src/sg-bps_flux.jl")
include("./../src/sg-zz_flux.jl")
include("./../src/sg-szz_flux.jl")
include("./test_adam_flux.jl")
using Plots
# Number of points to generate.
N = 80
M = round(Int, N / 4)
Random.seed!(1234)

# Generate artificial data.
x1s = rand(M) * 4.5;
x2s = rand(M) * 4.5;
xt1s = Array([[x1s[i] + 0.5; x2s[i] + 0.5] for i in 1:M])
x1s = rand(M) * 4.5;
x2s = rand(M) * 4.5;
append!(xt1s, Array([[x1s[i] - 5; x2s[i] - 5] for i in 1:M]))

x1s = rand(M) * 4.5;
x2s = rand(M) * 4.5;
xt0s = Array([[x1s[i] + 0.5; x2s[i] - 5] for i in 1:M])
x1s = rand(M) * 4.5;
x2s = rand(M) * 4.5;
append!(xt0s, Array([[x1s[i] - 5; x2s[i] + 0.5] for i in 1:M]))

# Store all the data for later.
xs = [xt1s; xt0s]
ts = [ones(2 * M); zeros(2 * M)]

# Plot data points.
function plot_data()
    x1 = map(e -> e[1], xt1s)
    y1 = map(e -> e[2], xt1s)
    x2 = map(e -> e[1], xt0s)
    y2 = map(e -> e[2], xt0s)

    Plots.scatter(x1, y1; color="red", clim=(0, 1))
    return Plots.scatter!(x2, y2; color="blue", clim=(0, 1))
end

# plot_data()

x = zeros(Float32, 2, 80)
[x[:,i] = xs[i] for i in eachindex(xs)]
x
ts
y = zeros(Float32, 1, 80)
[y[1,i] = ts[i] for i in eachindex(ts)]
model = Chain(Dense(2, 3, tanh), Dense(3, 2, tanh), Dense(2, 1, σ))
λ = 0.1
function loss(m, x, y)
    params, _ = Flux.destructure(m) 
    0.5*Flux.mse(m(x), y) + sum(abs2, params)*λ*0.5
end
param, _ = Flux.destructure(model)
param[1:10]

loss(model, x, y)
mb = 1
model1 = AdamCV((x, y, model, loss, mb), 100_000;eps = 1e-8, α = 0.0001, β1 = 0.9, β2 = 0.999)
loss(model1, x, y)


model2, trace  = sgld_flux((x, y, model1, loss), 100_000, 0.000001, 100)
f2 = plot(trace)

model3, trace  = sgbps_flux((x, y, model1, loss), 1.0, 100_000, 0.00001, 100)
f2 = plot!(trace)

model4, trace  = sgzz_flux((x, y, model1, loss), 0.0, 100_000, 0.00001, 100)
f2 = plot!(trace)

model5, trace  = sgszz_flux((x, y, model1, loss), 1.0, 1.0, 100_000, 0.00001, 100)
f2 = plot!(trace)












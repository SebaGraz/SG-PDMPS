using StatsBase, LinearAlgebra
str_app = "toy"
str_data = "./scripts/"*str_app*"/data/"
str_d = "d_"
str_csv = ".csv"
str_trace = "./scripts/"*str_app*"/mse/"
str_h = "h_"
str_csv = ".csv"
str_sgld1 = "sgld1_" 
str_zz =  "zz_"
str_bps =  "bps_"
str_iter = "_iter_"
d = 5
DIRINDATA = str_data*str_d*string(d)*str_csv 
data  = Matrix(CSV.read(DIRINDATA, DataFrame, header=false))
A, y = Matrix(data[:,1:(end-1)]), Vector(data[:,end])
nobs = size(A, 1)
σ = sqrt.(diag(inv((A'A + I*1/10))*nobs/100))

hh = [0.1, collect(0.05:-0.01:0.02)..., collect(0.01:-0.002:0.001)...,collect(0.001:-0.0002:0.0001)..., collect(0.0001:-0.00002:0.00002)..., 0.00001]# hh = [0.1,]
niter = 100
h = 0.0001
iter = 1
test_function(x, t, σ) = mean(((std(x[:,1:t], dims = 2) - σ)./σ).^2) 

res1 = zeros(length(hh), niter)
str_sampler = "sgld1_"
for (i, h) in zip(eachindex(hh), hh)
    for iter in 1:niter
        DIRINTRACE = str_trace*str_sampler*str_h*string(h)*str_iter*string(iter)*str_csv
        trace = Matrix(CSV.read(DIRINTRACE, DataFrame, header=false))
        res1[i, iter] = test_function(trace, size(trace,2), σ)
    end
end

v1 = var(res1, dims = 2)
b1 = mean(res1.^2, dims = 2)


res2 = zeros(length(hh), niter)
str_sampler = "zz_"
for (i, h) in zip(eachindex(hh), hh)
    for iter in 1:niter
        DIRINTRACE = str_trace*str_sampler*str_h*string(h)*str_iter*string(iter)*str_csv
        trace = Matrix(CSV.read(DIRINTRACE, DataFrame, header=false))
        res2[i, iter] = test_function(trace, size(trace,2), σ)
    end
end

v2 = var(res2, dims = 2)
b2 = mean(res2.^2, dims = 2)

res3 = zeros(length(hh), niter)
str_sampler = "bps_"
for (i, h) in zip(eachindex(hh), hh)
    for iter in 1:niter
        DIRINTRACE = str_trace*str_sampler*str_h*string(h)*str_iter*string(iter)*str_csv
        trace = Matrix(CSV.read(DIRINTRACE, DataFrame, header=false))
        res3[i, iter] = test_function(trace, size(trace,2), σ)
    end
end

v3 = var(res3, dims = 2)
b3 = mean(res3.^2, dims = 2)

mse1 = v1 + b1
# mse1[8] = NaN 
f1 = plot(hh, mse1, xaxis = :log, yaxis = :log, title = "MSE", label = "sgld")

mse2 = v2 + b2 
plot!(f1, hh, mse2, xaxis = :log, yaxis = :log, label = "zz")

mse3 = v3 + b3 
plot!(f1, hh, mse3, xaxis = :log, yaxis = :log, label = "bps")
i3 = findmin(mse3[:])[2]
i2 = findmin(mse2[:])[2]
i1 = findmin(mse1[:][9:end])[2] + 8

vline!(f1, [hh[i1], hhx[i2], hh[i3]], color = :black, linestyle=:dash, label = "")
println("best for sgld: $(hh[i1])")
println("best for bps: $(hh[i2])")
println("best for zz: $(hh[i3])")
savefig("./scripts/"*str_app*"/output/mse.pdf")




plot(hh, v1, xaxis = :log, yaxis = :log, title = "Variance", label = "sgld")
plot!(hh, v2, xaxis = :log, yaxis = :log, label = "bps")
plot!(hh, v3, xaxis = :log, yaxis = :log, label = "zz")
savefig("./scripts/"*str_app*"/output/var.pdf")

plot(hh, b1, xaxis = :log, yaxis = :log, title = "Bias", label = "sgld")
plot!(hh, b2, xaxis = :log, yaxis = :log, label = "bps")
plot!(hh, b3, xaxis = :log, yaxis = :log, label = "zz")
savefig("./scripts/"*str_app*"/output/bias.pdf")




str_regression = "logistic_regression"


using LinearAlgebra, CSV, DataFrames, Plots
include("./../src/utilities.jl")

println("...defining gradient...")
include("./"*str_regression*"/grad.jl")



# hh = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
hh = [1e-02, 5e-03, 1e-03, 5e-04, 1e-04, 5e-05, 1e-05, 5e-06, 1e-06]

#Store the results
results = fill(NaN, length(hh), 5)

# INPUT STRINGS
str_folder = "./scripts/"*str_regression*"/stein_distance/posterior_samples/"
str_h = "h_"
str_csv = ".csv"
str_data = "data"


γ0 = 1/10
DIRINDATA = str_folder*str_data*str_csv 
data  = Matrix(CSV.read(DIRINDATA, DataFrame, header=false))
A, y = Matrix(data[:,1:(end-1)]), Vector(data[:,end])
At = A'
p = size(At, 1)
nobs = size(At, 2)
mb = Int(round(0.01*length(y))) #minibatch size, e.g. 1% of full data
Niter_opt = 10^6
x0 = randn(p)
cv = AdamCV(∇U!, ∇Ufull, x0, nobs, mb, Niter_opt, y, At)
println("norm of gradient at control variates: $(norm(cv.∇x0))")

Γ =  ΔU(cv.x0, y, At, γ0)
dd = sqrt.(diag(Γ))
σ = 1.0./dd
for i in eachindex(hh)
    h = hh[i]
    println("h = $(h)")
    str_sampler = "sgld1_" # "sgld2_" "zz_" "bps_" "szz_"
    DIRIN = str_folder*str_sampler*str_h*string(h)*str_csv 
    if isfile(DIRIN)
        trace = Matrix(CSV.read(DIRIN, DataFrame; header=false))
        results[i, 1] = 1/length(σ)*sum(abs2, (std(trace, dims = 2) .- σ)./σ)
    end
    str_sampler = "sgld2_" # "sgld2_" "zz_" "bps_" "szz_"
    DIRIN = str_folder*str_sampler*str_h*string(h)*str_csv 
    if isfile(DIRIN)
        trace = Matrix(CSV.read(DIRIN, DataFrame; header=false))
        results[i, 2] = 1/length(σ)*sum(abs2, (std(trace, dims = 2) .- σ)./σ)
    end

    str_sampler = "sgld3_" # "sgld2_" "zz_" "bps_" "szz_"
    DIRIN = str_folder*str_sampler*str_h*string(h)*str_csv 
    if isfile(DIRIN)
        trace = Matrix(CSV.read(DIRIN, DataFrame; header=false))
        results[i, 3] = 1/length(σ)*sum(abs2, (std(trace, dims = 2) .- σ)./σ)
    end
    str_sampler = "zz_" # "sgld2_" "zz_" "bps_" "szz_"
    DIRIN = str_folder*str_sampler*str_h*string(h)*str_csv 
    if isfile(DIRIN)
        trace = Matrix(CSV.read(DIRIN, DataFrame; header=false))
        results[i, 4] = 1/length(σ)*sum(abs2, (std(trace, dims = 2) .- σ)./σ)
    end
    str_sampler = "bps_" # "sgld2_" "zz_" "bps_" "szz_"
    DIRIN = str_folder*str_sampler*str_h*string(h)*str_csv
    if isfile(DIRIN)
        trace = Matrix(CSV.read(DIRIN, DataFrame; header=false))
        results[i, end] = 1/length(σ)*sum(abs2, (std(trace, dims = 2) .- σ)./σ)
    end
end


res = [hh results]
CSV.write("./scripts/"*str_regression*"/stein_distance/output/laplace.csv", DataFrame(res, :auto), header = false)                


f1 = plot(title = "Distance to LA: "*str_regression, hh[3:end], linestyle = :dash, results[3:end,1], xaxis = :log, label = "sgld1", legend=:outertopright )
plot!(f1, hh[3:end], results[3:end, 2],  linestyle = :dash, label = "sgld10")
plot!(f1, hh[3:end], results[3:end, 3],  linestyle = :dash, label = "sgld100")
plot!(f1, hh[3:end], results[3:end, 4], label = "zz")
plot!(f1, hh[3:end], results[3:end, 5], label = "bps")
f1
savefig(f1, "./scripts/"*str_regression*"/stein_distance/output/laplace.pdf")       
f1     
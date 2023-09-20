# str_regression = "linear_regression"
str_regression = "logistic_regression"
# str_regression = "poisson_regression"

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
c = 1.0
β = -0.5
γ0 = 1/10
DIRINDATA = str_folder*str_data*str_csv 
data  = Matrix(CSV.read(DIRINDATA, DataFrame, header=false))
A, y = Matrix(data[:,1:(end-1)]), Vector(data[:,end])
At = A'
for i in eachindex(hh)
    h = hh[i]
    println("h = $(h)")
    str_sampler = "sgld1_" # "sgld2_" "zz_" "bps_" "szz_"
    DIRIN = str_folder*str_sampler*str_h*string(h)*str_csv 
    if isfile(DIRIN)
        trace = Matrix(CSV.read(DIRIN, DataFrame; header=false))
        trace = reshapetov(trace)
        results[i, 1] =  stein_kernel(∇Ufull, trace, c, β, y, At, γ0)
    end
    str_sampler = "sgld2_" # "sgld2_" "zz_" "bps_" "szz_"
    DIRIN = str_folder*str_sampler*str_h*string(h)*str_csv 
    if isfile(DIRIN)
        trace = Matrix(CSV.read(DIRIN, DataFrame; header=false))
        trace = reshapetov(trace)
        results[i, 2] = stein_kernel(∇Ufull, trace, c, β, y, At, γ0)
    end

    str_sampler = "sgld3_" # "sgld2_" "zz_" "bps_" "szz_"
    DIRIN = str_folder*str_sampler*str_h*string(h)*str_csv 
    if isfile(DIRIN)
        trace = Matrix(CSV.read(DIRIN, DataFrame; header=false))
        trace = reshapetov(trace)
        results[i, 3] = stein_kernel(∇Ufull, trace, c, β, y, At, γ0)
    end
    str_sampler = "zz_" # "sgld2_" "zz_" "bps_" "szz_"
    DIRIN = str_folder*str_sampler*str_h*string(h)*str_csv 
    if isfile(DIRIN)
        trace = Matrix(CSV.read(DIRIN, DataFrame; header=false))
        trace = reshapetov(trace)
        results[i, 4] = stein_kernel(∇Ufull, trace, c, β, y, At, γ0)
    end
    str_sampler = "bps_" # "sgld2_" "zz_" "bps_" "szz_"
    DIRIN = str_folder*str_sampler*str_h*string(h)*str_csv
    if isfile(DIRIN)
        trace = Matrix(CSV.read(DIRIN, DataFrame; header=false))
        trace = reshapetov(trace)
        results[i, end] = stein_kernel(∇Ufull, trace, c, β, y, At, γ0)
    end
end


res = [hh results]
CSV.write("./scripts/"*str_regression*"/stein_distance/output/output.csv", DataFrame(res, :auto), header = false)                

f1 = plot(title = "Stein Discrepancy "*str_regression, hh, results[:,1], xaxis = :log, label = "sgld1", legend=:outertopright )
plot!(f1, hh, results[:, 2], label = "sgld10")
plot!(f1, hh, results[:, 3], label = "sgld100")
plot!(f1, hh, results[:, 4], label = "bps")
plot!(f1, hh, results[:, 5], label = "zz")

savefig(f1, "./scripts/"*str_regression*"/stein_distance/output/output.png")
CSV.write("./scripts/"*str_regression*"/stein_distance/output/output.csv", DataFrame(res, :auto), header = false)                
str_app = "toy"
using LinearAlgebra, CSV, DataFrames, Plots. StatsBase
include("./../src/utilities.jl")
σ2
γ0 = 1/10
p = 5
str_d = "d_"
str_csv = ".csv"
str_data = "./scripts/"*str_app*"/data/"
DIR_DATA = str_data*str_d*string(p)*str_csv 
data  = Matrix(CSV.read(DIR_DATA, DataFrame, header=false))
A, y = Matrix(data[:,1:(end-1)]), Vector(data[:,end])
Σ = inv(A'A + γ0*I)*σ2
σ = sqrt.(diag(Σ))
hh = [0.1, collect(0.05:-0.01:0.02)..., collect(0.01:-0.002:0.001)...,collect(0.001:-0.0002:0.0002)..., 0.0001]# hh = [0.1,]

# pp = [10, 50, 100, 200, 300, 400, 500, 600, 700,  800,  900, 1000]
burnin = 1
# INPUT STRINGS
res = zeros(3, length(hh))
count = 0
for h in hh
    count += 1
    #Store the results
    
    str_trace = "./scripts/"*str_app*"/posterior_samples/"
    str_h = "h_"
    str_csv = ".csv"
    γ0 = 1/10
    println("h = $(h)")
    str_sampler = "sgld1_" # "sgld2_" "zz_" "bps_" "szz_"
    DIRIN = str_trace*str_sampler*str_h*string(h)*str_csv
    if isfile(DIRIN)
        trace = Matrix(CSV.read(DIRIN, DataFrame; header=false))
        res[1,count] = 1/p*sum(abs2, (std(trace[:,burnin:end], dims = 2) - σ)./σ)
        # res[1,count] = var(trace[1,burnin:end])
    end
    # str_sampler = "sgld2_" # "sgld2_" "zz_" "bps_" "szz_"
    # DIRIN = str_trace*str_sampler*str_h*string(h)*str_d*string(d)*str_csv 
    # if isfile(DIRIN)
    #     trace = Matrix(CSV.read(DIRIN, DataFrame; header=false))
    #     CSV.write("./scripts/"*str_regression*"/scaling/variances/"*str_sampler*str_h*string(h)*"_"*str_d*string(d)*".csv", DataFrame((std(trace, dims = 2) - σ)./σ, :auto), header = false)
    # end

    # str_sampler = "sgld3_" # "sgld2_" "zz_" "bps_" "szz_"
    # DIRIN = str_trace*str_sampler*str_h*string(h)*str_d*string(d)*str_csv 
    # if isfile(DIRIN)
    #     trace = Matrix(CSV.read(DIRIN, DataFrame; header=false))
    #     CSV.write("./scripts/"*str_regression*"/scaling/variances/"*str_sampler*str_h*string(h)*"_"*str_d*string(d)*".csv", DataFrame((std(trace, dims = 2) - σ)./σ, :auto), header = false)
    # end
    str_sampler = "zz_" # "sgld2_" "zz_" "bps_" "szz_"
    DIRIN = str_trace*str_sampler*str_h*string(h)*str_csv 
    trace = Matrix(CSV.read(DIRIN, DataFrame; header=false))
    res[2,count] = 1/p*sum(abs2, (std(trace[:,burnin:end], dims = 2) - σ)./σ)
    # res[2,count] = var(trace[1,burnin:end])
    str_sampler = "bps_" # "sgld2_" "zz_" "bps_" "szz_"
    DIRIN = str_trace*str_sampler*str_h*string(h)*str_csv
    trace = Matrix(CSV.read(DIRIN, DataFrame; header=false))
    res[3,count] = 1/p*sum(abs2, (std(trace[:,burnin:end], dims = 2) - σ)./σ)
    # res[3,count] = var(trace[1,burnin:end])

end
res[1,8] = NaN
using Plots
plot(hh[2:end], res[2,2:end], xaxis = :log, label = "sg-zz", legend=:topleft, title = "Comparison linear regression" )
plot!(hh[2:end], res[3,2:end], xaxis = :log, label = "sg-bps" )
plot!(hh[2:end], res[1,2:end], xaxis = :log, label = "sgld" )
savefig("./linear_regression.pdf")
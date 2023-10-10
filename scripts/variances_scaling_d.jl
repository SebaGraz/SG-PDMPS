
str_regression = "linear_regression"

using LinearAlgebra, CSV, DataFrames, Plots. StatsBase
include("./../src/utilities.jl")

println("...defining gradient...")
include("./"*str_regression*"/grad.jl")

hh = [5e-07]
# pp = [10, 50, 100, 200, 300, 400, 500, 600, 700,  800,  900, 1000]
pp = [10, 20, 30, 40, 50, 60, 70, 100]
burnin = 100
# INPUT STRINGS
for h in hh
    for d in pp
        #Store the results
        res = fill(d, 5)
        str_data = "./scripts/"*str_regression*"/scaling/data/"
        str_trace = "./scripts/"*str_regression*"/scaling/posterior_samples/"
        str_h = "h_"
        str_d = "d_"
        str_csv = ".csv"
        str_dat = "data_"
        γ0 = 1/10
        DIRINDATA = str_data*str_dat*str_d*string(d)*str_csv 
        data  = Matrix(CSV.read(DIRINDATA, DataFrame, header=false))
        A, y = Matrix(data[:,1:(end-1)]), Vector(data[:,end])
        At = A'
        Σ = inv((A'A + I*γ0))
        σ = sqrt.(diag(Σ))
        p = size(At, 1)
        nobs = size(At, 2)
        println("h = $(h)")
        str_sampler = "sgld1_" # "sgld2_" "zz_" "bps_" "szz_"
        DIRIN = str_trace*str_sampler*str_h*string(h)*str_d*string(d)*str_csv
        true_variance = A 
        if isfile(DIRIN)
            trace = Matrix(CSV.read(DIRIN, DataFrame; header=false))
            CSV.write("./scripts/"*str_regression*"/scaling/variances/"*str_sampler*str_h*string(h)*str_d*string(d)*".csv", DataFrame((std(trace[:,burnin:end], dims = 2) - σ)./σ, :auto), header = false)
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
        DIRIN = str_trace*str_sampler*str_h*string(h)*str_d*string(d)*str_csv 
        if isfile(DIRIN)
            trace = Matrix(CSV.read(DIRIN, DataFrame; header=false))
            CSV.write("./scripts/"*str_regression*"/scaling/variances/"*str_sampler*str_h*string(h)*str_d*string(d)*".csv", DataFrame((std(trace[:,burnin:end], dims = 2) - σ)./σ, :auto), header = false)
        end
        str_sampler = "bps_" # "sgld2_" "zz_" "bps_" "szz_"
        DIRIN = str_trace*str_sampler*str_h*string(h)*str_d*string(d)*str_csv
        if isfile(DIRIN)
            trace = Matrix(CSV.read(DIRIN, DataFrame; header=false))
            CSV.write("./scripts/"*str_regression*"/scaling/variances/"*str_sampler*str_h*string(h)*str_d*string(d)*".csv", DataFrame((std(trace[:,burnin:end], dims = 2) - σ)./σ, :auto), header = false)
        end
    end
end

str_regression = "linear_regression"


using LinearAlgebra, CSV, DataFrames, Plots
include("./../src/utilities.jl")

println("...defining gradient...")
include("./"*str_regression*"/grad.jl")


hh = [1e-03, 1e-04, 1e-05]
pp = [50, 100]
# INPUT STRINGS
for h in hh
    for d in pp
        #Store the results
        res = fill(d, 10)
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
        p = size(At, 1)
        nobs = size(At, 2)
        println("h = $(h)")
        str_sampler = "sgld1_" # "sgld2_" "zz_" "bps_" "szz_"
        DIRIN = str_trace*str_sampler*str_h*string(h)*str_d*string(d)*str_csv 
        if isfile(DIRIN)
            trace = Matrix(CSV.read(DIRIN, DataFrame; header=false))
            res = zeros(d, 10)
            c = autocorr(trace, 0)
            res = [autocorr(trace, lag)./c for lag in 1:10]
            CSV.write("./scripts/"*str_regression*"/scaling/autocorrelation/"*str_sampler*str_h*string(h)*"_"*str_d*string(d)*".csv", DataFrame(res, :auto), header = false)
        end
        str_sampler = "sgld2_" # "sgld2_" "zz_" "bps_" "szz_"
        DIRIN = str_trace*str_sampler*str_h*string(h)*str_d*string(d)*str_csv 
        if isfile(DIRIN)
            trace = Matrix(CSV.read(DIRIN, DataFrame; header=false))
            res = zeros(d, 10)
            c = autocorr(trace, 0)
            res = [autocorr(trace, lag)./c for lag in 1:10]
            CSV.write("./scripts/"*str_regression*"/scaling/autocorrelation/"*str_sampler*str_h*string(h)*"_"*str_d*string(d)*".csv", DataFrame(res, :auto), header = false)
        end

        str_sampler = "sgld3_" # "sgld2_" "zz_" "bps_" "szz_"
        DIRIN = str_trace*str_sampler*str_h*string(h)*str_d*string(d)*str_csv 
        if isfile(DIRIN)
            trace = Matrix(CSV.read(DIRIN, DataFrame; header=false))
            res = zeros(d, 10)
            c = autocorr(trace, 0)
            res = [autocorr(trace, lag)./c for lag in 1:10]
            CSV.write("./scripts/"*str_regression*"/scaling/autocorrelation/"*str_sampler*str_h*string(h)*"_"*str_d*string(d)*".csv", DataFrame(res, :auto), header = false)
        end
        str_sampler = "zz_" # "sgld2_" "zz_" "bps_" "szz_"
        DIRIN = str_trace*str_sampler*str_h*string(h)*str_d*string(d)*str_csv 
        if isfile(DIRIN)
            trace = Matrix(CSV.read(DIRIN, DataFrame; header=false))
            res = zeros(d, 10)
            c = autocorr(trace, 0)
            res = [autocorr(trace, lag)./c for lag in 1:10]
            CSV.write("./scripts/"*str_regression*"/scaling/autocorrelation/"*str_sampler*str_h*string(h)*"_"*str_d*string(d)*".csv", DataFrame(res, :auto), header = false)
        end
        str_sampler = "bps_" # "sgld2_" "zz_" "bps_" "szz_"
        DIRIN = str_trace*str_sampler*str_h*string(h)*str_d*string(d)*str_csv
        if isfile(DIRIN)
            trace = Matrix(CSV.read(DIRIN, DataFrame; header=false))
            res = zeros(d, 10)
            c = autocorr(trace, 0)
            res = [autocorr(trace, lag) for lag in 1:10]
            CSV.write("./scripts/"*str_regression*"/scaling/autocorrelation/"*str_sampler*str_h*string(h)*"_"*str_d*string(d)*".csv", DataFrame(res, :auto), header = false)
        end
    end
end
error("")

# res = [hh results]
# CSV.write("./scripts/"*str_regression*"/stein_distance/output/laplace.csv", DataFrame(res, :auto), header = false)                


# f1 = plot(title = "Distance to LA: "*str_regression, hh, linestyle = :dash, results[:,1], xaxis = :log, label = "sgld1", legend=:outertopright )
# plot!(f1, hh, results[:, 2],  linestyle = :dash, label = "sgld10")
# plot!(f1, hh, results[:, 3],  linestyle = :dash, label = "sgld100")
# plot!(f1, hh, results[:, 4], label = "zz")
# plot!(f1, hh, results[:, 5], label = "bps")
# f1
# savefig(f1, "./scripts/"*str_regression*"/stein_distance/output/laplace.png")            
# str_regression = "linear_regression"
str_regression = "logistic_regression"
# str_regression = "poisson_regression"

using CSV, DataFrames, StatsBase, LinearAlgebra
include("./../src/utilities.jl")

# hh = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
hh = [1e-02, 5e-03, 1e-03, 5e-04, 1e-04, 5e-05, 1e-05, 5e-06, 1e-06]

#Store the results
sgld1_results = zeros(length(hh),10);
sgld2_results = similar(sgld1_results);
sgld3_results = similar(sgld1_results);
bps_results = similar(sgld1_results);
zz_results = similar(sgld1_results);
szz_results = similar(sgld1_results);

sgld1_results2 = zeros(length(hh),10);
sgld2_results2 = similar(sgld1_results);
sgld3_results2 = similar(sgld1_results);
bps_results2 = similar(sgld1_results);
zz_results2 = similar(sgld1_results);
szz_results2 = similar(sgld1_results);

# INPUT STRINGS
str_folder = "./scripts/"*str_regression*"/prediction/posterior_samples/"
str_h = "h_"
str_iter = "_iter_"
str_csv = ".csv"
str_data = "test_data"

for iter in 1:10
    println("iter = $(iter)")
    for i in eachindex(hh)
        h = hh[i]
        DIRINDATA = str_folder*str_data*str_iter*string(iter)*str_csv 
        data  = Matrix(CSV.read(DIRINDATA, DataFrame, header=false))
        X_test, y_test = Matrix(data[:,1:(end-1)]), Vector(data[:,end])
        str_sampler = "sgld1_" # "sgld2_" "zz_" "bps_" "szz_"
        DIRIN = str_folder*str_sampler*str_h*string(h)*str_iter*string(iter)*str_csv 
        trace = Matrix(CSV.read(DIRIN, DataFrame; header=false))
        sgld1_results[i, iter], sgld1_results2[i, iter]  = evaluation((X_test,y_test), trace)

        str_sampler = "sgld2_" # "sgld2_" "zz_" "bps_" "szz_"
        DIRIN = str_folder*str_sampler*str_h*string(h)*str_iter*string(iter)*str_csv 
        trace = Matrix(CSV.read(DIRIN, DataFrame; header=false))
        sgld2_results[i, iter], sgld2_results2[i, iter]  = evaluation((X_test,y_test), trace)

        str_sampler = "sgld3_" # "sgld2_" "zz_" "bps_" "szz_"
        DIRIN = str_folder*str_sampler*str_h*string(h)*str_iter*string(iter)*str_csv 
        trace = Matrix(CSV.read(DIRIN, DataFrame; header=false))
        sgld3_results[i, iter], sgld3_results2[i, iter]   = evaluation((X_test,y_test), trace)

        str_sampler = "zz_" # "sgld2_" "zz_" "bps_" "szz_"
        DIRIN = str_folder*str_sampler*str_h*string(h)*str_iter*string(iter)*str_csv 
        trace = Matrix(CSV.read(DIRIN, DataFrame; header=false))
        zz_results[i, iter], zz_results2[i, iter]  = evaluation((X_test,y_test), trace)

        str_sampler = "szz_" # "sgld2_" "zz_" "bps_" "szz_"
        DIRIN = str_folder*str_sampler*str_h*string(h)*str_iter*string(iter)*str_csv 
        trace = Matrix(CSV.read(DIRIN, DataFrame; header=false))
        szz_results[i, iter], szz_results2[i, iter]  = evaluation((X_test,y_test), trace)

        str_sampler = "bps_" # "sgld2_" "zz_" "bps_" "szz_"
        DIRIN = str_folder*str_sampler*str_h*string(h)*str_iter*string(iter)*str_csv 
        trace = Matrix(CSV.read(DIRIN, DataFrame; header=false))
        bps_results[i, iter], bps_results2[i, iter]  = evaluation((X_test,y_test), trace)
    end
end


using Plots
f1 = plot(title = "Predicition power "*str_regression, hh, mean(sgld1_results, dims = 2)[:], xaxis = :log, label = "sgld1")
plot!(f1, hh, mean(sgld2_results, dims = 2)[:], label = "sgld10")
plot!(f1, hh, mean(sgld3_results, dims = 2)[:], label = "sgld100")
plot!(f1, hh, mean(bps_results, dims = 2)[:], label = "bps")
plot!(f1, hh, mean(zz_results, dims = 2)[:], label = "zz")
plot!(f1, hh, mean(szz_results, dims = 2)[:], label = "szz")

f2 = plot(title = "Average Loss "*str_regression, hh, linestyle=:dash , mean(sgld1_results2, dims = 2)[:], xaxis = :log, label = "sgld1")
plot!(f2, hh, mean(sgld2_results2, dims = 2)[:], linestyle=:dash ,label = "sgld10")
plot!(f2, hh, mean(sgld3_results2, dims = 2)[:], linestyle=:dash , label = "sgld100")
plot!(f2, hh, mean(bps_results2, dims = 2)[:], label = "bps")
plot!(f2, hh, mean(zz_results2, dims = 2)[:], label = "zz")
plot!(f2, hh, mean(szz_results2, dims = 2)[:], label = "szz")

f21 = plot(hh, mean(bps_results2, dims = 2)[:], label = "bps", xaxis = :log)
plot!(f21, hh, mean(zz_results2, dims = 2)[:], label = "zz")
plot!(f21, hh, mean(szz_results2, dims = 2)[:], label = "szz")


f3 = plot(title = "Max Loss "*str_regression, hh, maximum(sgld1_results2, dims = 2)[:], xaxis = :log, label = "sgld1")
plot!(f3, hh, maximum(sgld2_results2, dims = 2)[:], label = "sgld10")
plot!(f3, hh, maximum(sgld3_results2, dims = 2)[:], label = "sgld100")
plot!(f3, hh, maximum(bps_results2, dims = 2)[:], label = "bps")
plot!(f3, hh, maximum(zz_results2, dims = 2)[:], label = "zz")
plot!(f3, hh, maximum(szz_results2, dims = 2)[:], label = "szz")

savefig(f2, "./scripts/"*str_regression*"/prediction/output/average_loss.pdf")

using CSV, DataFrames, StatsBase, LinearAlgebra
include("./../../../src/utilities.jl")

   
function evaluation(testdata, param)
    X, y = testdata
    d = size(param, 2)
    res = 0.0
    for j in 1:d
        res += norm2(y - X*param[:, j])/d
    end
    return res
end



#Store the results
sgld1_results = zeros(length(hh),10);
sgld2_results = similar(sgld1_results);
sgld3_results = similar(sgld1_results);
bps_results = similar(sgld1_results);
zz_results = similar(sgld1_results);
szz_results = similar(sgld1_results);
str_txt = out*"INFO.txt"
    file = open(str_txt, "w")
# INPUT STRINGS
str_folder = "./scripts/linear_regression/prediction/posterior_samples/"
str_h = "h_"
str_iter = "_iter_"
str_csv = ".csv"
str_data = "test_data"

for iter in 1:10
    for i in eachindex(hh)
        h = hh[i]
        DIRINDATA = str_folder*str_data*str_iter*string(iter)*str_csv 
        data  = Matrix(CSV.read(DIRINDATA, DataFrame, header=false))
        X_test, y_test = Matrix(data[:,1:(end-1)]), Vector(data[:,end])
        str_sampler = "sgld1_" # "sgld2_" "zz_" "bps_" "szz_"
        DIRIN = str_folder*str_sampler*str_h*string(h)*str_iter*string(iter)*str_csv 
        trace = Matrix(CSV.read(DIRIN, DataFrame; header=false))
        sgld1_results[i, iter]  = evaluation((X_test,y_test), trace)

        str_sampler = "sgld2_" # "sgld2_" "zz_" "bps_" "szz_"
        DIRIN = str_folder*str_sampler*str_h*string(h)*str_iter*string(iter)*str_csv 
        trace = Matrix(CSV.read(DIRIN, DataFrame; header=false))
        sgld2_results[i, iter]  = evaluation((X_test,y_test), trace)

        str_sampler = "sgld3_" # "sgld2_" "zz_" "bps_" "szz_"
        DIRIN = str_folder*str_sampler*str_h*string(h)*str_iter*string(iter)*str_csv 
        trace = Matrix(CSV.read(DIRIN, DataFrame; header=false))
        sgld3_results[i, iter]  = evaluation((X_test,y_test), trace)

        str_sampler = "zz_" # "sgld2_" "zz_" "bps_" "szz_"
        DIRIN = str_folder*str_sampler*str_h*string(h)*str_iter*string(iter)*str_csv 
        trace = Matrix(CSV.read(DIRIN, DataFrame; header=false))
        zz_results[i, iter]  = evaluation((X_test,y_test), trace)

        str_sampler = "szz_" # "sgld2_" "zz_" "bps_" "szz_"
        DIRIN = str_folder*str_sampler*str_h*string(h)*str_iter*string(iter)*str_csv 
        trace = Matrix(CSV.read(DIRIN, DataFrame; header=false))
        szz_results[i, iter]  = evaluation((X_test,y_test), trace)

        str_sampler = "bps_" # "sgld2_" "zz_" "bps_" "szz_"
        DIRIN = str_folder*str_sampler*str_h*string(h)*str_iter*string(iter)*str_csv 
        trace = Matrix(CSV.read(DIRIN, DataFrame; header=false))
        bps_results[i, iter] = evaluation((X_test,y_test), trace)
    end
end


using Plots
f1 = plot(title = "MSE", hh, mean(sgld1_results, dims = 2)[:], xaxis = :log, label = "sgld1", linestyle = :dash)
plot!(f1, hh, mean(sgld2_results, dims = 2)[:], label = "sgld10", linestyle = :dash)
plot!(f1, hh, mean(sgld3_results, dims = 2)[:], label = "sgld100", linestyle = :dash)
plot!(f1, hh, mean(bps_results, dims = 2)[:], label = "bps")
plot!(f1, hh, mean(zz_results, dims = 2)[:], label = "zz")
plot!(f1, hh, mean(szz_results, dims = 2)[:], label = "szz")

f1
savefig(fout, "./scripts/logistic_regression/prediction/output.png")
sgld1_results    
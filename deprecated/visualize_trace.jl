using Plots, CSV, DataFrames

# INPUT STRINGS
str_folder = "scripts/logistic_regression/prediction/posterior_samples/"
str_h = "h_"
str_iter = "_iter_"
str_csv = ".csv"
str_data = "test_data"


h = 0.01
iter = 1
str_sampler = "sgld1_" # "sgld2_" "zz_" "bps_" "szz_"
DIRIN = str_folder*str_sampler*str_h*string(h)*str_iter*string(iter)*str_csv 
trace = Matrix(CSV.read(DIRIN, DataFrame, header=false))
f1 = plot(trace[2,:],  label = "sgld1", color = :red)

str_sampler = "sgld2_" # "sgld2_" "zz_" "bps_" "szz_"
DIRIN = str_folder*str_sampler*str_h*string(h)*str_iter*string(iter)*str_csv 
trace = Matrix(CSV.read(DIRIN, DataFrame, header=false))
f2 = plot( trace[2,:],  label = "sgld10",  color = :black)

str_sampler = "sgld3_" # "sgld2_" "zz_" "bps_" "szz_"
DIRIN = str_folder*str_sampler*str_h*string(h)*str_iter*string(iter)*str_csv 
trace = Matrix(CSV.read(DIRIN, DataFrame, header=false))
f3 = plot(trace[2,:],  label = "sgld100",  color = :black)

str_sampler = "zz_" # "sgld2_" "zz_" "bps_" "szz_"
DIRIN = str_folder*str_sampler*str_h*string(h)*str_iter*string(iter)*str_csv 
trace = Matrix(CSV.read(DIRIN, DataFrame, header=false))
f4 = plot(trace[2,:],  label = "zz",  color = :blue)

str_sampler = "bps_" # "sgld2_" "zz_" "bps_" "szz_"
DIRIN = str_folder*str_sampler*str_h*string(h)*str_iter*string(iter)*str_csv 
trace = Matrix(CSV.read(DIRIN, DataFrame, header=false))
f5 = plot(trace[2,:], label = "bps",  color = :black)


str_sampler = "szz_" # "sgld2_" "zz_" "bps_" "szz_"
DIRIN = str_folder*str_sampler*str_h*string(h)*str_iter*string(iter)*str_csv 
trace = Matrix(CSV.read(DIRIN, DataFrame, header=false))
f6 = plot(trace[2,:], label = "szz",  color = :black)
l = @layout [a b c; d e f]
f = plot(f1,f2,f3,f4,f5,f6, layout = l, plot_title = "traces h = $h")
savefig(f, "./scripts/logistic_regression/prediction/output_traces_h00001.png")







DIRINDATA = str_folder*str_data*str_iter*string(iter)*str_csv 
data  = Matrix(CSV.read(DIRINDATA, DataFrame, header=false))
X_test, y_test = Matrix(data[:,1:(end-1)]), Vector(data[:,end])
evaluation((X_test,y_test), trace)


function ∇Uj(x, j, y, At)
    At[:,j] *(sigmoid(dot(x,At[:, j])) - y[j])
end

# args = y, At,
function ∇Ufull(x, y, At)
    nobs = size(At, 2)
    ∇Ux = zeros(length(x))
    for j in 1:nobs
        ∇Ux .+= ∇Uj(x, j, y, At)
    end
    return ∇Ux 
end
sigmoid(x) = inv(one(x) + exp(-x))
lsigmoid(x) = -log(one(x) + exp(-x))

CSV.read()
data = Matrix(CSV.read("./scripts/logistic_regression/prediction/posterior_samples/test_data_iter_1.csv", DataFrame, header=false))
X_test, y_test = Matrix(data[:,1:(end-1)]), Vector(data[:,end])
Xt = X_test'

[trace[:,1] ∇Ufull(trace[:,1], y_test, Xt) ]
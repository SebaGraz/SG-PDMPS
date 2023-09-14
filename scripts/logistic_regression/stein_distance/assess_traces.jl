using Plots

# INPUT STRINGS
str_folder = "./scripts/logistic_regression/stein_distance/posterior_samples/"
str_h = "h_"
str_csv = ".csv"
str_data = "data"

h = 0.1
str_sampler = "sgld1_" # "sgld2_" "zz_" "bps_" "szz_"
DIRIN = str_folder*str_sampler*str_h*string(h)*str_csv 
trace = Matrix(CSV.read(DIRIN, DataFrame, header=false))
f1 = plot(trace[2,:], title = "trace first coordinate h = $(h)", label = "sgld1", alpha = 0.2, color = :red)

str_sampler = "sgld2_" # "sgld2_" "zz_" "bps_" "szz_"
DIRIN = str_folder*str_sampler*str_h*string(h)*str_csv 
trace = Matrix(CSV.read(DIRIN, DataFrame, header=false))
plot!(f1, trace[2,:], title = "trace first coordinate h = $(h)", label = "sgld10", alpha = 0.2, color = :pink)

str_sampler = "sgld3_" # "sgld2_" "zz_" "bps_" "szz_"
DIRIN = str_folder*str_sampler*str_h*string(h)*str_csv 
trace = Matrix(CSV.read(DIRIN, DataFrame, header=false))
plot!(f1, trace[2,:], title = "trace first coordinate h = $(h)", label = "sgld100", alpha = 0.2, color = :yellow)

str_sampler = "zz_" # "sgld2_" "zz_" "bps_" "szz_"
DIRIN = str_folder*str_sampler*str_h*string(h)*str_csv 
trace = Matrix(CSV.read(DIRIN, DataFrame, header=false))
plot!(f1, trace[2,:], title = "trace first coordinate h = $(h)", label = "zz", alpha = 0.2, color = :blue)

str_sampler = "bps_" # "sgld2_" "zz_" "bps_" "szz_"
DIRIN = str_folder*str_sampler*str_h*string(h)*str_csv 
trace = Matrix(CSV.read(DIRIN, DataFrame, header=false))
plot!(f1, trace[2,:], title = "trace first coordinate h = $(h)", label = "sgld1", alpha = 0.2, color = :black)

savefig(f1, "./scripts/logistic_regression/stein_distance/output_traces.png")







DIRINDATA = str_folder*str_data*str_iter*string(iter)*str_csv 
data  = Matrix(CSV.read(DIRINDATA, DataFrame, header=false))
X_test, y_test = Matrix(data[:,1:(end-1)]), Vector(data[:,end])
evaluation((X_test,y_test), trace)

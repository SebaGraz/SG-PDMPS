using Plots

# INPUT STRINGS
str_folder = "scripts/logistic_regression/prediction/posterior_samples/"
str_h = "h_"
str_iter = "_iter_"
str_csv = ".csv"
str_data = "test_data"


h = 0.0001
iter = 1
str_sampler = "sgld1_" # "sgld2_" "zz_" "bps_" "szz_"
DIRIN = str_folder*str_sampler*str_h*string(h)*str_iter*string(iter)*str_csv 
trace = Matrix(CSV.read(DIRIN, DataFrame, header=false))
f1 = plot(trace[2,:],  label = "sgld1", color = :red)

str_sampler = "sgld2_" # "sgld2_" "zz_" "bps_" "szz_"
DIRIN = str_folder*str_sampler*str_h*string(h)*str_iter*string(iter)*str_csv 
trace = Matrix(CSV.read(DIRIN, DataFrame, header=false))
f2 = plot( trace[2,:],  label = "sgld10",  color = :pink)

str_sampler = "sgld3_" # "sgld2_" "zz_" "bps_" "szz_"
DIRIN = str_folder*str_sampler*str_h*string(h)*str_iter*string(iter)*str_csv 
trace = Matrix(CSV.read(DIRIN, DataFrame, header=false))
f3 = plot(trace[2,:],  label = "sgld100",  color = :yellow)

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

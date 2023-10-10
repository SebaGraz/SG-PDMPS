using CVS, DataFrames
include("./../src/utilities.jl")
str_data = "./scripts/neural_net/test_data/"
dataset = "boston_"
iter = 1
str_iter = "iter_" 
str_csv =".csv"
data = Matrix(CSV.read(str_data*dataset*str_iter*string(iter)*str_csv, DataFrame; header=false))
X_test = data[:,1:end-1]
y_test = data[:,end]
using Flux
nn_initial = Chain(Dense(size(X_test)[2]=>50),Dense(50=>1,relu)) 
parameters_initial, reconstruct = Flux.destructure(nn_initial)

str_trace = "./scripts/neural_net/prediction/"
str_sampler = "bps_"
str_h = "h_"
h = 0.001
trace = Matrix(CSV.read(str_trace*dataset*str_sampler*str_h*string(h)*"_"*str_iter*string(iter)*str_csv, DataFrame; header=false))
nn_forward(x, theta) = reconstruct(theta)(x)

nn_forward(X_test[1,:], trace[:,4])

y_test

# using ReverseDiff
# Turing.setadbackend(:reversediff)

# Extract weights and a helper function to reconstruct NN from weights
parameters_initial, reconstruct = Flux.destructure(nn_initial)
using LinearAlgebra
include("./../../../src/utilities.jl")



sigmoid(x) = inv(one(x) + exp(-x))
lsigmoid(x) = -log(one(x) + exp(-x))
# gradient controbution from ith data point
function ∇Uj(x, j, y, At)
    At[:,j] *(sigmoid(dot(x,At[:, j])) - y[j])
end

# args = y, At,
function ∇Ufull(x, y, At, γ0)
    nobs = size(At, 2)
    ∇Ux = γ0*x
    for j in 1:nobs
        ∇Ux .+= ∇Uj(x, j, y, At)
    end
    return ∇Ux 
end

hh = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
#Store the results
sgld1_results = zeros(length(hh));
sgld2_results = similar(sgld1_results);
sgld3_results = similar(sgld1_results);
bps_results = similar(sgld1_results);
zz_results = similar(sgld1_results);

# INPUT STRINGS
str_folder = "./scripts/logistic_regression/stein_distance/posterior_samples/"
str_h = "h_"
str_csv = ".csv"
str_data = "data"
c = 2.0
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
    trace = Matrix(CSV.read(DIRIN, DataFrame; header=false))
    sgld1_results[i] = stein_metric(∇Ufull, trace, c, β, y, At, γ0)

    str_sampler = "sgld2_" # "sgld2_" "zz_" "bps_" "szz_"
    DIRIN = str_folder*str_sampler*str_h*string(h)*str_csv 
    trace = Matrix(CSV.read(DIRIN, DataFrame; header=false))
    sgld2_results[i] = stein_metric(∇Ufull, trace, c, β, y, At, γ0)

    str_sampler = "sgld3_" # "sgld2_" "zz_" "bps_" "szz_"
    DIRIN = str_folder*str_sampler*str_h*string(h)*str_csv 
    trace = Matrix(CSV.read(DIRIN, DataFrame; header=false))
    sgld3_results[i] = stein_metric(∇Ufull, trace, c, β, y, At, γ0)

    str_sampler = "zz_" # "sgld2_" "zz_" "bps_" "szz_"
    DIRIN = str_folder*str_sampler*str_h*string(h)*str_csv 
    trace = Matrix(CSV.read(DIRIN, DataFrame; header=false))
    zz_results[i] = stein_metric(∇Ufull, trace, c, β, y, At, γ0)

    str_sampler = "bps_" # "sgld2_" "zz_" "bps_" "szz_"
    DIRIN = str_folder*str_sampler*str_h*string(h)*str_csv 
    trace = Matrix(CSV.read(DIRIN, DataFrame; header=false))
    bps_results[i] = stein_metric(∇Ufull, trace, c, β, y, At, γ0)
end

f1 = plot(title = "Stein Discrepancy", hh[4:end], sgld1_results[4:end], xaxis = :log, label = "sgld1")
plot!(f1, hh[4:end], sgld2_results[4:end], label = "sgld10")
plot!(f1, hh[4:end], sgld3_results[4:end], label = "sgld100")
plot!(f1, hh[4:end], bps_results[4:end], label = "bps")
plot!(f1, hh[4:end], zz_results[4:end], label = "zz")

res = [sgld1_results sgld2_results sgld3_results zz_results bps_results]
savefig(f1, "./scripts/logistic_regression/stein_distance/output_zoom.png")
CSV.write("./scripts/logistic_regression/stein_distance/output.csv", DataFrame(res, :auto), header = false)                
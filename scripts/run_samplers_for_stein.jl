# str_regression = "linear_regression"
str_regression = "logistic_regression"
# str_regression = "poisson_regression"

using Random, Distributions 
using StatsBase, LinearAlgebra, DataFrames, CSV
SRC = "./../src"
include(SRC*"/utilities.jl")
include(SRC*"/sg-zz.jl")
include(SRC*"/sg-bps.jl")
include(SRC*"/sg-szz.jl")
include(SRC*"/sgld.jl")


println("...defining gradient...")
include("./"*str_regression*"/grad.jl")


Random.seed!(1234);
p = 10
nobs = 100_000
sparsity = 0.5
println("...generating data...")
include("./"*str_regression*"/data_generation.jl")
hh = [1e-02, 5e-03, 1e-03, 5e-04, 1e-04, 5e-05, 1e-05, 5e-06, 1e-06]

Niter = 10^6 # number of iterations
thin = 500

function runall(A, y, Niter, thin, hh, str_regression)
    println("number of sample points: $(Niter ÷ thin)")
    γ0 = [1/10, 1/10]

    # OUTPUT STRINGS
    out = "./scripts/"*str_regression*"/stein_distance/posterior_samples/"
    str_sgld1 = "sgld1_" 
    str_sgld2 = "sgld2_"
    str_sgld3 = "sgld3_"
    str_h = "h_"
    str_zz =  "zz_"
    str_bps =  "bps_"
    str_csv = ".csv"
    str_data = "data"
    CSV.write(out*str_data*str_csv, DataFrame(hcat(A,y), :auto), header = false)    
    str_txt = out*"INFO.txt"
    file = open(str_txt, "w")
    nobs = size(A, 1)
    write(file, "N = $(nobs), p = $(p), Niter = $(Niter), gamma0 = $(γ0[1]), rel number of 0 values = $(sparsity)")
    At = A'
    nobs = size(At, 2)

    # FIND MODE
    mb = Int(round(0.01*length(y))) #minibatch size, e.g. 1% of full data
    Niter_opt = 10^6
    x0 = randn(p)
    @show norm(x0 - xtrue)
    cv = AdamCV(∇U!, ∇Ufull, x0, nobs, mb, Niter_opt, y, At)
    println("norm of gradient at control variates: $(norm(cv.∇x0))")
    @show cv.x0 - xtrue
    @show norm(cv.x0 - xtrue)
    # args...
    args = y, At, γ0

    sgld_max_h = 1/nobs
    pdmp_min_h = 1e-07
    for h in hh
        println("h = $(h)")
        if h <= sgld_max_h
                #SGLD 
                println("Running SGLD mb = 1")
                mb = 1
                x0 = copy(cv.x0) 
                verbose = false
                control_variates = cv
                trace = true
                out_sgld1 = sgld(∇Ucv!, x0, h, Niter, args...; 
                        thin = thin, 
                        cv = cv,
                        trace = trace,
                        verbose = verbose, 
                        minibatch = mb,
                        nobs = nobs) 
                CSV.write(out*str_sgld1*str_h*string(h)*str_csv, DataFrame(out_sgld1.trace, :auto), header = false)    

                #SGLD 
                println("Running SGLD mb = 10")
                mb = 10
                x0 = copy(cv.x0) 
                verbose = false
                control_variates = cv
                trace = true
                out_sgld2 = sgld(∇Ucv!, x0, h, Niter, args...; 
                        thin = thin, 
                        cv = cv,
                        trace = trace,
                        verbose = verbose, 
                        minibatch = mb,
                        nobs = nobs)
                CSV.write(out*str_sgld2*str_h*string(h)*str_csv, DataFrame(out_sgld2.trace, :auto), header = false)    
                #SGLD 
                println("Running SGLD mb = 100")
                mb = 100
                x0 = copy(cv.x0) 
                verbose = false
                control_variates = cv
                trace = true
                out_sgld3 = sgld(∇Ucv!, x0, h, Niter, args...; 
                        thin = thin, 
                        cv = cv,
                        trace = trace,
                        verbose = verbose, 
                        minibatch = mb,
                        nobs = nobs)  
                CSV.write(out*str_sgld3*str_h*string(h)*str_csv, DataFrame(out_sgld3.trace, :auto), header = false)   
        end
        if h >= pdmp_min_h           
                #Zig-Zag
                println("Running ZZ mb = 1")
                mb = 1
                x0 = copy(cv.x0)
                verbose = false
                control_variates = cv
                trace = true
                out_zz = sg_zz_model(∇Ucv!, x0, h, Niter, args...; 
                        thin = thin,
                        verbose = verbose, cv = control_variates, 
                        minibatch = mb, nobs = nobs);
                CSV.write(out*str_zz*str_h*string(h)*str_csv, DataFrame(out_zz.trace, :auto), header = false)                
                #BPS
                println("Running BPS mb = 1")
                mb = 1
                x0 = copy(cv.x0)
                verbose = false
                control_variates = cv
                trace = true
                out_bps = sg_bps_model(∇Ucv!, x0, h, Niter, args...; thin = thin, 
                        verbose = verbose, cv = control_variates,
                        minibatch = mb, nobs = nobs);   
                        
                CSV.write(out*str_bps*str_h*string(h)*str_csv, DataFrame(out_bps.trace, :auto), header = false)   
        end 
    end
end

runall(A, y, Niter, thin, hh, str_regression)
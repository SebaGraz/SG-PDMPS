# str_regression = "linear_regression"
str_regression = "linear_regression"
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
h = [5e-07]
pp = [10, 50, 100, 200, 300, 400, 500, 650, 800, 1000]
Niter = 100_000 # number of iterations
thin = 10
function runall(Niter, thin, hh, pp, str_regression)
    println("number of sample points: $(Niter ÷ thin)")
    γ0 = [1/10, 1/10]

    # OUTPUT STRINGS
    out = "./scripts/"*str_regression*"/scaling/posterior_samples/"
    str_sgld1 = "sgld1_" 
    str_sgld2 = "sgld2_"
    str_sgld3 = "sgld3_"
    str_h = "h_"
    str_zz =  "zz_"
    str_bps =  "bps_"
    str_csv = ".csv"
    str_data = "data_"
    str_folder = "./scripts/"*str_regression*"/scaling/data/"
    str_d = "d_"
    for p in pp 
        for h in  hh
                println("p = $(p), h = $h")
                DIROUTDATA = str_folder*str_data*str_d*string(p)*str_csv 
                data = Matrix(CSV.read(DIROUTDATA, DataFrame, header=false))
                A, y = Matrix(data[:,1:(end-1)]), Vector(data[:,end])
                At = A'
                nobs = size(At, 2)
                # FIND MODE
                mb = Int(round(0.01*length(y))) #minibatch size, e.g. 1% of full data
                # Niter_opt = 10^6
                # @show norm(x0 - xtrue)
                # cv = AdamCV(∇U!, ∇Ufull, x0, nobs, mb, Niter_opt, y, At)
                # println("norm of gradient at control variates: $(norm(cv.∇x0))")
                # @show cv.x0 - xtrue
                # @show norm(cv.x0 - xtrue)
                γ0 = 1/10
                #     xhat = inv(A'*A + I*γ0)A'y
                xhat = inv(A'*A)A'y 
                #     @show norm(∇Ufull(xhat, y, At, γ0))
                args_no_prior = y, At
                cv = CV(true, xhat, ∇Ufull(xhat, args_no_prior...))
                @show norm(cv.∇x0)
                # args...
                args = y, At, γ0
                x0 = cv.x0
                sgld_max_h = Inf
                pdmp_min_h = -Inf
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
                    CSV.write(out*str_sgld1*str_h*string(h)*str_d*string(p)*str_csv, DataFrame(out_sgld1.trace, :auto), header = false)    

                #     #SGLD 
                #     println("Running SGLD mb = 10")
                #     mb = 10
                #     x0 = copy(cv.x0) 
                #     verbose = false
                #     control_variates = cv
                #     trace = true
                #     out_sgld2 = sgld(∇Ucv!, x0, h, Niter, args...; 
                #             thin = thin, 
                #             cv = cv,
                #             trace = trace,
                #             verbose = verbose, 
                #             minibatch = mb,
                #             nobs = nobs)
                #     CSV.write(out*str_sgld2*str_h*string(h)*str_d*string(p)*str_csv, DataFrame(out_sgld2.trace, :auto), header = false)    
                    #SGLD 
                #     println("Running SGLD mb = 100")
                #     mb = 100
                #     x0 = copy(cv.x0) 
                #     verbose = false
                #     control_variates = cv
                #     trace = true
                #     out_sgld3 = sgld(∇Ucv!, x0, h, Niter, args...; 
                #             thin = thin, 
                #             cv = cv,
                #             trace = trace,
                #             verbose = verbose, 
                #             minibatch = mb,
                #             nobs = nobs)  
                #     CSV.write(out*str_sgld3*str_h*string(h)*str_d*string(p)*str_csv, DataFrame(out_sgld3.trace, :auto), header = false)   
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
                CSV.write(out*str_zz*str_h*string(h)*str_d*string(p)*str_csv, DataFrame(out_zz.trace, :auto), header = false)                
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
                        
                CSV.write(out*str_bps*str_h*string(h)*str_d*string(p)*str_csv, DataFrame(out_bps.trace, :auto), header = false)   
            end 
        end
    end
end
        

runall(Niter, thin, hh, pp, str_regression)
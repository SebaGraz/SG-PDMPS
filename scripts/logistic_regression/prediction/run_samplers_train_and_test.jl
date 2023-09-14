using Random, Distributions 
using StatsBase, LinearAlgebra, DataFrames, CSV
SRC = "./../../../src"
include(SRC*"/utilities.jl")
include(SRC*"/sg-zz.jl")
include(SRC*"/sg-bps.jl")
include(SRC*"/sg-szz.jl")
include(SRC*"/sgld.jl")

Random.seed!(1234);
sigmoid(x) = inv(one(x) + exp(-x))
lsigmoid(x) = -log(one(x) + exp(-x))
# gradient controbution from ith data point
function ∇Uj(x, j, y, At)
    At[:,j] *(sigmoid(dot(x,At[:, j])) - y[j])
end

# stochastic gradient 
function ∇U!(∇Ux, x, nobs, minibatch, y, At)
    jj = rand(1:nobs, minibatch)
    ∇Ux .= 0.0
    for j in jj
        ∇Ux .+= nobs/minibatch*(∇Uj(x, j, y, At)) 
    end
    ∇Ux
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


# gradient with control variates and prior
function ∇Ucv!(∇Ux, x, nobs, minibatch, cv::CV, y, At, γ0)
    xhat, ∇Uhat = cv.x0, cv.∇x0
    ∇Ux .=  ∇Uhat
    # prior
    ∇Ux[1] += γ0[1]*x[1]
    ∇Ux[2:end] += γ0[2]*x[2:end]
    jj = rand(1:nobs, minibatch)
    for j in jj
        ∇Ux .+= nobs/minibatch*(∇Uj(x, j, y, At) - ∇Uj(xhat, j, y, At))
    end
    return ∇Ux
end

function runall()
    p = 20
    nobs = 20_000
    println("generate data... p=$(p) and N = $(nobs)")
    Σ = zeros(p-1,p-1)
    for i in 1:p-1
        for j in 1:i
            Σ[i,j] = Σ[j,i] = (rand()*(0.8) - 0.4)^abs(i-j)
        end
    end

    A = ones(Float64,nobs,p)
    Xij = MultivariateNormal(Σ)
    for j in 1:nobs
        A[j,2:end]= rand(Xij)
    end
    At = A'

    σ2 = 1 #variance rescaled to avoid posterior contraction
    xtrue_ = randn(p)*sqrt(σ2) ##true parameter value
    sparsity = 0.5
    jj = sample(1:p, Int(p*sparsity), replace = false)
    xtrue = xtrue_
    xtrue[jj] .= 0.0 

    # simulate responses
    y = (rand(nobs) .< sigmoid.(A*xtrue))



    Niter = 500_000 # number of iterations
    thin = 100
    println("number of sample points: $(Niter ÷ thin)")
    γ0 = [1/10, 1/10]


    # Build the training and testing data set
    train_ratio = 0.9 # We create the train and test sets with 90% and 10% of the data
    size_train = Int64(round(size(A,1) * train_ratio));

    # OUTPUT STRINGS
    out = "./scripts/logistic_regression/prediction/posterior_samples/"
    str_sgld1 = "sgld1_" 
    str_sgld2 = "sgld2_"
    str_sgld3 = "sgld3_"
    str_h = "h_"
    str_zz =  "zz_"
    str_bps =  "bps_"
    str_szz = "szz_"
    str_iter = "_iter_"
    str_csv = ".csv"
    str_data = "test_data"

    hh = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    for iter in 1:10
        println("loop = $(iter)")
        permute = StatsBase.sample(1:size(A,1),size(A,1),replace=false);
        X_train, y_train = A[permute[1:size_train], : ], y[ permute[1:size_train] ];
        X_test, y_test = A[permute[(size_train+1):end], : ], y[ permute[(size_train+1):end] ];
        CSV.write(out*str_data*str_iter*string(iter)*str_csv, DataFrame(hcat(X_test,y_test), :auto), header = false)    

        Xt_train = X_train'
        nobs = size(Xt_train, 2)

        # FIND MODE
        mb = Int(round(0.01*length(y_train))) #minibatch size, e.g. 1% of full data
        Niter_opt = 10^6
        x0 = randn(p)
        @show norm(x0 - xtrue)
        cv = AdamCV(∇U!, ∇Ufull, x0, nobs, mb, Niter_opt, y_train, Xt_train)
        println("norm of gradient at control variates: $(norm(cv.∇x0))")
        @show cv.x0 - xtrue
        @show norm(cv.x0 - xtrue)
        ypred = sigmoid.(X_train*cv.x0) .> 0.5
        sum(y_train .== ypred)/length(y_train)
        println("predictive power: $(sum(y_train .== ypred)/length(y_train))")
        # args...
        args = y_train, Xt_train, γ0

        for h in hh
            println("h = $(h)")
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
            CSV.write(out*str_sgld1*str_h*string(h)*str_iter*string(iter)*str_csv, DataFrame(out_sgld1.trace, :auto), header = false)    

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
            CSV.write(out*str_sgld2*str_h*string(h)*str_iter*string(iter)*str_csv, DataFrame(out_sgld2.trace, :auto), header = false)    
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
            CSV.write(out*str_sgld3*str_h*string(h)*str_iter*string(iter)*str_csv, DataFrame(out_sgld3.trace, :auto), header = false)            
            #------------------------------------------
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
            CSV.write(out*str_zz*str_h*string(h)*str_iter*string(iter)*str_csv, DataFrame(out_zz.trace, :auto), header = false)                
            #------------------------------------------
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
                        
            CSV.write(out*str_bps*str_h*string(h)*str_iter*string(iter)*str_csv, DataFrame(out_bps.trace, :auto), header = false)    
            #------------------------------------------
            #Sticky Zig-Zag
            println("Running sticky ZZ mb = 1")
            mb = 1
            x0 = copy(cv.x0)
            # κ = sparsity/(1-sparsity)
            κs = ones(length(cv.x0)).*5.0
            verbose = false
            control_variates = cv
            trace = true
            out_szz = sg_sticky_zz_model(∇Ucv!, κs, x0, h, Niter, args...; thin = thin, 
                    verbose = verbose, cv = control_variates, 
                    minibatch = mb, nobs = nobs);  
            CSV.write(out*str_szz*str_h*string(h)*str_iter*string(iter)*str_csv, DataFrame(out_szz.trace, :auto), header = false)              
        end
    end
end

runall()
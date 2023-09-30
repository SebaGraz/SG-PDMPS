# run 
# if dataset exists, do not run toy example, only plot variance and traces
using Random, LinearAlgebra, CSV, DataFrames, Distributions, StatsBase
SRC = "./../src"
include(SRC*"/utilities.jl")
include(SRC*"/sg-zz.jl")
include(SRC*"/sg-bps.jl")
include(SRC*"/sg-szz.jl")
include(SRC*"/sgld.jl")
include(SRC*"/utilities.jl")
str_app = "toy"
str_data = "./scripts/"*str_app*"/data/"
str_d = "d_"
str_csv = ".csv"
out = "./scripts/"*str_app*"/posterior_samplers/"
str_h = "h_"
str_csv = ".csv"
str_sgld1 = "sgld1_" 
str_sgld2 = "sgld2_"
str_sgld3 = "sgld3_"
str_zz =  "zz_"
str_bps =  "bps_"

nobs = 10^6
p = 5
# sparsity = 0.5
# println("p=$(p) and N = $(nobs), sparsity = $(sparsity)")
Random.seed!(1234)
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
# jj = sample(1:p, Int(p*sparsity), replace = false)
xtrue = xtrue_
xtrue[1] = 0.0 
# simulate responses
y = A*xtrue + randn(nobs)     
DIROUTDATA = str_data*str_d*string(p)*str_csv 
CSV.write(DIROUTDATA, DataFrame(hcat(A,y), :auto), header = false) 




# gradient controbution from ith data point
function ∇Uj(x, j, y, At)
    -At[:,j] *(y[j] - dot(x,At[:, j]))
end


# args = y, At,
function ∇Ufull(x, y, At, γ0)
    nobs = size(At, 2)
    ∇Ux = γ0.*x
    for j in 1:nobs
        ∇Ux .+= ∇Uj(x, j, y, At)
    end
    return ∇Ux 
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


# stochastic gradient 
function ∇U!(∇Ux, x, nobs, minibatch, y, At)
    jj = rand(1:nobs, minibatch)
    ∇Ux .= 0.0
    for j in jj
        ∇Ux .+= nobs/minibatch*(∇Uj(x, j, y, At)) 
    end
    ∇Ux
end

# gradient with control variates and prior
function ∇Ucv!(∇Ux, x, nobs, minibatch, cv::CV, y, At, γ0)
    xhat, ∇Uhat = cv.x0, cv.∇x0
    ∇Ux .=  ∇Uhat
    # prior
    ∇Ux += γ0*x
    jj = rand(1:nobs, minibatch)
    for j in jj
        ∇Ux .+= nobs/minibatch*(∇Uj(x, j, y, At) - ∇Uj(xhat, j, y, At))
    end
    return ∇Ux
end


# CV ls estimate used for centering negative gradient
β̂ = (A'A)\A'y
args_no_prior = y, A' 
cv = CV(true, β̂, ∇Ufull(β̂, args_no_prior...))
@show norm(cv.∇x0)
x0 = cv.x0
hh = [collect(0.05:-0.01:0.02)..., collect(0.01:-0.002:0.001)...,collect(0.001:-0.0002:0.0002)..., 0.0001]
# hh = [0.1,]
sgld_max_h = Inf
pdmp_min_h = -Inf
thin = 100
Niter = 10^5
γ0 = 1/10
args = y, A', γ0
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



# TODO PLOT VARIANCES 

# TODO PLOT TRACES
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
out = "./scripts/"*str_app*"/mse/"
str_h = "h_"
str_csv = ".csv"
str_sgld1 = "sgld1_" 
str_zz =  "zz_"
str_bps =  "bps_"
str_iter = "_iter_"

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
# A = ones(Float64,nobs,p)
# G= I(p-1)*1.0 ## covariates are Gamma %*% z where z is standard normal; first covariate an intercept

# for j in 1:nobs
#     A[j,2:end]=G * randn(p-1)
# end
σ2 = nobs/100 #variance rescaled to avoid posterior contraction
xtrue = [3.0*randn()*sqrt(2) for _ in 1:p] ##true parameter value
xtrue[1] = 0.0
#simulate responses
y=zeros(Float64,nobs)
for j in 1:nobs
    y[j]=randn()*sqrt(σ2)+ dot(A[j,1:end], xtrue)
end
At = A'
# simulate responses
# γ0 = 1/10 
# @show inv((A'A + I*γ0))[1,1]


DIROUTDATA = str_data*str_d*string(p)*str_csv 
CSV.write(DIROUTDATA, DataFrame(hcat(A,y), :auto), header = false) 



# CV ls estimate used for centering negative gradient
β̂ = (A'A)\A'y
∇Uβ̂ = zeros(p)
for j in 1:nobs  
    ∇Uβ̂ += (dot(A[j,1:end], β̂) - y[j])*A[j,1:end]/σ2 
end
cv = CV(true, β̂, ∇Uβ̂)


# function ∇U!(∇Ux, β, j, y, X, σ2, β̂, ∇U0)
#     nobs = size(X,1)
#     ∇Ux .= nobs*dot(X[j,1:end], β - β̂)*X[j, 1:end]/σ2 + ∇U0
#     return ∇Ux
# end

function ∇Ucv!(∇Ux, x, nobs, minibatch, cv::CV, y, A, γ0, σ2)
    β̂, ∇U0 = cv.x0, cv.∇x0
    jj = rand(1:nobs, minibatch)
    ∇Ux .= γ0*x + ∇U0
    ∇Ux .+= nobs/minibatch*sum([dot(A[j,1:end], x - β̂)*A[j, 1:end] for j in jj])/σ2 
    return ∇Ux
end
 

@show norm(cv.∇x0)
x0 = cv.x0
# hh = [collect(0.05:-0.01:0.02)..., collect(0.01:-0.002:0.001)...,collect(0.001:-0.0002:0.0002)..., 0.0001]
hh = [0.1, collect(0.05:-0.01:0.02)..., collect(0.01:-0.002:0.001)...,collect(0.001:-0.0002:0.0001)..., collect(0.0001:-0.00002:0.00002)..., 0.00001]# hh = [0.1,]
sgld_max_h = Inf
pdmp_min_h = -Inf
thin = 1000
Niter = 10^6
γ0 = 1/10
args = y, A, γ0, σ2
Random.seed!(1)
for h in hh
    for iter in 1:100
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
            CSV.write(out*str_sgld1*str_h*string(h)*str_iter*string(iter)*str_csv, DataFrame(out_sgld1.trace, :auto), header = false)       
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
            CSV.write(out*str_zz*str_h*string(h)*str_iter*string(iter)*str_csv, DataFrame(out_zz.trace, :auto), header = false)                
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
    end  
    end 
end

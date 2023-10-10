###### COMPARISONS LINEAR REGRESSIONS: 
# 1) sg-zz
# 2) sg-bps
# 3) sgld 
# 4) sg-bps second order
# for h = 0.001:0.005:0.1
# metric: wost var(x_i), i =1,2,…,d
# ground truth zz pdmp
using StatsBase, Plots, LinearAlgebra, DataFrames, CSV
include("./../../src/sg-zz-o1.jl")
include("./../../src/sg-bps-o1.jl")
include("./../../src/sgld.jl")
include("./../../src/utilities.jl")
function runall!(df)
    println("generating data")
    Random.seed!(0)
    p = 5
    n = 1000000
    ##simulate covariates
    Gamma= I(p-1)*1.0 ## covariates are Gamma %*% z where z is standard normal; first covariate an intercept
    X = ones(Float64,n,p)
    for j in 1:n
        X[j,2:end]=Gamma * randn(p-1)
    end
    σ2 = n/100 #variance rescaled to avoid posterior contraction
    β0 = [3.0*randn()*sqrt(2) for _ in 1:p] ##true parameter value
    ##simulate responses
    y=zeros(Float64,n)
    for j in 1:n
        y[j]=randn()*sqrt(σ2)+ dot(X[j,1:end], β0)
    end
    ##CV -- ls estimate used for centering negative gradient
    β̂ = (X'X)\X'y

    # gradient at β̂
    ∇U0 = zeros(Float64, p) 
    for j in 1:n  
        ∇U0 += (dot(X[j,1:end], β̂) - y[j])*X[j,1:end]/σ2 
    end
    normsq(x) = dot(x,x)
    
    function ∇U!(∇Ux, β, j, y, X, σ2, β̂, ∇U0)
        nobs = size(X,1)
        ∇Ux .= nobs*dot(X[j,1:end], β - β̂)*X[j, 1:end]/σ2 + ∇U0
        return ∇Ux
    end
    function ∇Usgld!(∇Ux, β, jj, y, X, σ2, β̂, ∇U0)
        nobs = size(X,1)
        nminibatch = length(jj)
        ∇Ux .= nobs/nminibatch*sum([dot(X[j,1:end], β - β̂)*X[j, 1:end] for j in jj])/σ2 + ∇U0
        return ∇Ux
    end
    x0 = β0 
    N = 5*10^6
    hh = [0.1, collect(0.05:-0.01:0.02)..., collect(0.01:-0.002:0.001)...,collect(0.001:-0.0002:0.0002)..., 0.0001]
    # NN = fill(2*10^4, length(hh))
    # NN[end-10:end] .= 2*10^6
    max_grad = Inf
    for h in hh
        println("N = $(N), max grad = $(max_grad), h = $(h)")
        println(" N = $(N), BPS, x0 = $(x0)")
        tp, xx, grad_eval, h, Npost, fixed_grad_eval = sg_bps(∇U!, x0, h, N, y, X, σ2, β̂, ∇U0; λref = 1.0, nobs = n, max_grad_ev = max_grad)
        res = mshape(xx)
        sigmas = var(res[:, 1:5], dims = 1)[:]
        push!(df, [tp, sigmas[1:5]..., grad_eval, h, Npost, fixed_grad_eval])
        println(" N = $(N), ZZ,  x0 = $(x0)")
        tp, xx, grad_eval, h, Npost, fixed_grad_eval = sg_zz(∇U!, x0, h, N, y, X, σ2, β̂, ∇U0; λref = 0.0, nobs = n, max_grad_ev = max_grad)
        res = mshape(xx)
        sigmas = var(res[:, 1:5], dims = 1)[:]
        push!(df, [tp, sigmas[1:5]..., grad_eval, h, Npost, fixed_grad_eval])
        batch_size = 1
        println(" N = $(N), SGLD,  x0 = $(x0)")
        tp, xx, grad_eval, h, Npost, fixed_grad_eval  = sgld(∇Usgld!, x0, h, N, batch_size, y, X, σ2, β̂, ∇U0; nobs = n, max_grad_ev = max_grad)
        res = mshape(xx)
        sigmas = var(res[:, 1:5], dims = 1)[:]
        push!(df, [tp, sigmas[1:5]..., grad_eval, h, Npost, fixed_grad_eval])    
    end
    return df, X, σ2
end


io = "./scripts/new_linear_regression/output/linear_regression.csv"
if isfile(io)
    df = DataFrame(CSV.File(io))
else
    df = DataFrame("tp" => String[],
                    "var1" => Float64[],
                    "var2" => Float64[],
                    "var3" => Float64[],
                    "var4" => Float64[],
                    "var5" => Float64[],
                    # "var6" => Float64[],
                    # "var7" => Float64[],
                    # "var8" => Float64[],
                    # "var9" => Float64[],
                    # "var10" => Float64[],
                    "grad eval" => Int64[],
                    "h" => Float64[],
                    "N" => Int64[], 
                    "fixed_grad_eval" => Bool[])
    df, X, σ2 = runall!(df) 
    push!(df, ["true",diag(inv(X'X)*σ2)[1:5]..., 0, 0.0, 0, 0])
    CSV.write(io, df)
end

tvar = Matrix(df[df.tp .== "TRUE",[:var1]])[:]

tpp = ["sg-zz", "sg-bps", "sgld"]
# tpp = ["sg-zz", "sg-bps", "sgld"]
f1 = plot(xlabel = "discretization step", ylabel = "var x1",  xflip = true, title = "comparison linear regression")
for tp in tpp
    # filter(row -> rf1ow.tp == tp  ,  df)
    x = Matrix(df[(df.tp .== tp) .& (df.fixed_grad_eval .== false),[:h]])[:]
    y = Matrix(df[(df.tp .== tp) .& (df.fixed_grad_eval .== false),[:var1]])[:]
    plot!(f1, x, y,  xaxis=:log, label = tp)
end
f1
hline!(f1, tvar,linestyle = :dot, label = "true var")
savefig(f1, "./scripts/new_linear_regression/output/fig1.pdf")
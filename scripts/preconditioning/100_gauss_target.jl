include("./../../src/utilities.jl")
using LinearAlgebra, Distributions
p = 10
Pr = Diagonal(ones(p))
Pr[1,1] = 10
N = 50
X = randn(N,p)*Pr
β = randn(p)
y = X*β + randn(N)



γ0 = 1/100
# gradient controbution from ith data point
function ∇Uj(x, j, y, At)
    -At[:,j] *(y[j] - dot(x,At[:, j]))
end


# gradient with control variates and prior
function ∇Ucv!(∇Ux, x, nobs, minibatch, xhat, y, At, γ0)
    ∇Ux .=  0.0
    # prior
    ∇Ux .+= γ0*x
    jj = rand(1:nobs, minibatch)
    for j in jj
        ∇Ux .+= nobs/minibatch*(∇Uj(x, j, y, At) - ∇Uj(xhat, j, y, At))
    end
    return ∇Ux
end

function ∇Ufull(x, y, At)
    nobs = size(At, 2)
    ∇Ux = zeros(length(x))
    for j in 1:nobs
        ∇Ux .+= ∇Uj(x, j, y, At)
    end
    return ∇Ux 
end

# CONTROL VARIATES, POSTERIOR MODE
βhat = inv(X'X)X'y
Λpos = X'X + I(p)*γ0
μpos = inv(Λpos)*(X'*X*βhat)
println("check gradient at mode")
∇Ufull(μpos, y, X') + μpos*γ0

# POSTERIOR SEE https://en.wikipedia.org/wiki/Bayesian_linear_regression


γ(i::Integer, κ::Float64) = i^(-κ)

function adapt_sgld(∇U!, x0, h::Float64, N::Int64, κ, args...;
    thin = 100, 
    max_grad_ev = Inf, 
    verbose = false,
    trace = true,
    minibatch,
    nobs)

    if verbose
        println("Info:\n 
          sampler: sgld,\n
          step size: $h,\n
          batchsize: $(minibatch),\n
          N obs: $(nobs),\n
          control variates: $(cv.cv),\n
          ")
      end
    out = Output("sgld")
    x = copy(x0)
    p = length(x)
    ∇Ux  = zeros(p)
    grad_ev = 0
    j = 1
    μp = copy(x0)
    σ2 = ones(p)
    hadapt = [(copy(μp), copy(σ2)),]
    while(j < N)
        if grad_ev > max_grad_ev
          out.max_grad = true
          break
        end
        grad_ev += minibatch
        ∇Ux = ∇U!(∇Ux, x,  nobs, minibatch, args...) 
        x -= h/2*(∇Ux) + sqrt(h)*randn(p)
        μp = μp + γ(j+1, κ)*(x - μp)
        σ2 = σ2 + γ(j+1, κ)*((x - μp).^2 - σ2) 
        push!(hadapt,(copy(μp), copy(σ2)))
        if j % thin == 0 
            if trace 
                push!(out.trace, copy(x))
              end
              push!(out.grad_eval, grad_ev)
              push!(out.iter, j) 
            # println("logdensity = $(logp(x))")
        end
        if verbose && j*100/N % 10 == 0
            println("progress: $(j/N*100)%...")
        end
        j += 1
    end
    out, hadapt
end

h = 1e-4
Niter = 10^5 
κ = 0.9
args = μpos, y, X', γ0
x0 = randn(p)
res, adapt = adapt_sgld(∇Ucv!, x0, h, Niter, κ, args...; nobs = N, minibatch = 1)

using Plots    
plot()

plot(getindex.(res.trace,1))    
hline!([μpos[1] - 2*sqrt(inv(Λpos)[1]),μpos[1] + 2*sqrt(inv(Λpos)[1])])

var(getindex.(res.trace,2))
mean(getindex.(getindex.(adapt, 2),2))



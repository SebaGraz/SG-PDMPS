using Distributions, LinearAlgebra, StatsBase, ForwardDiff
include("../src/utilities.jl")
println("linear regression...")
p = 5
N = 10_000
A = rand(N, p)
β = randn(p)
y = A*β + randn(N)
βols = inv(A'*A)*A'*y

function ∇U!(∇Uβ, β, nobs, mb, A, y)
    ∇Uβ .= 0.0
    jj = sample(1:nobs, mb)
    for j in jj
        ∇Uβ .-=   (nobs/mb)*(y[j] - dot(A[j,:], β))*A[j,:]
    end
    return ∇Uβ
end



function ∇Ufull(β, A, y)
    ∇Uβ = zeros(length(β))
    for j in 1:size(A,1)
        ∇Uβ .-=   (y[j] - dot(A[j,:], β)).*A[j,:]
    end
    return ∇Uβ
end

cv = AdamCV(∇U!, ∇Ufull, rand(p), size(A,1), 10, 10^6, A, y)

[cv.x0 βols]
cv.∇x0
∇Ufull(cv.x0, A, y)


println("logistic regression")
sigmoid(x) = inv(one(x) + exp(-x))
lsigmoid(x) = -log(one(x) + exp(-x))
lsigmmoid2(x) = log(one(x) - sigmoid(x))

function ∇Uj(x, j, y, At)
    At[:,j] *(sigmoid(dot(x,At[:, j])) - y[j])
end


# stochastic gradient 
# no prior
# no cv
# args = y, At, γ0, x0, ∇U0
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



p = 10
nobs = 10_000
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
y = (rand(nobs) .< sigmoid.(A*xtrue))


# check gradient
x0 = rand(20)
function ϕ(x, y, A) 
    res = 0.0
    for i in 1:size(A,1)
        res += -y[i]*log(1/(1 + exp(-dot(A[i,:], x)))) - (1-y[i])*log(1 - 1/(1 + exp(-dot(A[i,:], x))))
    end
    res
end
[ForwardDiff.gradient(x -> ϕ(x, y_test, X), x0)[1:10] ∇Ufull(x0, y_test, Xt)[1:10]]
X = Xt'

mb = Int(round(0.01*length(y))) #minibatch size, e.g. 1% of full data
Niter_opt = 10^6
x0 = randn(p)*10
# x0 = xtrue
# U!, ∇Ufull, θ0, nobs, minibatch, epochs, args...;
@show x0 - xtrue
@show norm(x0 - xtrue)
nobs = size(At, 2)
@show nobs
cv = AdamCV(∇U!, ∇Ufull, x0, nobs, mb, Niter_opt, y, At)
println("norm of gradient at control variates: $(norm(cv.∇x0))")
@show cv.x0 - xtrue
@show norm(cv.x0 - xtrue)
@show cv.∇x0
ypred = sigmoid.(A*cv.x0) .> 0.5
sum(y .== ypred)/length(y)
println("predictive power: $(sum(y .== ypred)/length(y))")



println("after normalization...")
dt_x = fit(ZScoreTransform, A, dims=2)
X = StatsBase.transform(dt_x,A)
Xt = X'
x0 = randn(p)*10
# x0 = xtrue
# U!, ∇Ufull, θ0, nobs, minibatch, epochs, args...;
cvn = AdamCV(∇U!, ∇Ufull, x0, nobs, mb, Niter_opt, y, Xt)
println("norm of gradient at control variates: $(norm(cvn.∇x0))")

ypred = sigmoid.(X*cvn.x0) .> 0.5
println("predictive power: $(sum(y .== ypred)/length(y))")
println("WARNING: normalising does not seem to improve the predictive power and help with the control variates")




struct Output{T1}
    sampler::String
    trace::Vector{T1}
    mse::Vector{Float64}
    grad_eval::Vector{Int64}
    iter::Vector{Int64}
    max_grad::Bool
end

Output(sampler::String) = Output(sampler, Vector{Vector{Float64}}(), Vector{Float64}(), Vector{Int64}(), Vector{Int64}(), false)


struct TestData{T1, T2}
    in::Vector{T1}
    out::Vector{T2}
end

struct Test{T}
    test::Bool
    data::TestData
    f::T
end

function Test(test)
    if test === true
        error("please provide test data ::TestData as second argument and a predictive function f(x,params) = ŷ as third argument")
    else
        return Test(false, TestData(Vector{Float64}(),Vector{Float64}()), ())
    end
end


function mse(t::Test, params::Vector{Float64})
    if t.test == false 
        error("cannot test model")
    else
        x = t.data.in
        y = t.data.out
        ŷ = [t.f(x[i], params) for i in 1:length(x)]
        return sum(abs2, y - ŷ)
    end
end

struct CV
    cv::Bool
    x0::Vector{Float64}
    ∇x0::Vector{Float64}
end

function CV(cv) 
    if cv == true
        error("please provide location and gradient at the location as second and third argument")
    else
        return(CV(false, Vector{Float64}(), Vector{Float64}()))
    end
end

CV(x0::Vector{Float64}, ∇x0::Vector{Float64}) = CV(true, x0, ∇x0)


function mshape(res)
    res2 = Matrix{Float64}(undef, length(res), length(res[1]))
    for i in eachindex(res)
        res2[i,:] = res[i]
    end
    res2
end


function vshape(res)
    [res[:, i] for i in 1:size(res,2)]
end

# ∇U! has to be an unbiased estimator
function AdamCV(∇U!, ∇Ufull, θ0, nobs, minibatch, epochs, args...; eps = 1e-8, α = 0.001, β1 = 0.9, β2 = 0.999)
    println("optimization routine for finding local mode")
    t = 0
    mt = zeros(length(θ0))
    vt = zeros(length(θ0))
    θ = copy(θ0)
    gt = zeros(length(θ0))
    while t < epochs 
        t += 1
        t % (epochs÷10) == 0 && println("...$(t*100/epochs)%")
        gt = ∇U!(gt, θ, nobs, minibatch, args...)
        mt = β1*mt + (1-β1)*gt
        vt = β2*vt + (1-β2)gt.^2
        mthat = mt/(1-β1^t)
        vthat = vt/(1-β2^t)
        θ -= α*mthat./(sqrt.(vthat) .+ eps) 
    end
    return CV(true, θ, ∇Ufull(θ, args...))
end

reshapetov(xx) = [xx[:, i] for i in 1:size(xx,2)]

norm2(x) = dot(x,x)

function stein_k(j, θ1, θ2, ∇θ1, ∇θ2, c::Float64, β::Float64)
    res = ∇θ1[j]*∇θ2[j]*(c^2 + norm2(θ1 - θ2))^β + 
        2*β*(c^2 + norm2(θ1 - θ2))^(β-1)*(θ1[j] - θ2[j])*(∇θ2[j] - ∇θ1[j]) - 
        4*(β)*(β-1)*(c^2 + norm2(θ1 - θ2))^(β-2)*(θ1[j] - θ2[j])^2 - 
        2*β*(c^2 + norm2(θ1 - θ2))^(β-1)
    return res 
end


function stein_kernel(∇U, xx, c::Float64, β::Float64, args...)
    ∇Uxx = [∇U(x, args...) for x in xx]
    res = 0.0
    n = length(xx)
    p = length(xx[1])
    for j in 1:p
        resi = 0.0
        for i in eachindex(xx)
            for k in 1:i
                m = i == k ? 1.0 : 2.0 
                resi += m*stein_k(j, xx[i], xx[k], ∇Uxx[i], ∇Uxx[k], c, β)
            end
        end
        res += sqrt(resi)
    end
    res/n
end



   
function evaluation(testdata, param)
    X, y = testdata
    y[y.==0].=-1
    d = size(param, 2)
    coff = zeros(length(y),d)
    prob = zeros(length(y),d)
    for i in eachindex(y)
        for j in 1:d
            coff[i,j] = y[i]*(-1*dot(X[i,:],param[:,j]))
            prob[i,j] = 1 ./(1 .+ exp(coff[i,j]))
        end
    end
    prob = mean(prob; dims=2)
    acc = mean(prob .> 0.5)
    llh = -mean(log.(prob))
    return (acc, llh)
end


using StatsBase
function autocorr(xx, lag)
    N = size(xx, 2)
    p = size(xx, 1)
    res = zeros(p)
    μ = mean(xx, dims = 2)
    if (any(μ .== NaN))
        return res
    else
        for i in 1:N-lag    
            res .+= (xx[:,i] .- μ).*(xx[:,i + lag] .- μ)/(N-lag)
        end
    end
    res
end


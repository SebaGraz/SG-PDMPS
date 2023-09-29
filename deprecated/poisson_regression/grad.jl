# x = [μ, η]
# args  = Y, σμ, ση
function U(x, Y, σμ, ση)
    I = size(Y,1)
    n = size(Y,2)
    μ = x[1]
    η = x[2:end]
    ll = 0.0
    for i in 1:I
        ysum = 0.0
        for j in 1:n
            ysum += Y[i,j]
        end
        ll += n*exp(η[i]) - η[i]*ysum
    end  
    return 0.5*μ^2/σμ^2 + sum([0.5*(ηi - μ)^2/ση^2 for ηi in η]) + ll
end

function ∇Ufull(x, Y, σμ, ση)
    d = length(x)
    n = size(Y,2)
    μ = x[1]
    η = x[2:end]
    grad = zeros(d)
    grad[1] = μ/σμ^2 - sum([(ηi - μ)/ση^2 for ηi in η])   
    for k in eachindex(η)
        ysum = 0.0
        for j in 1:n
            ysum += Y[k,j]
        end
        grad[k+1] = (η[k]- μ)/ση^2 +  n*exp(η[k]) - ysum
    end  
    return grad
end

function ∇U!(grad, x, nobs, minibatch, Y, σμ, ση)
    d = length(x)
    n = size(Y,2)
    μ = x[1]
    η = x[2:end]
    grad .= 0.0 
    grad[1] = μ/σμ^2 - sum([(ηi - μ)/ση^2 for ηi in η])   
    for k in eachindex(η)
        ysum = 0.0
        jj = rand(1:nobs, minibatch)
        for j in jj
            ysum += Y[k,j]*n/minibatch
        end
        grad[k+1] = (η[k]- μ)/ση^2 +  n*exp(η[k]) - ysum
    end  
    return grad
end

function ∇Ucv!(∇Ux, x, nobs, minibatch, cv::CV, Y, σμ, ση)
    ∇U!(∇Ux, x, nobs, minibatch, Y, σμ, ση)
end
sigmoid(x) = inv(one(x) + exp(-x))
lsigmoid(x) = -log(one(x) + exp(-x))
# gradient controbution from ith data point
function ∇Uj(x, j, y, At)
    At[:,j] *(sigmoid(dot(x,At[:, j])) - y[j])
end

# args = y, At,
function ∇Ufull(x, y, At, γ0)
    nobs = size(At, 2)
    ∇Ux = γ0*x
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
    ∇Ux[1] += γ0[1]*x[1]
    ∇Ux[2:end] += γ0[2]*x[2:end]
    jj = rand(1:nobs, minibatch)
    for j in jj
        ∇Ux .+= nobs/minibatch*(∇Uj(x, j, y, At) - ∇Uj(xhat, j, y, At))
    end
    return ∇Ux
end




function ΔU(x, y, At, γ0)
    d = length(x)
    ΔUx0 = zeros(d,d) + I(d).*γ0
    ΔUx0[1,1] = γ0 
    nobs = size(At, 2) 
    for k1 in 1:d
        for k2 in 1:k1
            for j in 1:nobs
                ΔUx0[k1,k2] += At[k2,j]*At[k1,j]*exp(-dot(x, At[:,j]))/(1+exp(-dot(x, At[:,j])))^2
            end
            if k1 != k2
                ΔUx0[k2,k1] = ΔUx0[k1,k2] 
            end
        end
    end
    return ΔUx0
end

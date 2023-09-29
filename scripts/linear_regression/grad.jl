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
    ∇Ux[1] += γ0*x[1]
    ∇Ux[2:end] += γ0*x[2:end]
    jj = rand(1:nobs, minibatch)
    for j in jj
        ∇Ux .+= nobs/minibatch*(∇Uj(x, j, y, At) - ∇Uj(xhat, j, y, At))
    end
    return ∇Ux
end

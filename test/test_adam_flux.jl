
# ∇U! has to be an unbiased estimator
function AdamCV((x, y, model, loss, mb), epochs; eps = 1e-8, α = 0.001, β1 = 0.9, β2 = 0.999)
    println("optimization routine for finding local mode")
    params, reconstruct = Flux.destructure(model)
    p = length(params)
    t = 0
    mt = zeros(Float32, p)
    vt = zeros(Float32, p)
    nobs = size(x, 2)
    while t < epochs 
        i = rand(1:61, mb)
        input, output = x[:, i], y[:,i]
        grads = Flux.gradient(model) do m 
                loss(m, input, output)
            end
        ∇Uθ, _ = Flux.destructure(grads)
        θ, _ = Flux.destructure(model)
        t += 1
        t % (epochs÷10) == 0 && println("...$(t*100/epochs)%")
        gt = ∇Uθ*nobs/mb 
        mt = β1*mt + (1-β1)*gt
        vt = β2*vt + (1-β2)gt.^2
        mthat = mt/(1-β1^t)
        vthat = vt/(1-β2^t)
        θ = θ -  α*mthat./(sqrt.(vthat) .+ eps)
        model = reconstruct(θ)
    end
    model
end


function StochGD((x, y, model, loss,), epochs; α = 0.001)
    println("optimization routine for finding local mode")
    params, reconstruct = Flux.destructure(model)
    p = length(params)
    t = 0
    nobs = size(x, 2)
    while t < epochs 
        i = rand(1:61)
        input, output = x[:, i], y[:,i]
        grads = Flux.gradient(model) do m 
                loss(m, input, output)
            end
        gt, _ = Flux.destructure(grads)
        θ, _ = Flux.destructure(model)
        
        t += 1
        t % (epochs÷10) == 0 && println("...$(t*100/epochs)%")
        θ = θ - α*gt*nobs
        model = reconstruct(θ)
    end
    model
end

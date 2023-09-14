
normsq(x) = dot(x,x)

function bounce!(∇Ux, v)
    v .-= (2*dot(∇Ux, v)/normsq(∇Ux))*∇Ux
    v
end


function eventbps(tmax, λ, λref)
    τ1 = λ <= 0 ? Inf : -log(rand())/λ
    τ2 = -log(rand())/λref
    if tmax < min(τ1, τ2)
      return 0, tmax
    elseif τ1 < τ2
      return 1, τ1
    else
      return 2, τ2
    end
end





function sg_bps_model(∇U!, x0, h::Float64, N::Int64, args...; 
        thin = 100, λref = 0.1, 
        max_grad_ev = Inf, 
        cv = CV(false),
        test = Test(false),
        verbose = false,
        trace = true, 
        minibatch = 10, 
        nobs)
    if verbose
      println("Info:\n 
        sampler: sg-bps,\n
        step size: $h,\n
        λref: $(λref),\n
        batchsize: $(minibatch),\n
        N obs: $(nobs),\n
        control variates: $(cv.cv),\n
        ")
    end
    x = copy(x0)
    p = length(x)
    out = Output("sg-bps")

    v = randn(p)
    # x = vi_current[spl]
    ∇Ux  = zeros(p)
    ∇Ux = ∇U!(∇Ux, x,  nobs, minibatch, cv, args...) 
    i = 1
    t = 0.0
    dt = h
    flips = 0
    grad_ev = minibatch
    while(i < N)
        if grad_ev > max_grad_ev
          out.max_grad = true
          break
        end
        ev, τ  = eventbps(dt, dot(∇Ux, v), λref)
        t += τ
        x .+= v*τ  
        if ev == 1
            dt = dt - τ
            # right now I am not able to reuse the same jth observation 
            # ∇Ux = ∇U!(∇Ux, x, j, args...)
            v = bounce!(∇Ux, v)
            flips += 1
        elseif ev == 2
            dt = dt - τ
            v = randn(p)
            flips += 1  
        else
            i += 1
            dt=h
            if i % thin == 0
                if trace 
                  push!(out.trace, copy(x))
                end
                if test.test
                  push!(out.mse, mse(test, x))
                end
                push!(out.grad_eval, grad_ev)
                push!(out.iter, i)
                # println("logdensity = $(logp(x))")
            end 
            if verbose && i*100/N % 10 == 0
              println("progress: $(i/N)...%")
            end
        end
        # update gradient
        grad_ev += minibatch
        ∇Ux = ∇U!(∇Ux, x,  nobs, minibatch, cv, args...) 
    end
   out
end



function hitting_time(x,v)
    if x*v>= 0.0 
        return Inf
    else
        return -x/v
    end  
end

function event(x, v, vc, s, tmax, λs, λref, κs)
    τ0 = tmax
    k0 = 0
    ev = 0
    for k in eachindex(λs)
        if s[k] == 0
            τ1 = λs[k] <= 0 ? Inf : -log(rand())/λs[k]
            τ2 = λref <= 0 ? Inf : -log(rand())/λref
            τ3 = hitting_time(x[k],v[k])
        else
            τ1 = Inf
            τ2 = λref <= 0 ? Inf : -log(rand())/λref
            τ3 = -log(rand())/(κs[k].*abs(vc[k]))
        end
        if τ0 < min(τ1,τ2, τ3)
            continue
        elseif τ1 < min(τ2, τ3)
             τ0, k0, ev = τ1, k, 1
        elseif τ2 < τ3
            τ0, k0, ev = τ2, k, 2
        else
            τ0, k0, ev = τ3, k, 3
        end
    end
    return τ0, k0, ev
end
   
function sg_sticky_zz_model(∇U!, κs, x0, h, N, args...; 
    thin = 100, λref = 0.0, 
    max_grad_ev = Inf, 
    cv = CV(false),
    test = Test(false),
    verbose = false,
    trace = true,
    minibatch = 10, 
    nobs)
    if verbose
        println("Info:\n 
          sampler: sticky-sg-zz,\n
          step size: $h,\n
          λref: $(λref),\n
          batchsize: $(minibatch),\n
          N obs: $(nobs),\n
          control variates: $(cv.cv),\n
         ")
      end
    # x = vi_current[spl]
    out = Output("sticky-sg-zz")
    x = copy(x0)
    x = x0
    p = length(x)
    v = rand([-1.0, +1.0], p) 
    ∇Ux = zeros(p) 
    ∇Ux = ∇U!(∇Ux, x,  nobs, minibatch, cv, args...) 
    vc = copy(v)
    s = zeros(Float64, p)
    i = 1
    t = 0.0
    τ0 = Inf
    k0 = 0
    ev = 0 
    dt = h
    flips = 0
    grad_ev = minibatch
    # IMPOPRTANT: ta can be used to derive the posterior probability to select a given coordinate
    ta = zeros(length(x0)) # active time of each coordinate
    while(i <= N)
        if grad_ev > max_grad_ev
            out.max_grad = true
            break
        end
        τ0, k0, ev  = event(x, v, vc, s, dt, ∇Ux.*v, λref, κs)
        t += τ0
        for i in eachindex(x)
            if v[i] != 0
                x[i] += v[i]*τ0
                ta[i] += τ0 
            end  
        end
        if ev == 1 || ev == 2 # either reflection or refreshment, can also check with if k0 = 0
            dt = dt - τ0
            v[k0] *= -1
            vc[k0] *= -1 
            flips += 1 
        elseif ev == 3
            dt = dt - τ0
            if s[k0] == 1
                x[k0] = 0.0
                v[k0] = vc[k0]
                s[k0] = 0
            else
                x[k0] = 0.0
                v[k0] = 0.0
                s[k0] = 1
            end 
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
        grad_ev += minibatch
        ∇Ux = ∇U!(∇Ux, x,  nobs, minibatch, cv, args...) 
    end
    out
end

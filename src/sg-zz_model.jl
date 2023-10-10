# it requires structure Output, Test, CV 
function eventzz(tmax, λs, λref)
    τ0 = tmax
    k0 = 0
    ev = 0
    for k in eachindex(λs)
        τ1 = λs[k] <= 0 ? Inf : -log(rand())/λs[k]
        τ2 = λref <= 0 ? Inf : -log(rand())/λref
        if τ0 < min(τ1,τ2)
            continue
        elseif τ1 < τ2
             τ0, k0, ev = τ1, k, 1
        else
            τ0, k0, ev = τ2, k, 2
        end
    end
    return τ0, k0, ev
end
   
function sg_zz_model(cond_model, x0, h, N; 
        thin = 100, λref = 0.0,
        max_grad_ev = Inf, 
        cv = CV(false),
        test = Test(false),
        verbose = false,
        trace = true,
        ad_backend = ForwardDiffAD{40}(),
        minibatch,
        nobs)
    if verbose
        println("Info:\n 
          sampler: sg-zz,\n
          step size: $h,\n
          λref: $(λref),\n
          batchsize: $(minibatch),\n
          N obs: $(nobs),\n
          control variates: $(cv.cv),\n
          ")
      end
      vi_orig = DynamicPPL.VarInfo(cond_model)
      spl = DynamicPPL.SampleFromPrior()
      vi_current = DynamicPPL.VarInfo(DynamicPPL.VarInfo(cond_model), spl, vi_orig[spl])
      f = LogDensityProblemsAD.ADgradient(
          ad_backend,
          Turing.LogDensityFunction(vi_current, cond_model, spl, DynamicPPL.DefaultContext())
          )   
      logp(x) = LogDensityProblems.logdensity_and_gradient(f, x)[1]
      _∇logp(x) = LogDensityProblems.logdensity_and_gradient(f, x)[2]*nobs/minibatch
      function ∇logp(x, cv) 
        if cv.cv == false
            return _∇logp(x)
        else
            return _∇logp(x) -  _∇logp(cv.x0) + cv.∇x0
        end
      end  
    out = Output("sg-zz")
    # x = vi_current[spl]
    x = copy(x0)
    p = length(x)
    v = rand([-1.0, +1.0], p) 
    ∇Ux = -∇logp(x, cv)
    i = 1
    t = 0.0
    τ0 = Inf
    k0 = 0
    ev = 0 
    dt = h
    flips = 0
    grad_ev = minibatch
    while(i <= N)
        if grad_ev > max_grad_ev
            out.max_grad = true
            break
        end
        τ0, k0, ev  = eventzz(dt, ∇Ux.*v, λref)
        t += τ0
        x .+= v*τ0  
        if ev != 0 # either reflection or refreshment, can also check with if k0 = 0
            dt = dt - τ0
            v[k0] *= -1
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
        grad_ev += minibatch
        ∇Ux = -∇logp(x, cv)
    end
    @show grad_ev
    out
end



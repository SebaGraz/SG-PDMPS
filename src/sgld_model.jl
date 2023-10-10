function sgld(cond_model, x0, h::Float64, N::Int64, args...;
    thin = 100, 
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
          sampler: sgld,\n
          step size: $h,\n
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
    out = Output("sgld")
    x = copy(x0)
    p = length(x)
    grad_ev = 0
    i = 1
    while(i < N)
        if grad_ev > max_grad_ev
          out.max_grad = true
          break
        end
        grad_ev += minibatch
        ∇Ux = -∇logp(x, cv)
        x -= h/2*∇Ux + sqrt(h)*randn(p)
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
            println("progress: $(i/N*100)%...")
        end
        i += 1
    end
    @show grad_ev
    out
end

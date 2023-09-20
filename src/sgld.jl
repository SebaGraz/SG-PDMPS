
function sgld(∇U!, x0, h::Float64, N::Int64, args...;
    thin = 100, 
    max_grad_ev = Inf, 
    cv = CV(false),
    test = Test(false),
    verbose = false,
    trace = true,
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
    out = Output("sgld")
    x = copy(x0)
    p = length(x)
    ∇Ux  = zeros(p)
    grad_ev = 0
    i = 1
    while(i < N)
        if grad_ev > max_grad_ev
          out.max_grad = true
          break
        end
        grad_ev += minibatch
        ∇Ux = ∇U!(∇Ux, x,  nobs, minibatch, cv, args...) 
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
    out
end

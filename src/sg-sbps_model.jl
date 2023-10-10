
sticky_normsq(x, s) = sticky_dot(x, x, s)
function sticky_dot(x::Vector{Float64}, y::Vector{Float64}, s::Vector{Bool}) 
    res = 0.0
    for k in eachindex(x)
        if s[k] == 0
            res = res + x[k]*y[k]
        end
    end
    res 
end


function bounce_sticky!(s, ∇Ux, v)
    for i in eachindex(s)
        if s[i] == 0
            v[i] -= (2*sticky_dot(∇Ux, v, s)/sticky_normsq(∇Ux, s))*∇Ux[i]
        end
    end
end


function event_sticky_bps(x, v, vc, s, κs, tmax, λ, λref)
    τ1 = λ <= 0 ? Inf : -log(rand())/λ
    τ2 = -log(rand())/λref
    j0 = 0
    τ3 = Inf
    for k in eachindex(x)
        if s[k] == 0
            t = hitting_time(x[k],v[k])
        else
            t = -log(rand())/(κs[k].*abs(vc[k]))
        end
        if t < τ3
            j0 = k
            τ3 = t
        end
    end
    if tmax < min(τ1, τ2, τ3)
        return 0, tmax, 0
    elseif τ1 < min(τ2, τ3)
        return 1, τ1, 0
    elseif τ2 < τ3
        return 2, τ2, 0
    else
        return 3, τ3, j0
    end
end

function hitting_time(x,v)
    if x*v>= 0.0 
        return Inf
    else
        return -x/v
    end  
end



function sg_sticky_bps_model(cond_model, κs, x0, h::Float64, N::Int64; 
        thin = 100, λref = 1.0,
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
        sampler: sg-bps,\n
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
    
    x = copy(x0)
    p = length(x)
    s = zeros(Bool, p)
    out = Output("sg-bps")
    v = randn(p)
    vc = copy(v)
    # x = vi_current[spl]
    ∇Ux = -∇logp(x, cv)
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
        ev, τ, k  = event_sticky_bps(x, v, vc, s, κs, dt, sticky_dot(∇Ux, v, s), λref)
        t += τ
        x .+= v*τ  
        if ev == 1
            dt = dt - τ
            # right now I am not able to reuse the same jth observation 
            # ∇Ux = ∇U!(∇Ux, x, j, args...)
            bounce_sticky!(s, ∇Ux, v)
            flips += 1
        elseif ev == 2
            dt = dt - τ
            vc = randn(p)
            for i in eachindex(x)
                if s[i] == 0
                    v[i] = vc[i]
                end
            end
            flips += 1  
        elseif ev == 3
            dt = dt - τ
            if s[k] == true
                x[k] = 0.0
                v[k] = vc[k]
                s[k] = 0
            else
                x[k] = 0.0
                v[k] = 0.0
                s[k] = 1
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
        # update gradient
        grad_ev += minibatch
        ∇Ux = -∇logp(x, cv)
    end
   out
end


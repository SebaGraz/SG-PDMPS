println("generate data... p=$(p) and N = $(nobs)")
Σ = zeros(p-1,p-1)
for i in 1:p-1
    for j in 1:i
        Σ[i,j] = Σ[j,i] = (rand()*(0.8) - 0.4)^abs(i-j)
    end
end

A = ones(Float64,nobs,p)
Xij = MultivariateNormal(Σ)
for j in 1:nobs
    A[j,2:end]= rand(Xij)
end
At = A'

σ2 = 1 #variance rescaled to avoid posterior contraction
xtrue_ = randn(p)*sqrt(σ2) ##true parameter value
# sparsity = 0.5
# jj = sample(1:p, Int(p*sparsity), replace = false)
xtrue = xtrue_
# xtrue[jj] .= 0.0 

# simulate responses
y = (rand(nobs) .< sigmoid.(A*xtrue))


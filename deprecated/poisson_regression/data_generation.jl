
# μ = randn()*10.0
I = 50 # numb groups
In = 50 # numb groups
n = 10_000 # numb of observation per group
μ0 = 5.0
ση = 1.0
η = ση*randn(In) .+ μ0  # samle η from the prior
Y = zeros(Int64, In, n)
for i in 1:In
    for j in 1:n
        Y[i,j] = rand(Poisson(exp(η[i])))
    end
end

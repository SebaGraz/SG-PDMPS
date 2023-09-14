include("./../src/utilities.jl")
∇U(x) = x/100.0
xx = randn(10,1000)
stein_metric(∇U, xx*5, 0.5, -0.05)


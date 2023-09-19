using LinearAlgebra
include("./../src/utilities.jl")
∇U(x) = x/100.0
xx = randn(10,1000)
stein_metric(∇U, xx , 1.0, -0.5)

new_stein_metric(∇U, xx*14, 1.0, -0.5)

c = 1.0
β = -0.5
using ForwardDiff, LinearAlgebra
u(x) = x^2/2
norm2(x) = dot(x,x)
∇U(x, i) = x[i]

k0(x, y, c, β) = (c^2 + norm2(x-y))^(β) 
∇1k0(x,y, i, c, β) = ForwardDiff.gradient(z -> k0(z,y,c, β), x)[i]
∇2k0(x,y,i,c, β) = ForwardDiff.gradient(z -> k0(x,z,c, β), y)[i]
∇12k0(x,y,i,j,c, β) =  ForwardDiff.gradient(z -> ∇2k0(z,y,i,c, β), x)[j]



function kernel(xx, c, β)
    res = 0.0
    n = length(xx)
    p = length(xx[1])
    for j in 1:p
        resi = 0.0
        for x in xx
            for y in xx
                resi += ∇U(x,j)*∇U(y,j)*k0(x,y,c, β) + ∇U(x,j)*∇2k0(x,y,j,c, β) + ∇U(y,j)*∇1k0(x,y,j,c, β) + ∇12k0(x,y,j,j,c, β)
            end
        end
        res += sqrt(resi)
    end
    res/n
end


function stein_k(j, θ1, θ2, ∇θ1, ∇θ2, c::Float64, β::Float64)
    res = ∇θ1[j]*∇θ2[j]*(c^2 + norm2(θ1 - θ2))^β + 2*β*(c^2 + norm2(θ1 - θ2))^(β-1)*(θ1[j] - θ2[j])*(∇θ2[j] - ∇θ1[j]) - 4*(β)*(β-1)*(c^2 + norm2(θ1 - θ2))^(β-2)*(θ1[j] - θ2[j])^2 - 2*β*(c^2 + norm2(θ1 - θ2))^(β-1)
    return res 
end



function stein_kernel(xx, c::Float64, β::Float64)
    res = 0.0
    n = length(xx)
    p = length(xx[1])
    for j in 1:p
        resi = 0.0
        for x in xx
            for y in xx
                resi += stein_k(j, x, y, x, y, c, β)
            end
        end
        res += sqrt(resi)
    end
    res/n
end

function stein_kernel2(xx, c::Float64, β::Float64)
    res = 0.0
    n = length(xx)
    p = length(xx[1])
    for j in 1:p
        resi = 0.0
        for i in eachindex(xx)
            for k in 1:i
                m = (i != k)*1.0 + 1.0 
                resi += m*stein_k(j, xx[i], xx[k], ∇U(xx[i]), ∇U(xx[k]), c, β)
            end
        end
        res += sqrt(resi)
    end
    res/n
end



xx = [randn(10) for _ in 1:1000]
∇U(x) = x
c = 1.0
β = -0.1
stein_kernel(xx*1.0, c, β)
stein_kernel(xx*0.9, c, β)
stein_kernel(xx*1.1, c, β)

stein_kernel2(xx*1.1, c, β)

# kernel(xx, c, β)
# kernel(xx*0.8, c, β)
# kernel(xx*1.2, c, β)
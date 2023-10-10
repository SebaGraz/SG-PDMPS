using Plots, StatsBase, CSV, DataFrames
str_regression = "linear_regression"
str_h = "h_"
str_d = "d_"
# h = 5e-07
h = 5e-07
# pp = [10, 50, 100, 200, 300, 400, 500, 600, 700,  800,  900, 1000]
pp = [10, 20, 30, 40, 50, 60, 70, 100]


# str_sampler = "sgld3_"
# res = zeros(length(pp))
# for i in eachindex(pp)
#         d = pp[i]
#         str_datain = "./scripts/"*str_regression*"/scaling/variances/"*str_sampler*str_h*string(h)*"_"*str_d*string(d)*".csv"
#         ac = Matrix(CSV.read(str_datain, DataFrame, header=false))
#         res[i] = sum(ac.^2)/length(ac)
# end
# plot!(pp, res)

str_sampler = "zz_"
res = zeros(length(pp))
for i in eachindex(pp)
        d = pp[i]
        str_datain = "./scripts/"*str_regression*"/scaling/variances/"*str_sampler*str_h*string(h)*str_d*string(d)*".csv"
        ac = Matrix(CSV.read(str_datain, DataFrame, header=false))
        res[i] = sum(ac.^2)/length(ac)
end
plot(pp, res, label = "sg-zz")

str_sampler = "bps_"
res = zeros(length(pp))
for i in eachindex(pp)
        d = pp[i]
        str_datain = "./scripts/"*str_regression*"/scaling/variances/"*str_sampler*str_h*string(h)*str_d*string(d)*".csv"
        ac = Matrix(CSV.read(str_datain, DataFrame, header=false))
        res[i] = sum(ac.^2)/length(ac)
end
plot!(pp, res, label = "sg-bps")
str_sampler = "sgld1_"
res = zeros(length(pp))
for i in eachindex(pp)
        d = pp[i]
        str_datain = "./scripts/"*str_regression*"/scaling/variances/"*str_sampler*str_h*string(h)*str_d*string(d)*".csv"
        ac = Matrix(CSV.read(str_datain, DataFrame, header=false))
        res[i] = sum(ac.^2)/length(ac)
end
# res[end] =  res[end-1] = res[3] = NaN # unstable algorithm 
@show res
res[end-1] = res[end-2] = NaN # sampler diverges
plot!(pp, res, label = "sgld")

title!("Scaling")
xlabel!("d")
savefig("./scripts/linear_regression/scaling/output/scaling.pdf")
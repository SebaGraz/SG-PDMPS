using Plots, StatsBase, CSV, DataFrames
str_regression = "linear_regression"
str_h = "h_"
        str_d = "d_"
str_sampler = "bps_"
d = 50
h = 0.000001
pp = [10, 50, 100, 150, 200]
f1 = plot(title = "BPS 1 coordinate")
f2 = plot(title = "BPS 1 mean")
for d in pp
        str_datain = "./scripts/"*str_regression*"/scaling/autocorrelation/"*str_sampler*str_h*string(h)*"_"*str_d*string(d)*".csv"
        ac = Matrix(CSV.read(str_datain, DataFrame, header=false))
        plot!(f1, ac[1,:])
        plot!(f2, mean(ac, dims = 1)[:])
end
f1
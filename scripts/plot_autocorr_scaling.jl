using Plots, StatsBase
str_regression = "linear_regression"
str_h = "h_"
        str_d = "d_"
str_sampler = "bps_"
d = 50
h = 0.001
str_datain = "./scripts/"*str_regression*"/scaling/autocorrelation/"*str_sampler*str_h*string(h)*"_"*str_d*string(d)*".csv"
ac = Matrix(CSV.read(str_datain, DataFrame, header=false))
plot(ac[1,:])
plot(mean(ac, dims = 1)[:])
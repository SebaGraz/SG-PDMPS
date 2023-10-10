
using Plots, StatsBase, CSV, DataFrames

str_iter = "_iter_"
s3 = ".csv"
str_h = "_h_"

# s2 = "boston"
# str_sampler = "_sgld"
# h = 1e-06
# iter = 1
# trace = Matrix(CSV.read(out_traces*str_sampler*str_h*string(h)*str_iter*string(iter)*s3, DataFrame, header=false))[:]


ff = []
ff2 = []
iter0 = 1
s0 = "boston" # "boston" "wine" "concrete" "kin8nm "protein-structure"
out_traces = "./scripts/neural_net/prediction/"
out_traces2 = "./scripts/neural_net/prediction2/"

hh = [1e-03, 1e-04, 1e-05, 1e-06]
ssamplers = ["_zz", "_bps", "_sgld", "_szz"]
for h in hh
    f = plot(title = "h = $(h)")
    f2 = plot(title = "h = $(h)")
    for sampler in ssamplers
        trace = Matrix(CSV.read(out_traces*s0*sampler*str_h*string(h)*str_iter*string(iter0)*s3, DataFrame, header=false))[:]
        trace2 = Matrix(CSV.read(out_traces2*s0*sampler*str_h*string(h)*str_iter*string(iter0)*s3, DataFrame, header=false))[:]
        plot!(f, trace, label = sampler, alpha = 0.4)
        plot!(f2, trace2, label = sampler)
    end
    push!(ff, f)
    push!(ff2, f2)

end
plot(ff..., layout = 4, plot_title = s0*" iter = $(iter0)")
savefig(s0*"_neuralnet_h.pdf")
plot(ff2..., layout = 5, plot_title = s0*" iter = $(iter0)")
savefig(s0*"_mean_neuralnet_h.png")
ff2


using StatsBase
ff2 = []
fdata = []
# ddataset = ["boston", "concrete", "wine", "kin8nm", "protein-structure"]

    # f = plot(title = dataset, xlabel = "h",  yaxis = "av ℓ", xaxis = :log)
hh = [1e-03, 1e-04, 1e-05, 1e-06]
num = zeros(4, length(hh))
k = 1
str = 0.0
for sampler in ssamplers
    res = zeros(length(hh))
    count = 1
    for h in hh
        for iter in 1:3
            trace = Matrix(CSV.read(out_traces*s0*sampler*str_h*string(h)*str_iter*string(iter)*s3, DataFrame, header=false))[:]
            num[k, count] += mean(trace)/3
            
            # num[k, count] += res[count]
        end
        count += 1
    end
    # plot!(f, hh, res, label = sampler)
    k += 1
end
num

trace
# hline!([str], label = "control_variate")
# push!(ff2, f)
# push!(fdata, num)


# plot(ff2..., layout = (5))
# savefig("./neuralnet_loss.png")
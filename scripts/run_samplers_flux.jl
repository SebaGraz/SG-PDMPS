using CSV, DataFrames, StatsBase, Random, Test
include("./../src/sgld_flux.jl")
include("./../src/sg-bps_flux.jl")
include("./../src/sg-zz_flux.jl")
include("./../src/sg-szz_flux.jl")
include("./../test/test_adam_flux.jl")
ss = ["protein-structure"]
# ss = ["boston"]
hh = [1e-03, 1e-04, 1e-05, 1e-06]
str_iter = "_iter_"

Random.seed!(1)
s1 = "scripts/neural_net/data/"
s3 = ".csv"
str_h = "_h_"
Niter = 10^6 #number of iterations for each algorithm
# Niter = 10^4
Niter_optim = 10^5 #number of iterations
# Niter_optim = 2*10^3 #number of iterations

# Niter = 10^5 #number of iterations for each algorithm
# Niter_optim = 10^5 #number of iterations
thin = 100


for s2 in ss
    println("dataset = $(s2)")
    out_traces = "./scripts/neural_net/prediction/"*s2
    out_traces2 = "./scripts/neural_net/prediction2/"*s2


    data=DataFrame(CSV.File(s1*s2*s3));
    println("data size (N,p) = $(size(data))")
    train_ratio = 0.9 # We create the train and test sets with 90% and 10% of the data
    # The last column is the label and the other columns are features
    X_input = Matrix(data[ :, 1:size(data)[2]-1]);
    y_input = data[ :, size(data)[2]];


    #Normalise the data
    dt_x = fit(ZScoreTransform, X_input, dims=2)
    dt_y = fit(ZScoreTransform, y_input)

    X_input = StatsBase.transform(dt_x, X_input)
    y_input = StatsBase.transform(dt_y, y_input)

    X_input = Float32.(X_input)
    y_input = Float32.(y_input)

    size_train = Int64(round(size(X_input)[1] * train_ratio));
    for iter in 2:3
        println("iter = $(iter)")
        
        # Build the training and testing data set
        
        


        permute = StatsBase.sample(1:size(X_input)[1],size(X_input)[1],replace=false);
        X_train, y_train = X_input[permute[1:size_train], : ], y_input[ permute[1:size_train] ];
        X_test, y_test = X_input[permute[(size_train+1):end], : ], y_input[ permute[(size_train+1):end] ];

        Xt_train, yt_train = X_train', y_train' 
        Xt_test, yt_test = X_test', y_test'

        model = Chain(Dense(size(Xt_train)[1]=>50),Dense(50=>1,relu)) 

        λ = 0.1
        function loss(m, x, y)
            params, _ = Flux.destructure(m) 
            0.5*Flux.mse(m(x), y) + sum(abs2, params)*λ*0.5
        end
        function loss2(m, x, y)
            0.5*Flux.mse(m(x), y) 
        end

        mb = Int64(round(size(X_input)[1] * 0.01)) # size minibatch 1% x (data size)
        ll0 = loss(model, Xt_train, yt_train)
        println("initial loss: $(ll0)")
        model1 = AdamCV((Xt_train, yt_train, model, loss, mb), Niter_optim)
        ll1 = loss(model1, Xt_train, yt_train)
        println("loss after adam: $(ll1)")
        @test ll1 < ll0

        
        
        for h in hh
            println("h = $(h)")
            str_sampler = "_sgld"
            println(str_sampler)
            model2, trace, trace2  = sgld_flux((Xt_train, yt_train, model1, loss), loss2, Niter, h, thin, (Xt_test, yt_test))
            CSV.write(out_traces*str_sampler*str_h*string(h)*str_iter*string(iter)*s3, Tables.table(trace), header = false)     
            CSV.write(out_traces2*str_sampler*str_h*string(h)*str_iter*string(iter)*s3, Tables.table(trace2), header = false)     

            str_sampler = "_bps"
            println(str_sampler)
            model3, trace,  trace2  = sgbps_flux((Xt_train, yt_train, model1, loss), loss2, 1.0,  Niter, h, thin, (Xt_test, yt_test))
            CSV.write(out_traces*str_sampler*str_h*string(h)*str_iter*string(iter)*s3, Tables.table(trace), header = false)     
            CSV.write(out_traces2*str_sampler*str_h*string(h)*str_iter*string(iter)*s3, Tables.table(trace2), header = false)     

        
            str_sampler = "_zz"
            println(str_sampler )
            model4, trace, trace2  = sgzz_flux((Xt_train, yt_train, model1, loss), loss2, 0.0,  Niter, h, thin, (Xt_test, yt_test))
            CSV.write(out_traces*str_sampler*str_h*string(h)*str_iter*string(iter)*s3, Tables.table(trace), header = false)     
            CSV.write(out_traces2*str_sampler*str_h*string(h)*str_iter*string(iter)*s3, Tables.table(trace2), header = false)     

        
            str_sampler = "_szz"
            println(str_sampler)
            sparsity = 1.0
            model5, trace, trace2  = sgszz_flux((Xt_train, yt_train, model1, loss), loss2, sparsity, 0.0,  Niter, h, thin, (Xt_test, yt_test))
            CSV.write(out_traces*str_sampler*str_h*string(h)*str_iter*string(iter)*s3, Tables.table(trace), header = false)    
            CSV.write(out_traces2*str_sampler*str_h*string(h)*str_iter*string(iter)*s3, Tables.table(trace2), header = false)      
        end
    end
end
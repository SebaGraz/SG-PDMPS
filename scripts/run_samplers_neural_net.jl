#This script runs a Bayesian neural network on standard UCI benchmark datasets
using Plots
using Flux
using Random
using Turing
using LazyArrays
using LogDensityProblems
using Turing.Essential: ForwardDiffAD, ReverseDiffAD
using LogDensityProblemsAD
using StatsBase, Plots, LinearAlgebra, DataFrames, CSV
include("./../src/utilities.jl")
include("./../src/sgld_model.jl")
include("./../src/sg-bps_model.jl")
include("./../src/sg-sbps_model.jl")
function runall()
    s1 = "scripts/neural_net/data/"
    s2 = "boston"
    s3 = ".csv"

    hh = [1e-03, 1e-04, 1e-05, 1e-06]

    data=DataFrame(CSV.File(s1*s2*s3));

    size(data)

    Niter = 10^4 #number of iterations
    thin = 10
    Random.seed!(1);
    # The last column is the label and the other columns are features
    X_input = Matrix(data[ :, 1:size(data)[2]-1]);
    y_input = data[ :, size(data)[2]];

    #Normalise the data
    dt_x = fit(ZScoreTransform, X_input, dims=2)
    dt_y = fit(ZScoreTransform, y_input)

    X_input = StatsBase.transform(dt_x, X_input)
    y_input = StatsBase.transform(dt_y, y_input)

    # Build the training and testing data set
    train_ratio = 0.9 # We create the train and test sets with 90% and 10% of the data
    size_train = Int64(round(size(X_input)[1] * train_ratio));

    # strings
    out = "./scripts/neural_net/test_data/"*s2
    out_traces = "./scripts/neural_net/prediction/"*s2
    str_iter = "_iter_"
    str_h = "_h_"

    for iter in 3:5
        # split dataset in train and test
        permute = StatsBase.sample(1:size(X_input)[1],size(X_input)[1],replace=false);
        X_train, y_train = X_input[permute[1:size_train], : ], y_input[ permute[1:size_train] ];
        X_test, y_test = X_input[permute[(size_train+1):end], : ], y_input[ permute[(size_train+1):end] ];
        CSV.write(out*str_iter*string(iter)*s3, DataFrame(hcat(X_test,y_test), :auto), header = false)    
        # todo save test dataset
    # end  
        # define model (neural network) 
        nn_initial = Chain(Dense(size(X_train)[2]=>50),Dense(50=>1,relu)) 
        # using ReverseDiff
        # Turing.setadbackend(:reversediff)

        # Extract weights and a helper function to reconstruct NN from weights
        parameters_initial, reconstruct = Flux.destructure(nn_initial)
        @show length(parameters_initial) # number of parameters in NN
        
        # Create a regularization term and a Gaussian prior variance term.
        alpha = 0.09
        sigm(alpha) = sqrt(1.0 / alpha)

        @model function bayes_nn(xs, ys, nparameters, reconstruct; alpha=0.09, minibatch = 10, Nobs)
            N = size(xs, 1)
            # Create the weight and bias vector.
            parameters ~ MvNormal(zeros(nparameters), sigm(alpha) .* ones(nparameters))

            # Construct NN from parameters
            nn = reconstruct(parameters)
            # Forward NN to make predictions
            for _ in 1:minibatch
                i = rand(1:N)
                predi = nn(xs[i, :])
                # Observe prediction i.
                ys[i] ~ Normal(predi[1],0.1)
            end
        end;
        mb_sgld = Int(round(0.01*length(y_train)))
        # data subsample model
        cond_model_sgld= bayes_nn(X_train, y_train, length(parameters_initial), reconstruct; minibatch = mb_sgld, Nobs = size(X_train, 1))| (;y_train);
        cond_model_1= bayes_nn(X_train, y_train, length(parameters_initial), reconstruct; minibatch = 1, Nobs = size(X_train, 1))| (;y_train);

        # run optimizer
        @model function bayes_nn_full(xs, ys, nparameters, reconstruct; alpha=0.09)
            N = size(xs, 1)
            # Create the weight and bias vector.
            parameters ~ MvNormal(zeros(nparameters), sigm(alpha) .* ones(nparameters))
            # Construct NN from parameters
            nn = reconstruct(parameters)
            # Forward NN to make predictions
            for i in 1:N
                predi = nn(xs[i, :])
                # Observe prediction i.
                ys[i] ~ Normal(predi[1], 0.1)
            end
        end;
        full_cond_model = bayes_nn_full(X_train, y_train, length(parameters_initial), reconstruct)| (;y_train);
        #Run the optimiser to find the MAP
        Nobs = size(X_train, 1)
        # Niter_opt = 10^4
        Niter_opt = 10^4
        ϵ = 1e-7
        ad_backend = ForwardDiffAD{40}()
        @time cv = findCV(cond_model_sgld, full_cond_model, ϵ, Niter_opt, mb_sgld, Nobs; ad_backend = ad_backend);
        println("norm of gradient at control variates: $(norm(cv.∇x0))")
        for h in hh
            println("h = $(h)")
            Niter_sgld = Niter ÷ cond_model_sgld.defaults.minibatch
            str_sampler = "_sgld2"
            println("Running SGLD mb = 1% x data = $(cond_model_sgld.defaults.minibatch)")
            x0 = copy(cv.x0) 
            verbose = false
            control_variates = cv
            trace = true
            res = sgld(cond_model_sgld, x0, h, Niter_sgld; 
                    thin = thin, 
                    cv = cv,
                    trace = trace,
                    verbose = verbose, 
                    minibatch = cond_model_sgld.defaults.minibatch, nobs = cond_model_sgld.defaults.Nobs);
            
            CSV.write(out_traces*str_sampler*str_h*string(h)*str_iter*string(iter)*s3, DataFrame(res.trace, :auto), header = false)     

            println("Running SGLD mb = 1")
            str_sampler = "_sgld1"
            x0 = copy(cv.x0) 
            verbose = false
            control_variates = cv
            trace = true
            res = sgld(cond_model_1, x0, h, Niter; 
                    thin = thin, 
                    cv = cv,
                    trace = trace,
                    verbose = verbose, 
                    minibatch = cond_model_1.defaults.minibatch, nobs = cond_model_1.defaults.Nobs);
            CSV.write(out_traces*str_sampler*str_h*string(h)*str_iter*string(iter)*s3, DataFrame(res.trace, :auto), header = false)     

            println("Running BPS mb = 1")
            str_sampler = "_bps"
            x0 = copy(cv.x0)
            verbose = false
            control_variates = cv
            trace = true
            res = sg_bps_model(cond_model_1, x0, h, Niter; 
                            thin = thin, 
                            verbose = verbose, cv = control_variates,
                            minibatch = cond_model_1.defaults.minibatch, nobs = cond_model_1.defaults.Nobs);   
                            
                            CSV.write(out_traces*str_sampler*str_h*string(h)*str_iter*string(iter)*s3, DataFrame(res.trace, :auto), header = false)     
            
            
            println("Running sticky BPS mb = 1")
            str_sampler = "_sbps"
            x0 = copy(cv.x0)
            verbose = false
            control_variates = cv
            trace = true
            κs = ones(length(cv.x0))
            res = sg_sticky_bps_model(cond_model_1, κs, x0, h, Niter; 
                    thin = thin, 
                    verbose = verbose, cv = control_variates,
                    minibatch = cond_model_1.defaults.minibatch, nobs = cond_model_1.defaults.Nobs);   
            # str_sampler = "_zz"
            # println("Running ZZ mb = 1")
            # x0 = copy(cv.x0)
            # verbose = true
            # control_variates = cv
            # trace = true
            # res = sg_zz_model(cond_model_1, x0, h, Niter; 
            #                 thin = thin,
            #                 verbose = verbose, cv = control_variates, 
            #                 minibatch = cond_model_1.defaults.minibatch, nobs = cond_model_1.defaults.Nobs)
            #                 CSV.write(out_traces*str_sampler*str_h*string(h)*str_iter*string(iter)*s3, DataFrame(res.trace, :auto), header = false)     
            
            # str_sampler = "_szz"   
            # println("Running sticky ZZ mb = 1")
            # spars = 1.0
            # x0 = copy(cv.x0)
            # κs = ones(length(cv.x0)).*spars
            # verbose = false
            # control_variates = cv
            # trace = true
            # res = sg_sticky_zz_model(cond_model_1, κs, x0, h, Niter; thin = thin, 
            #         verbose = verbose, cv = control_variates, 
            #         minibatch = cond_model_1.defaults.minibatch, nobs = cond_model_1.defaults.Nobs)
            #         CSV.write(out_traces*str_sampler*str_h*string(h)*str_iter*string(iter)*s3, DataFrame(res.trace, :auto), header = false)     
                    
            

        end
    end
end

runall()
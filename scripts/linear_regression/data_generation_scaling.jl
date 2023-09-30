using CSV, DataFrames, Distributions, Random
str_regression = "linear_regression"
str_folder = "./scripts/"*str_regression*"/scaling/data/"
str_d = "d_"
str_csv = ".csv"
str_data = "data_"
nobs = 10^4
dd = [10, 50, 100, 200, 300, 400, 500, 650, 800, 1000]

Random.seed!(1234)
for p in dd
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
    xtrue = randn(p)*sqrt(σ2) ##true parameter value
    # simulate responses
    y = A*xtrue + randn(nobs)     
    DIROUTDATA = str_folder*str_data*str_d*string(p)*str_csv 
    CSV.write(DIROUTDATA, DataFrame(hcat(A,y), :auto), header = false) 
end
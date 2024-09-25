###### 
# Lecture 2: Structural Estimation

using Plots, Parameters, Distributions,LinearAlgebra, Optim

# working directory is the current file
cd(dirname(@__FILE__))

# Define the Parameter in a mutable construct
@with_kw mutable struct set_para
    # Timing parameters
    T::Int = 20                                        # Maximum age of life
    Tᵣ::Int = 15                                       # Retirement age
	l::Vector{Float16} = vcat(ones(Tᵣ), zeros(T-Tᵣ))     # Retirement age

    # Prices
    r::Float64 = 0.13                                  # Gross interest rate after taxes
    w::Float64 = 1.0                                   # Wage

    # Preferences
    β::Float64 = 0.96                                  # Patience
    ρ::Float64 = 2.0                                   # Relative Risk Aversion (RRA) / Inverse IES
	
	# Distribution
    μ::Float64 = 0.0                                   # Location of Lognormal
    σ::Float64 = 1.0                                   # Scale of Lognormal
	
end

# Create an instance of the struct
para = set_para()

############# 
# Part A: Load functions from lecture 1
include("Functions_Lecture1.jl")

############# 
# Part B: Data
# 1. We simulate the model with the "true" parameters.
# 2. The outcome is our "empirical" data set.
# 3. We therefore know exactly what our estimation should lead to.

# The true parameter are
para.β = 0.96
# set the seed to replicaiton purposes
Random.seed!(1)
# We simulate the model with the "true" parameters
ConsumptionData,SavingData = SimLCM(para,1000)
# How many of individuals are in the the data
Ndata,_ = size(ConsumptionData)

############# 
# Part C: Calculate the data moments
# Ages to be targeted
ages = [5,10,15]
# Create moments of the data
Λᵈ = mean(ConsumptionData[:,ages],dims=1)
N_mom = length(Λᵈ) # number of moments

############# 
# Part C: Wrtie a function that simulates the moments
S = 100 # set the number of simulations
function sim_moments(S,ages,Ndata,para)
    # Goal: This functions simulates the moments based on S simulations
	# Inputs:
	# S: Number of simulations
	# ages: vector of ages to targeted
    # Ndata: Number of individuals in the empirical data
    # para: parameters of the model
    
    # Set the seed
    Random.seed!(1)
    # Store moments per simulation in the following vectors
    Λˢₛ = zeros((S,length(ages)))
    for s in 1:S
        SimCon, SimSav = SimLCM(para,Ndata)
        Λˢₛ[s,:] = mean(SimCon[:,ages],dims=1)
    end
    # stack them into a vector
    Λˢ = mean(Λˢₛ,dims=1)
	return Λˢ
end
############# 
# Part D: Wrtie a function that calculates the error vector
function moments_diff(data,ages,S,para)
    # Goal: This function calculates the error vectors
	# Inputs:
	# data: empirical data
	# ages: vector of ages to targeted
    # S: Number of simulations
    # para: parameters of the model
    
    Ndata,_ = size(data) # Get the number of individuals in the empirical data
    # Data moments
    Λᵈ= mean(data[:,ages],dims=1)
    # Simulate moments based on parameters
    Λˢ =  sim_moments(S,ages,Ndata,para)
    # return the error vector
	return vec(Λᵈ - Λˢ)
end
############# 
# Part E: Wrtie a function for the objective function
function smm_objective(Θ,data,ages,S,W,para)
    # Goal: This function returns the objective function to be minimized
	# Inputs:
    # Θ: vector of parameters to be estimated
	# data: empirical data
	# ages: vector of ages to targeted
    # S: Number of simulations
    # W: Weighting matrix
    # para: parameters of the model

    # set parameter
    para.β = Θ[1]
    # If W is not provided (i.e., is nothing), default to identity matrix
    if W === nothing
        W = Diagonal(ones(length(ages)))  # Identity matrix in Julia
    end
    # Compute the difference between data and model moments
    Mdiff = moments_diff(data,ages,S,para)
    # Compute the quadratic form
    objective_value = Mdiff' * W * Mdiff
    
    return objective_value
end

# Optimize using Nelder-Mead
result = optimize(x -> smm_objective(x, ConsumptionData,ages,S, nothing,para), [0.90], NelderMead())
# access the SSM estimator
β_hat = Optim.minimizer(result)[1]

# Print the result
println("Optimization details: ", result)
println("Estimated parameter: ", β_hat)


# Plot the objective function around the minimimum
β_vec = collect(0.94:0.001:0.98)
obj_vec = zeros(length(β_vec))
for (i,β_i) in enumerate(β_vec)
    obj_vec[i] = smm_objective([β_i], ConsumptionData,ages,S, nothing,para)
end

# Plot 
gr();
plot(β_vec,obj_vec,label="Objective")
vline!([0.96],label="True β")
vline!([β_hat],label="Estimated β")
plot!(xlabel="Guess of β", ylabel="Objective function")
savefig("figtabs/Objective")

#############
# Optional Part: Estimation of variance of estimator

# Bootstrap the data with replacement
B = 1000 # Number of bootstrap replications
# To store bootstrap results
bootstrap_moments = zeros(B,N_mom)
# Perform bootstrapping
for b in 1:B
    for (i,age) in enumerate(ages)
    # Sample with replacement for each age group
    sample = ConsumptionData[rand(1:Ndata, Ndata),age]
    # Calculate mean consumption for each age in the bootstrap sample
    bootstrap_moments[b, i] = mean(sample)
    end
end
# Calculate the bootstrapped variance-covariance matrix of the moments in the data
Ω_hat = cov(bootstrap_moments)

# approximate Jacobian
h = 0.001
para.β = β_hat - h
M_h_minus = sim_moments(S,ages,Ndata,para)
para.β = β_hat + h
M_h_plus = sim_moments(S,ages,Ndata,para)
G_hat = - vec((M_h_minus-M_h_plus)/(2*h))

# Now, V is the computed variance-covariance matrix
V_hat = inv(G_hat' * W * G_hat) * G_hat' * W * Ω_hat * W' * G_hat * inv(G_hat' * W * G_hat)


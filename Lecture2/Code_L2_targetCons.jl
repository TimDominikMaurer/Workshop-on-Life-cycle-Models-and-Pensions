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
	
	# Simulation
	Nsim::Int = 1000                                      # Number of simulated agents
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

# We simulate the model with the "true" parameters.
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
function sim_moments(S,ages,Ndata)
	# This functions simulates the moments on S simulations
    # Ndata: Number of individuals in the data
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
function moments_diff(data,ages,S)
	# This function calculates the error vectors
    
    Ndata,_ = size(data) # Get the number of individuals in the data
    # Data moments
    # stack them into a vector
    Λᵈ= mean(data[:,ages],dims=1)
    
    # Simulate moments
    Λˢ =  sim_moments(S,ages,Ndata)
    
	return vec(Λᵈ - Λˢ)
end
############# 
# Part E: Wrtie a function for the objective function
function smm_objective(Θ,data,ages,S,W)

    # set parameter
    para.β = Θ[1]
    # para.ρ = Θ[2]
        
    # If W is not provided (i.e., is nothing), default to identity matrix
    if W === nothing
        W = Diagonal(ones(length(ages)))  # Identity matrix in Julia
    end
    # Compute the difference between data and model moments
    Mdiff = moments_diff(data,ages,S)
    
    # Compute the quadratic form
    objective_value = Mdiff' * W * Mdiff
    
    return objective_value
end

# Optimize using Nelder-Mead
result = optimize(x -> smm_objective(x, ConsumptionData,ages,S, nothing), [0.90], NelderMead())

# Print the result
println("Estimated parameters: ", Optim.minimizer(result))
println("Optimization details: ", result)

optimize(x -> smm_objective(x, ConsumptionData,ages,S,nothing), [0.90], NelderMead()).minimizer


# Bootstrap the data with replacement
B = 1000 # Number of bootstrap replications
# To store bootstrap results
bootstrap_moments = zeros(B,N_mom)

rand(1:Ndata,Ndata)
# Perform bootstrapping
for b in 1:B
    # Sample with replacement for each age group
    for (i,age) in enumerate(ages)
    sample = ConsumptionData[rand(1:Ndata, Ndata),age]
    
    # Calculate mean consumption for each age in the bootstrap sample
    bootstrap_moments[b, i] = mean(sample)
    end
end

cov_moments = cov(bootstrap_moments)
var_moments = diag(cov_moments)
W_d = diagm(var_moments.^-1)

















function smm_objective2(Θ,data,ages,S,W)
    
    # set parameter
    para.β = Θ[1]
    para.ρ = Θ[2]
        
    # If W is not provided (i.e., is nothing), default to identity matrix
    if W === nothing
        W = Diagonal(ones(length(ages)))  # Identity matrix in Julia
    end
    # Compute the difference between data and model moments
    Mdiff = moments_diff(data,ages,S)
    
    # Compute the quadratic form
    objective_value = Mdiff' * W * Mdiff
    
    return objective_value
end
result = optimize(x -> smm_objective2(x, ConsumptionData,ages,S, W_d), [0.95;2.2], NelderMead())

# Print the result
println("Estimated parameters: ", Optim.minimizer(result))
println("Optimization details: ", result)

smm_objective([0.95,2.5],ConsumptionData,ages,S,W_d)
para.β


moments_diff(ConsumptionData,ages,S)

moments_diff(ConsumptionData,ages,S)'*Diagonal([1.0, 1.0, 1.0])*moments_diff(ConsumptionData,ages,S)


para.ρ = Optim.minimizer(result)[1]
Con1, A1 = solveLCM(para,0.0)

# The true parameters are
para.β = 0.98
para.ρ = 2.0
Con2, A2 = solveLCM(para,0)

plot(Con1)
plot!(Con2)
plot(A1)
plot!(A2)

ages = [1,5,10,15,21]


# Optimize using Nelder-Mead
result = optimize(x -> smm_objective(x, ConsumptionData,ages,S, nothing), [0.9], NelderMead())

# Print the result
println("Estimated parameters: ", Optim.minimizer(result))
println("Optimization details: ", result)
moments_diff(ConsumptionData,ages,S)'*Diagonal(ones(length(ages)))*moments_diff(ConsumptionData,ages,S)


para.ρ = Optim.minimizer(result)[1]
Con1, A1 = solveLCM(para,0.0)

# The true parameters are
para.β = 0.98
para.ρ = 2.0
Con2, A2 = solveLCM(para,0)

plot(Con1)
plot!(Con2)
plot(A1)
plot!(A2)

ages = [1,5,8,10,12,15,21]


# Optimize using Nelder-Mead
result = optimize(x -> smm_objective(x, ConsumptionData,ages,S, nothing), [0.9], NelderMead(),tol=0.0000001)

optimize()
# Print the result
println("Estimated parameters: ", Optim.minimizer(result))
println("Optimization details: ", result)
moments_diff(ConsumptionData,ages,S)'*Diagonal(ones(length(ages)))*moments_diff(ConsumptionData,ages,S)


para.ρ = Optim.minimizer(result)[1]
Con1, A1 = solveLCM(para,0.0)

# The true parameters are
para.β = 0.98
para.ρ = 2.0
Con2, A2 = solveLCM(para,0)

plot(Con1)
plot!(Con2)
plot(A1)
plot!(A2)
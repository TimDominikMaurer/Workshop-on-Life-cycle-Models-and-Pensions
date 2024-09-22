# Identification Problem

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
    r::Float64 = 0.04                                  # Gross interest rate after taxes
    w::Float64 = 1.0                                   # Wage

    # Preferences
    β::Float64 = 0.98                                  # Patience
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

# The true parameters are
para.β = 0.98
para.ρ = 2.0
# We simulate the model with the "true" parameters.
ConsumptionData,SavingData = SimLCM(para,10000)


# How many number of individuals are in the the data
Ndata,_ = size(ConsumptionData)

############# 
# Part C: Create the data moments

# Mean Consumption at age 5
Λᵈ₁ = mean(ConsumptionData[:,2])
Λᵈ₂ = mean(ConsumptionData[:,10])
Λᵈ₃ = mean(ConsumptionData[:,18])
# stack them into a vector
Λᵈ=[Λᵈ₁,Λᵈ₂,Λᵈ₃]


############# 
# Part C: Wrtie a function that simulates the moments
S = 100
function sim_moments(S,Ndata)
	# This functions simulates the moments on S simulations
    # Ndata: Number of individuals in the data
    
    # Set the seed
    Random.seed!(1)
    # Store moments per simulation in the following vectors
    Λˢ₁ₛ = zeros(S)
    Λˢ₂ₛ = zeros(S)
    Λˢ₃ₛ = zeros(S)
    for s in 1:S
        _, SimCon = SimLCM(para,Ndata)
        Λˢ₁ₛ[s] = mean(SimCon[:,2])
        Λˢ₂ₛ[s] = mean(SimCon[:,10])
        Λˢ₃ₛ[s] = mean(SimCon[:,18])
    end
    # take the average across the number of simulations S
    Λˢ₁ = mean(Λˢ₁ₛ)
    Λˢ₂ = mean(Λˢ₂ₛ)
    Λˢ₃ = mean(Λˢ₃ₛ)
    # stack them into a vector
    Λˢ = [Λˢ₁,Λˢ₂,Λˢ₃]
	return Λˢ
end

sim_moments(S,Ndata)



# write a fucntion for the difference
function moments_diff(data,S)
	# This function solves for the
    
    # Get the number of individuals in the data
    Ndata,_ = size(data)
    
    # Start with the data moments
    Λᵈ₁ = mean(data[:,2])
    Λᵈ₂ = mean(data[:,10])
    Λᵈ₃ = mean(data[:,18])
    # stack them into a vector
    Λᵈ=[Λᵈ₁,Λᵈ₂,Λᵈ₃]
    
    # Simulate moments
    Λˢ = sim_moments(S,Ndata)
    
	return Λᵈ - Λˢ
end

moments_diff(SavingData,S)

function smm_objective(Θ,data,S,W)
    
    # set parameter
    para.β = Θ[1]
    para.ρ = Θ[2]
    
    # If W is not provided (i.e., is nothing), default to identity matrix
    if W === nothing
        W = Diagonal([1.0, 1.0, 1.0])  # Identity matrix in Julia
    end
    # Compute the difference between data and model moments
    Mdiff = moments_diff(data,S)
    
    # Compute the quadratic form
    objective_value = Mdiff' * W * Mdiff
    
    return objective_value
end

# Optimize using Nelder-Mead
result = optimize(x -> smm_objective(x, SavingData,S, nothing), [0.9,2.1], NelderMead())

# Print the result
println("Estimated parameters: ", Optim.minimizer(result))
println("Optimization details: ", result)
moments_diff(SavingData,S)'*Diagonal([1.0, 1.0, 1.0])*moments_diff(SavingData,S)




para.β = Optim.minimizer(result)[1]
para.ρ = Optim.minimizer(result)[2]
Con1, A1 = solveLCM(para,0.0)

# The true parameters are
para.β = 0.98
para.ρ = 2.0
Con2, A2 = solveLCM(para,0)

plot(Con1)
plot!(Con2)
plot(A1)
plot!(A2)



smm_objective(Optim.minimizer(result),SavingData,S,nothing)
moments_diff(SavingData,S)
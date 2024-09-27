###### 
# Lecture 2: Structural Estimation using SMM

########## 
# Preamble

# Install and load required packages
using Pkg
# List of required packages
packages = ["Plots", "Parameters", "Distributions", "LinearAlgebra", "Optim"]
# Install any packages that are not already installed
for pkg in packages
    if !haskey(Pkg.installed(), pkg)
        Pkg.add(pkg)
    end
end

# Load installed packages
using Plots, Parameters, Distributions, LinearAlgebra, Optim

# Set working directory to current file location
cd(dirname(@__FILE__))

########## 
# Implementing the Structural Model

# Define model parameters using a mutable struct
@with_kw mutable struct set_para
    T::Int = 20                                        # Maximum lifespan
    Tᵣ::Int = 15                                       # Retirement age
    l::Vector{Float16} = vcat(ones(Tᵣ), zeros(T-Tᵣ))   # Labor supply vector: 1 before retirement, 0 after

    # Prices
    r::Float64 = 0.13                                  # Gross interest rate after taxes
    w::Float64 = 1.0                                   # Wage rate

    # Preferences
    β::Float64 = 0.96                                  # Discount factor (Patience)
    ρ::Float64 = 2.0                                   # Relative Risk Aversion (RRA) / Inverse Intertemporal Elasticity of Substitution (IES)

    # Distribution parameters
    μ::Float64 = 0.0                                   # Mean of Lognormal distribution
    σ::Float64 = 1.0                                   # Standard deviation of Lognormal distribution
end

# Create an instance of the parameter struct
para = set_para()

############# 
# Part A: Load functions from Lecture 1
include("Functions_Lecture1.jl")

############# 
# Part B: Generate synthetic data
# 1. Simulate the model with the "true" parameters.
# 2. Treat the simulated data as "empirical" data for estimation purposes.
# 3. The true parameters are known, so estimation should recover them.

# Set the true discount factor (β)
para.β = 0.96

# Set the seed for reproducibility
Random.seed!(1)

# Simulate consumption and savings data for 1000 individuals
ConsumptionData, SavingData = SimLCM(para, 1000)

# Get the number of individuals in the dataset
Ndata, _ = size(ConsumptionData)

############# 
# Part C: Compute data moments
# Target ages for calculating moments
ages = [5, 10, 15]

# Calculate moments from the simulated data
Λᵈ = mean(ConsumptionData[:, ages], dims=1)
N_mom = length(Λᵈ)  # Number of moments

############# 
# Part C: Function to simulate moments

S = 100 # Number of simulations

function sim_moments(S, ages, Ndata, para)
    """
    Simulate moments for the life-cycle model based on S simulations.
    
    Inputs:
    - S: Number of simulations.
    - ages: Vector of ages for which to compute moments.
    - Ndata: Number of individuals in the empirical data.
    - para: Model parameters.

    Output:
    - Λˢ: Simulated moments (average consumption at target ages).
    """
    Random.seed!(1)  # Set seed for reproducibility
    Λˢₛ = zeros(S, length(ages))  # Initialize matrix for simulated moments

    # Perform S simulations and calculate average consumption for each age
    for s in 1:S
        SimCon, _ = SimLCM(para, Ndata)
        Λˢₛ[s, :] = mean(SimCon[:, ages], dims=1)
    end
    
    # Return the average simulated moments across simulations
    return mean(Λˢₛ, dims=1)
end

############# 
# Part D: Function to calculate the error vector

function moments_diff(data, ages, S, para)
    """
    Compute the error vector (difference between empirical and simulated moments).
    
    Inputs:
    - data: Empirical data matrix.
    - ages: Target ages for moments.
    - S: Number of simulations.
    - para: Model parameters.
    
    Output:
    - Error vector: Difference between empirical moments (Λᵈ) and simulated moments (Λˢ).
    """
    Ndata, _ = size(data)  # Number of individuals in empirical data

    # Calculate empirical moments from data
    Λᵈ = mean(data[:, ages], dims=1)

    # Simulate moments based on model parameters
    Λˢ = sim_moments(S, ages, Ndata, para)

    # Return the error vector
    return vec(Λᵈ - Λˢ)
end

############# 
# Part E: Function for the SMM objective function

function smm_objective(Θ, data, ages, S, W, para)
    """
    Compute the SMM objective function to be minimized (quadratic form).
    
    Inputs:
    - Θ: Parameter vector (to be estimated, includes β).
    - data: Empirical data matrix.
    - ages: Target ages for moments.
    - S: Number of simulations.
    - W: Weighting matrix (if not provided, defaults to identity).
    - para: Model parameters.
    
    Output:
    - Objective value to be minimized (quadratic form of error vector).
    """
    # Update the model parameter (β)
    para.β = Θ[1]

    # Use identity matrix as default weighting if none is provided
    if W === nothing
        W = Diagonal(ones(length(ages)))
    end

    # Compute the error vector between data and model moments
    e = moments_diff(data, ages, S, para)

    # Return the quadratic form of the error vector
    return e' * W * e
end

############# 
# Optimization using Nelder-Mead
result = optimize(beta -> smm_objective(beta, ConsumptionData, ages, S, nothing, para), [0.90], NelderMead())
β_hat = Optim.minimizer(result)[1]  # Retrieve the estimated parameter (β)

# Output results
println("Optimization details: ", result)
println("Estimated parameter: ", β_hat)

############# 
# Plot the objective function around the estimated minimum
β_vec = collect(0.94:0.001:0.98)
obj_vec = zeros(length(β_vec))

# Evaluate the objective function for a range of β values
for (i, β_i) in enumerate(β_vec)
    obj_vec[i] = smm_objective([β_i], ConsumptionData, ages, S, nothing, para)
end

# Plot the results
gr()
plot(β_vec, obj_vec, label="Objective")
vline!([0.96], label="True β")
vline!([β_hat], label="Estimated β")
plot!(xlabel="β guess", ylabel="Objective function")
savefig("figtabs/Objective")

############# 
# Optional: Bootstrap estimation of variance of the estimator

# Bootstrap the data with replacement
B = 1000  # Number of bootstrap replications
bootstrap_moments = zeros(B, N_mom)  # Storage for bootstrap moments

# Perform bootstrapping to estimate the variance-covariance matrix of the moments in the data
for b in 1:B
    for (i, age) in enumerate(ages)
        sample = ConsumptionData[rand(1:Ndata, Ndata), age]  # Resample data
        bootstrap_moments[b, i] = mean(sample)  # Calculate mean for each bootstrap sample
    end
end

# Estimate variance-covariance matrix of the moments in the data
Ω_hat = cov(bootstrap_moments)

# Approximate the Jacobian matrix numerically
h = 0.0001
para.β = β_hat - h
M_h_minus = sim_moments(S, ages, Ndata, para)
para.β = β_hat + h
M_h_plus = sim_moments(S, ages, Ndata, para)
G_hat = -vec((M_h_minus - M_h_plus) / (2 * h))

# Define weighting matrix (identity matrix)
W = Diagonal(ones(length(ages)))

# Estimate variance-covariance matrix of the estimator
V_hat = inv(G_hat' * W * G_hat) * G_hat' * W * Ω_hat * W' * G_hat * inv(G_hat' * W * G_hat)

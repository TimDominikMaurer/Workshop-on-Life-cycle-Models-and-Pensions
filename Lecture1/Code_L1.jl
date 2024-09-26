###### 
# Lecture 1: The Life-cycle Model

##########
# Preamble

# Install and load the required packages
using Pkg
# List of required packages
packages = ["Plots", "Parameters", "Distributions", "Random"]
# Loop through the packages and install those that are not already installed
for pkg in packages
    if !haskey(Pkg.installed(), pkg)
        Pkg.add(pkg)
    end
end

# Load the installed packages
using Plots, Parameters, Distributions, Random

# Set the working directory to the current file's location
cd(dirname(@__FILE__))

##########
# Implementing the model

# Calibrate the model parameters and store them in a mutable struct
@with_kw mutable struct set_para
    # Time parameters
    T::Int = 20                                        # Total time periods (maximum age)
    Tᵣ::Int = 15                                       # Retirement age (start of retirement)
    l::Vector{Float16} = vcat(ones(Tᵣ), zeros(T-Tᵣ))   # Exogenous labor supply: 1 before retirement, 0 after

    # Prices
    r::Float64 = 0.13                                  # Gross interest rate (after taxes)
    w::Float64 = 1.0                                   # Wage rate

    # Preferences
    β::Float64 = 0.96                                  # Discount factor (patience)
    ρ::Float64 = 2.0                                   # Relative risk aversion (inverse elasticity of substitution)

    # Distribution parameters
    μ::Float64 = 0.0                                   # Mean of the Lognormal distribution
    σ::Float64 = 1.0                                   # Standard deviation of the Lognormal distribution
end

# Create an instance of the parameter struct
para = set_para()

# Example access of parameters
println("Discount factor (Patience): ", para.β)
println("Labor supply vector: ", para.l)

# Update parameter values (and revert to the original)
para.β = 0.9
para.β = 0.96

###########
# Part A: Solve for initial consumption (c₁) given initial assets (a₀)

# A.1: Solve for X in equation (9)
# Vectorize labor supply (l) and divide by the cumulative product of (1 + r)
print(para.l)  # Output the labor supply vector
# Create a sequence from 1 to T for the powers of (1 + r)
collect(1:para.T)
(1 + para.r) .^ collect(1:para.T)  # Raise (1 + r) to powers 1 through T

# Evaluate X as in equation (9)
X = para.w * sum(para.l ./ ((1 + para.r) .^ collect(1:para.T)))

# Define a function to solve for initial consumption (c₁) given initial assets (a₀)
function c1i(para, a0i)
    """
    Solves for initial consumption (c₁) given initial assets (a₀).
    
    Inputs:
    - para: Struct with model parameters (T, l, r, w, β, ρ).
    - a0i: Initial assets (savings) of agent i.
    
    Output:
    - ci1: Initial consumption level (c₁) for agent i.
    """
    @unpack T, l, r, w, β, ρ = para
    X = w * sum(l ./ ((1 + r) .^ collect(1:T)))
    Yi = a0i
    Z = sum((β * (1 + r)) .^ ((collect(1:T) .- 1) / ρ) ./ (1 + r) .^ collect(1:T))
    ci1 = (X + Yi) / Z
    return ci1
end

# Save an instance of initial assets
a0i = 1.0

# Call the function to compute c₁
c1i(para, a0i)

###########
# Part B: Solve for the entire consumption path over the life cycle

# Vectorize the long-run Euler equation from equaiton (10)
LRE = (para.β * (1 + para.r)) .^ ((collect(1:para.T) .- 1) / para.ρ)

# Solve for the consumption path given an initial asset level as in equaiton (10)
C = c1i(para, a0i) .* LRE

# Alternative method: solve the consumption path using a loop and the short-run Euler equation (5)
Cloop = zeros(para.T)  # Initialize consumption vector
Cloop[1] = c1i(para, a0i)  # Set initial consumption
# Loop through periods to compute future consumption
for i in 1:para.T-1
    Cloop[i+1] = (para.β * (1 + para.r)) ^ (1 / para.ρ) * Cloop[i]
end

# Check if both methods give the same result (small numerical difference is allowed)
is_equal = all(abs.(Cloop - C) .< 1e-12)
println("Both techniques give the same result: ", is_equal)

###########
# Part C: Solve for the savings path over the life cycle

# Initialize the savings vector
A = zeros(para.T + 1)
A[1] = a0i  # Initial assets at birth

# Compute the savings path using the budget constraint
for t in 1:para.T
    A[t+1] = para.w * para.l[t] + (1 + para.r) * A[t] - C[t]
end

# Plot the results: consumption and savings over the life cycle
plot(0:para.T, [NaN; C], label="Consumption")
plot!(0:para.T, A, label="Savings")
plot!(xlabel="Age", ylabel="Consumption / Savings")
plot!(legend=:topleft)


###########
# Part D: Write a function to solve for both consumption and savings paths

function solveLCM(para, a0i)
    """
    Solves the life-cycle model for an agent, returning consumption and savings paths.
    
    Inputs:
    - para: Struct with model parameters (T, l, r, w, β, ρ).
    - a0i: Initial assets (savings) of agent i.
    
    Outputs:
    - C: Vector of consumption levels over the life-cycle.
    - A: Vector of savings levels over the life-cycle.
    """
    @unpack T, l, r, w, β, ρ = para
    LRE = (β * (1 + r)) .^ ((collect(1:T) .- 1) / ρ)
    C = c1i(para, a0i) .* LRE
    A = zeros(T + 1)
    A[1] = a0i
    for t in 1:T
        A[t + 1] = w * l[t] + (1 + r) * A[t] - C[t]
    end
    return C, A
end

# Call the function
C, A = solveLCM(para, a0i)

# Plot the results: consumption and savings over the life cycle
plot(0:para.T, [NaN; C], label="Consumption")
plot!(0:para.T, A, label="Savings")
plot!(xlabel="Age", ylabel="Consumption / Savings")
plot!(legend=:topleft)

###########
# Part D: Simulate the model for multiple agents

function SimLCM(para, Nsim)
    """
    Simulates the life-cycle model for multiple agents.
    
    Inputs:
    - para: Struct with model parameters, including (μ, σ, T).
    - Nsim: Number of agents (simulations).
    
    Outputs:
    - Csim: Matrix of consumption paths for Nsim agents.
    - Asim: Matrix of savings paths for Nsim agents.
    """
    @unpack μ, σ, T = para
    a0i_sim = rand(LogNormal(μ, σ), Nsim)
    Csim = zeros((Nsim, T))
    Asim = zeros((Nsim, T + 1))
    for i in 1:Nsim
        Csim[i, :], Asim[i, :] = solveLCM(para, a0i_sim[i])
    end
    return Csim, Asim
end

# Set the seed for reproducibility
Random.seed!(1)

# Call the simulation function
Nsim = 3
Csim, Asim = SimLCM(para, Nsim)

# Plot the simulation results for multiple agents
gr()
plot(0:para.T, hcat(fill(NaN, Nsim), Csim)', color=[:red :blue :green], label=nothing)
plot!(0:para.T, Asim', color=[:red :blue :green], line=:dash, label=nothing)
plot!(xlabel="Age", ylabel="Consumption / Savings")
plot!(legend=:topleft)

# Add dummy lines to simulate a legend for line styles
plot!(0:para.T, fill(NaN, para.T + 1), label="Consumption", color=:black, linestyle=:solid)
plot!(0:para.T, fill(NaN, para.T + 1), label="Savings", color=:black, linestyle=:dash)

# Add dummy lines to simulate a legend for agent colors
plot!(0:para.T, fill(NaN, para.T + 1), label="Agent 1", color=:blue, linestyle=:solid)
plot!(0:para.T, fill(NaN, para.T + 1), label="Agent 2", color=:red, linestyle=:solid)
plot!(0:para.T, fill(NaN, para.T + 1), label="Agent 3", color=:green, linestyle=:solid)

# Save the plot with high resolution
plot!(dpi=600, size=(600, 400))
savefig("figtabs/Simulation")


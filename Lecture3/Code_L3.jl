####
# Lecture 3: The OLG Model

##########
# Preamble

# Install and load the required packages
using Pkg
packages = ["Plots", "Parameters", "Optim"]
for pkg in packages
    if !haskey(Pkg.installed(), pkg)
        Pkg.add(pkg)
    end
end

using Plots, Parameters, Optim

# Set the working directory to the current file's directory
cd(dirname(@__FILE__))

##########
# Implementing the OLG model

# Define the parameters of the model
@with_kw mutable struct set_para
    # Timing parameters
    T::Int = 20          # Maximum age
    Tᵣ::Int = 15         # Retirement age
	l::Vector{Float16} = vcat(ones(Tᵣ), zeros(T-Tᵣ))  # Labor supply (1 before retirement, 0 after)
	L::Float64 = sum(l)   # Aggregate labor supply

    # Preferences
    β::Float64 = 0.96      # Discount factor (patience)
    ρ::Float64 = 2.0      # Relative risk aversion (inverse of intertemporal elasticity of substitution)
	
	# Production technology
	α::Float64 = 1/3      # Capital's share of income
    δ::Float64 = 0.07     # Depreciation rate (yearly)
end

# Create an instance of the parameter struct
para = set_para()

###########
# Part A: Function to solve for consumption, savings, and capital given a guess for capital
function solve(K_guess, para)
	"""
	This function solves for consumption, savings, and implied aggregate capital based on a given guess for capital.

	Parameters:
	- K_guess: Initial guess for capital.
	- para: Struct containing the model parameters.

	Returns:
	- C: Consumption path for each period.
	- A: Savings path for each period.
	- K_implied: Implied aggregate capital based on savings.
	"""
	
	@unpack T, l, β, ρ, L, α, δ = para  # Unpack parameters
	
	# Step 2: Compute the interest rate and wage from the capital guess
	r = α * (K_guess / L)^(α - 1) - δ  # Interest rate
	w = (1 - α) * (K_guess / L)^α       # Wage rate

	# Step 3: Solve for first-period consumption using the Euler equation
	c1 = (w * sum(l ./ ((1 + r) .^ collect(1:T)))) / sum((β * (1 + r)) .^ ((collect(1:T) .- 1) / ρ) ./ (1 + r) .^ collect(1:T))

	# Step 4: Solve for the entire consumption path using the Euler equation
    C = c1 * (β * (1 + r)) .^ ((collect(1:T) .- 1) / ρ)

	# Step 5: Solve for the savings path using the budget constraint
    A = zeros(T+1)  # Initialize savings (agents are born with zero savings)
	for j in 1:T    # Loop through periods
		A[j+1] = w * l[j] + (1 + r) * A[j] - C[j]  # Budget constraint for savings
	end
	
	# Step 6: Calculate implied aggregate capital by summing savings
    K_implied = sum(A)
	
	return C, A, K_implied
end

###########
# Part B: Objective function to minimize the difference between K_guess and K_implied
function objective(K_guess, para)
	"""
	Objective function that computes the squared difference between the guessed capital and implied capital.
	Used for finding the steady-state capital.

	Parameters:
	- K_guess: Initial guess for capital.
	- para: Struct containing the model parameters.

	Returns:
	- loss: Squared difference between K_guess and K_implied.
	"""
	C, A, K_implied = solve(K_guess, para)
	
	# Ensure that implied capital is positive
	if K_implied < 0 
		K_implied = 0.001
	end
	
	# Step 8: Compute loss as the squared difference between K_guess and K_implied
    loss = (K_guess - K_implied)^2
    return loss
end

###########
# Part C: Solve for the steady state

# Set up bounds for the capital guess
r_lb = 0.01 - para.δ  # Lower bound for interest rate
K_ub = para.L * ((r_lb + para.δ) / para.α)^(1 / (para.α - 1))  # Upper bound for capital
println("Upper bound of K: ", K_ub)

# Minimize the objective function to find the steady-state capital
result = optimize(K -> objective(K, para), 0.0, K_ub)
println(result)

# Extract the steady-state capital (K_ss)
K_ss = Optim.minimizer(result)
println("Steady-state capital (K_ss): ", K_ss)

###########
# Part D: Characterize the steady-state

# Calculate steady-state factor prices
r_ss = para.α * (K_ss / para.L)^(para.α - 1) - para.δ  # Interest rate
w_ss = (1 - para.α) * (K_ss / para.L)^para.α           # Wage rate

# Convert interest rate to yearly terms if one model period equals 4 years
r_ss_yearly = (1 + r_ss)^(1/4) - 1

# Solve for the steady-state consumption and savings paths
C_ss, A_ss, K_ss = solve(K_ss, para)

# Plot steady-state consumption and savings paths
gr()
plot(0:para.T, [NaN; C_ss], label="Consumption")
plot!(0:para.T, A_ss, label="Savings")
plot!(xlabel="Age", ylabel="Consumption/Savings")
plot!(legend=:topleft)

# Add the steady-state values to the plot title
plot!(title="K_ss = $(round(K_ss, digits=3)), r_ss = $(round(r_ss, digits=3)), w_ss = $(round(w_ss, digits=3))")
savefig("figtabs/SS_result")

############################
# Part D: Adding pensions

# Function to solve the model with a PAYG pension scheme
function solve_paygo(K_guess, τ, para)
	"""
	This function solves for consumption, savings, and implied aggregate capital given a guess for capital
	in a pay-as-you-go (PAYG) pension system.

	Parameters:
	- K_guess: Initial guess for capital.
	- τ: Contribution rate to the PAYG system.
	- para: Struct containing the model parameters.

	Returns:
	- C: Consumption path for each period.
	- A: Savings path for each period.
	- K_implied: Implied aggregate capital based on savings.
	"""
	
	@unpack T, Tᵣ, l, β, ρ, L, α, δ = para  # Unpack parameters
	
	# Step 2: Compute interest rate and wage from the capital guess
	r = α * (K_guess / L)^(α - 1) - δ  # Interest rate
	w = (1 - α) * (K_guess / L)^α       # Wage rate

	# Step 3: Solve for first-period consumption, accounting for PAYG benefits
	b = τ * w * L / (T - Tᵣ)  # PAYG pension benefit (balanced-budget condition)
	c1 = ((1 - τ) * w * sum(l ./ ((1 + r) .^ collect(1:T))) + b * sum((1 .- l) ./ ((1 + r) .^ collect(1:T)))) / sum((β * (1 + r)) .^ ((collect(1:T) .- 1) / ρ) ./ (1 + r) .^ collect(1:T))

	# Step 4: Solve for the entire consumption path using the Euler equation
	C = c1 * (β * (1 + r)) .^ ((collect(1:T) .- 1) / ρ)

	# Step 5: Solve for the savings path using the budget constraint
	A = zeros(T+1)  # Initialize savings
	for j in 1:T
		A[j+1] = (1 - τ) * w * l[j] + (1 - l[j]) * b + (1 + r) * A[j] - C[j]  # Budget constraint for savings
	end
	
	# Step 6: Calculate implied aggregate capital
	K_implied = sum(A)
	
	return C, A, K_implied
end

###########
# Adjust the objective function to include the PAYG system
function objective_paygo(K_guess, τ, para)
	"""
	Objective function for the PAYG system that computes the squared difference between guessed capital
	and implied capital, used for finding the steady-state capital.

	Parameters:
	- K_guess: Initial guess for capital.
	- τ: Contribution rate to the PAYG system.
	- para: Struct containing the model parameters.

	Returns:
	- loss: Squared difference between K_guess and K_implied.
	"""
	
	C, A, K_implied = solve_paygo(K_guess, τ, para)
	
	# Ensure that implied capital is positive
	if K_implied < 0 
		K_implied = 0.001
	end
	
	# Step 8: Compute loss as the squared difference between K_guess and K_implied
    loss = (K_guess - K_implied)^2
    return loss
end




# Set the pension contribution rate
τ = 0.1

# Minimize the objective function with the PAYG system to find the steady-state capital
result = optimize(K -> objective_paygo(K, τ, para), 0.0, K_ub)
println(result)

# Extract the steady-state capital for the PAYG system
K_ss_paygo = Optim.minimizer(result)

# Calculate the steady-state interest rate and wage for the PAYG system
r_ss_paygo = para.α * (K_ss_paygo / para.L)^(para.α - 1) - para.δ
w_ss_paygo = (1 - para.α) * (K_ss_paygo / para.L)^para.α

# Solve for steady-state consumption and savings with the PAYG system
C_ss_paygo, A_ss_paygo, K_ss_paygo = solve_paygo(K_ss_paygo, τ, para)

# Welfare calculations for the PAYG system
function crra_utility(c, ρ)
    if ρ == 1
        return log(c)
    else
        return (c^(1 - ρ) - 1) / (1 - ρ)
    end
end

function welfare(C, para)
	"""
	Compute the total welfare from the consumption path.

	Parameters:
	- C: Consumption path.
	- para: Struct containing model parameters.

	Returns:
	- total_welfare: Discounted total welfare.
	"""
	
	@unpack T, β, ρ = para
    total_welfare = 0.0
    for t in 1:T
        # Compute CRRA utility for each period's consumption
        u = crra_utility(C[t], ρ)
        # Sum up discounted utilities
        total_welfare += β^t * u
    end
    return total_welfare
end

# Calculate welfare for the steady-state
wel_ss = welfare(C_ss, para)
wel_ss_paygo = welfare(C_ss_paygo, para)

# Plot the steady-state results with and without a PAYG system
gr()
plot(0:para.T, [NaN; C_ss], label="Consumption")
plot!(0:para.T, A_ss, label="Savings")
plot!(0:para.T, [NaN; C_ss_paygo], label="Consumption (PAYG)")
plot!(0:para.T, A_ss_paygo, label="Savings (PAYG)")
plot!(xlabel="Age", ylabel="Consumption/Savings")
plot!(legend=:topleft)

# Add the differences between PAYG and non-PAYG results to the plot title
plot!(
    title = "K_ss_diff = $(round(K_ss_paygo-K_ss, digits=3)), r_ss_diff = $(round(r_ss_paygo-r_ss, digits=3)), w_ss_diff = $(round(w_ss_paygo-w_ss, digits=3)), Wel_ss_diff = $(round(wel_ss_paygo-wel_ss, digits=3))",
    titlefont = 9
)
savefig("figtabs/SS_result_comp")


###################
# Dynamic inefficiency

# Recalibrate
para.β = 1.1 # we increase the desire to save
para.α = 1/6 # the interest rate drops faster as captial/savings increase

# Minimize the objective function without pensions
K_ss_in = optimize(K -> objective_paygo(K, 0.0, para), 0.0, K_ub).minimizer

# Calculate the steady-state interest rate without pensions
r_ss_in = para.α * (K_ss_in / para.L)^(para.α - 1) - para.δ

# Solve for steady-state consumption and savings without the PAYG system
C_ss_in, A_ss_in, K_ss_in = solve_paygo(K_ss_in, 0.0, para)

# Minimize the objective function with pensions
K_ss_in_paygo = optimize(K -> objective_paygo(K, τ, para), 0.1, K_ub).minimizer

# Calculate the steady-state interest rate with pensions
r_ss_in_paygo = para.α * (K_ss_in_paygo / para.L)^(para.α - 1) - para.δ

# Solve for steady-state consumption and savings with the PAYG system
C_ss_in_paygo, A_ss_in_paygo, K_ss_in_paygo = solve_paygo(K_ss_in_paygo, 0.0, para)

# Can PAYG pension improve welfare now?
welfare(C_ss_in_paygo, para)>welfare(C_ss_in, para)



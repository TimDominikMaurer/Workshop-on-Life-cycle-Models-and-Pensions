# Functions from lecture 3
using Plots, Parameters, Optim
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
	for t in 1:T
		A[t+1] = (1 - τ) * w * l[t] + (1 - l[t]) * b + (1 + r) * A[t] - C[t]  # Budget constraint for savings
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
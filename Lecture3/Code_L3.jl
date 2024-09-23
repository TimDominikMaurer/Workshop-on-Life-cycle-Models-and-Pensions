####
# Lecture 3: The OLG Model


using Plots, Parameters, Distributions

# working directory is the current file
cd(dirname(@__FILE__))

# Define the Parameter in a mutable construct
@with_kw mutable struct set_para
    # Timing parameters
    T::Int = 20                                        # Maximum age of life
    Tᵣ::Int = 15                                       # Retirement age
	l::Vector{Float16} = vcat(ones(Tᵣ), zeros(T-Tᵣ))     # Exogenous labour supply
	L::Float64 = sum(l)                                  # Aggreage labour supply

    # Preferences
    β::Float64 = 0.98                                  # Patience
    ρ::Float64 = 2.0                                   # Relative Risk Aversion (RRA) / Inverse IES
	
	# Prodcution technology
	α::Float64 = 0.3                                   # Output elasticities of capital
    δ::Float64 = 0.07                                   # yearly depreciation rate
	
end

# Create an instance of the struct
para = set_para()

###########
# Part A: Write a function that solves for consumption, savings and aggregate capital given a guess of capital
function solve(K_guess,para)
	# This function solves consumption, savings and aggregate capital given a guess of capital
	# K_guess: Guess of capital
	@unpack T,l,β,ρ,L,α,δ = para # unpack parameters
	########
	# STEP 2: Solve for wage given guessed interest rate
	r = α * (K_guess / L)^(α - 1) - δ # See equation (4)
	w = (1 - α) * (K_guess / L)^α # See equation (4) 
	########
	# STEP 3: Solve for first periond consumption as in equations (4) an (5)
	c1 = (w*sum(l./((1+r).^collect(1:T))))/sum((β*(1+r)).^((collect(1:T).-1)/ρ)./(1+r).^collect(1:T))
	########
	# STEP 4: Solve for the whole consumption path using the Long-Run Euler equation (2)
    C = c1*(β*(1+r)).^((collect(1:T).-1)/ρ)
	########
	# STEP 5: Solve for the whole saviongs path using the budget constraint
    A = zeros(T+1) # Initialize savings vector
	A[1] = 0  # Agents are born with no savings
	# Solve the whole savings path using the budget constraint in equation (7)
	for t in 1:(T)  # Loop from the second period onward
		A[t+1] = w * l[t] + (1 + r) * A[t] - C[t]
	end
	########
	# STEP 6: Compute implied aggregate capital by summing over savings path
    K_implied = sum(A)
	
	return C,A,K_implied
end

###########
# Part B: Define a los function that we can optimize
function objective(K_guess,para)
	C,A,K_implied = solve(K_guess,para) 
    # STEP 8: Check distance between K_guess - K_implied (Note we define the loss as the squared difference)
    loss = (K_guess - K_implied)^2  
    return loss
end

###########
# Part C: Discipline capital guess
r_lb = 0.01 - para.δ  # lower bound interest rate
K_ub = L * ((r_lb + para.δ) / para.α)^(1 / (para.α - 1))  # upper bound K(r_lb)
println("    Upper bound of K: ", K_ub)

####
# Lecture 3: The OLG Model

##########
# Preamble

# start by installing and loading required packages
using Pkg
# List of required packages
packages = ["Plots", "Parameters", "Optim"]
# Loop through and only add packages that are not installed
for pkg in packages
    if !haskey(Pkg.installed(), pkg)
        Pkg.add(pkg)
    end
end

using Plots, Parameters, Optim

# working directory is the current file
cd(dirname(@__FILE__))

##########
# Implementing the OLG model

# Calibrate the parameter and store them in a mutable construct
@with_kw mutable struct set_para
    # Timing parameters
    T::Int = 20                                        # Maximum age of life
    Tᵣ::Int = 15                                       # Retirement age
	l::Vector{Float16} = vcat(ones(Tᵣ), zeros(T-Tᵣ))     # Exogenous labour supply
	L::Float64 = sum(l)                                  # Aggreage labour supply

    # Preferences
    β::Float64 = 0.95                                  # Patience
    ρ::Float64 = 2.0                                   # Relative Risk Aversion (RRA) / Inverse IES
	
	# Prodcution technology
	α::Float64 = 1/3                                   # Output elasticities of capital
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
	w = (1 - α) * (K_guess / L)^α # See equation (5) 
	########
	# STEP 3: Solve for first periond consumption as in equation (6)
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
	
	# ensure that K_implied is positive
	if K_implied<0 
		K_implied=0.001
	end
    # STEP 8: Check distance between K_guess - K_implied (Note we define the loss as the squared difference)
    loss = (K_guess - K_implied)^2  
    return loss
end


###########
# Part C: Solve for the steady state

# Discipline capital guess
r_lb = 0.01 - para.δ  # lower bound interest rate
K_ub = para.L * ((r_lb + para.δ) / para.α)^(1 / (para.α - 1))  # upper bound K(r_lb)
println("    Upper bound of K: ", K_ub)


# Minizing function using bounded opitmization with bounds (0,K_ub)
result = optimize(K -> objective(K,para), 0.0,K_ub)
print(result)
# Extract steady-state capital K_SS
K_ss = Optim.minimizer(result)
println("Optimal x: ", K_ss)

###########
# Part D: Characterize the Steady State

# What are the steady-state factor prices?
r_ss = para.α * (K_ss / para.L)^(para.α - 1) - para.δ # See equation (4)
w_ss = (1 - para.α) * (K_ss / para.L)^para.α # See equation (5) 

# if one model period equals 4 years the yearly interest rate is:
r_ss_yearly = (1+r_ss)^(1/4)-1

# Solve for steady-state consumption and saving
C_ss,A_ss,K_ss = solve(K_ss,para)


gr();
plot(0:para.T,[NaN;C_ss],label="Consumption")
plot!(0:para.T,A_ss,label="Savings")
plot!(xlabel="Age", ylabel="Consumption/Savings")
plot!(legend=:topleft)

# Add the variables to the title with rounding to 3 decimal places
plot!(title="K_ss = $(round(K_ss, digits=3)), r_ss = $(round(r_ss, digits=3)), w_ss = $(round(w_ss, digits=3))")
savefig("figtabs/SS_result")


############################
# Part D: Adding pensions

# Add pension scheme to the function that solves the model given a guess for capital
function solve_paygo(K_guess,τ,para)
	# This function solves consumption, savings and aggregate capital given a guess of capital
	# with a pay-as-you-go pension scheme
	# K_guess: Guess of capital
	# τ: PAYG contribution rate
	@unpack T,Tᵣ,l,β,ρ,L,α,δ = para # unpack parameters
	
	########
	# STEP 2: Solve for wage given guessed interest rate
	r = α * (K_guess / L)^(α - 1) - δ # See equation (4)
	w = (1 - α) * (K_guess / L)^α # See equation (5) 
	########
	# STEP 3: Solve for first periond consumption as in equation (15)
	b = τ*w*L/(T-Tᵣ) # first calculate the benefits based on the balanced-bugdet condition
	# then solve for c1
	c1 = ((1-τ)*w*sum(l./((1+r).^collect(1:T)))+b*sum((1 .-l)./((1+r).^collect(1:T))))/sum((β*(1+r)).^((collect(1:T).-1)/ρ)./(1+r).^collect(1:T))
	########
	# STEP 4: Solve for the whole consumption path using the Long-Run Euler equation (2)
    C = c1*(β*(1+r)).^((collect(1:T).-1)/ρ)
	########
	# STEP 5: Solve for the whole saviongs path using the budget constraint
    A = zeros(T+1) # Initialize savings vector
	A[1] = 0  # Agents are born with no savings
	# Solve the whole savings path using the budget constraint in equation (11)
	for t in 1:(T)  # Loop from the second period onward
		A[t+1] = (1-τ)*w * l[t] + (1-l[t])*b + (1 + r) * A[t] - C[t]
	end
	########
	# STEP 6: Compute implied aggregate capital by summing over savings path
    K_implied = sum(A)
	
	return C,A,K_implied
end

###########
# adjust the the objective function
function objective_paygo(K_guess,τ,para)
	C,A,K_implied = solve_paygo(K_guess,τ,para)
	
	# ensure that K_implied is positive
	if K_implied<0 
		K_implied=0.001
	end
    # STEP 8: Check distance between K_guess - K_implied (Note we define the loss as the squared difference)
    loss = (K_guess - K_implied)^2  
    return loss
end

# set the pension contribution rate
τ=0.1

# Minizing function using bounded opitmization with bounds (0,K_ub)
result = optimize(K -> objective_paygo(K,τ,para), 0.0,K_ub)
print(result)
# Extract steady-state capital K_SS
K_ss_paygo = Optim.minimizer(result)
# What are the steady-state factor prices?
r_ss_paygo = para.α * (K_ss_paygo / para.L)^(para.α - 1) - para.δ # See equation (4)
w_ss_paygo = (1 - para.α) * (K_ss_paygo / para.L)^para.α # See equation (5) 
# Solve for steady-state consumption and saving
C_ss_paygo,A_ss_paygo,K_ss_paygo = solve_paygo(K_ss_paygo,τ,para)

# caluclate steady-state welfare with and without a pension scheme 
# Define the CRRA utility function
function crra_utility(c, ρ)
    if ρ == 1
        return log(c)
    else
        return (c^(1 - ρ)-1) / (1 - ρ)
    end
end
# Define the welfare function
function welfare(C,para)
	# C: vector of conusmption path
	@unpack T,β,ρ = para # unpack parameters
    total_welfare = 0.0
    for t in 1:T
        # Calculate the CRRA utility for consumption at time t
        u = crra_utility(C[t], ρ)
        # Discount the utility and sum it up
        total_welfare += β^t * u
    end
    return total_welfare
end

# calculate welfare using the functions above
wel_ss = welfare(C_ss,para)
wel_ss_paygo = welfare(C_ss_paygo,para) # welfare goes down

# plot the result
gr();
plot(0:para.T,[NaN;C_ss],label="Consumption")
plot!(0:para.T,A_ss,label="Savings")
plot!(0:para.T,[NaN;C_ss_paygo],label="Consumption (PAYG)")
plot!(0:para.T,A_ss_paygo,label="Savings (PAYG)")
plot!(xlabel="Age", ylabel="Consumption/Savings")
plot!(legend=:topleft)

# Add the variables to the title with rounding to 3 decimal places
plot!(
    title = "K_ss_diff = $(round(K_ss_paygo-K_ss, digits=3)), r_ss_diff = $(round(r_ss_paygo-r_ss, digits=3)), w_ss_diff = $(round(w_ss_paygo-w_ss, digits=3)), Wel_ss_diff = $(round(wel_ss_paygo-wel_ss, digits=3))",
    titlefont = 9  # Change the number to adjust the font size
)
savefig("figtabs/SS_result_comp")



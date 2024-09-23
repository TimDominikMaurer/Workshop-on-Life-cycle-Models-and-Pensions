using Plots, Parameters, Distributions, Random

# working directory is the current file
cd(dirname(@__FILE__))

# Define the Parameter in a mutable construct
@with_kw mutable struct set_para
    # Timing parameters
    T::Int = 20                                        # Maximum age of life
    Tᵣ::Int = 15                                       # Retirement age
	l::Vector{Float16} = vcat(ones(Tᵣ), zeros(T-Tᵣ))     # Exogenous labour supply

    # Prices
    r::Float64 = 0.04                                  # Gross interest rate after taxes
    w::Float64 = 1.0                                   # Wage

    # Preferences
    β::Float64 = 0.98                                  # Patience
    ρ::Float64 = 2.0                                   # Relative Risk Aversion (RRA) / Inverse IES
	
	# Distribution
    μ::Float64 = 0.0                                   # Location of Lognormal
    σ::Float64 = 1.0                                   # Scale of Lognormal
end

# Create an instance of the struct
para = set_para()


# Access example
println("Patience: ", para.β)
println("Labour supply: ", para.l)

# Change calibration
para.β = 0.9
para.β
# and back
para.β = 0.95

para.r =0.11947891466173288
para.w = 0.8842349350880873

###########
# Part A: Solve for c1 given a_0

# A.1: Solve for X in eqaution (9) 
# A neat way of coding this is to vectorize and then take the sum over a vector
print(para.l) # is a vector
# We now want to divide the vector l by the cumulative product vector of (1+r)
# [(1+r),(1+r)^2,..(1+r)^{T-1},(1+r)^{T}]
collect(1:para.T) # first define a vector of a sequence from 1 to T used for the power
(1+para.r).^collect(1:para.T) # put (1+r) to the power
# We are now ready to evaluate X as:
X =  para.w*sum(para.l./((1+para.r).^collect(1:para.T)))

# Given that we have introduced the concept of evaluating sums using vectorization, 
# we can write a function that solves for c1 given an instance of a^i_0
function c1i(para,a0i)
	# This function solves for c^i_1 as in equation (8)
	# a0i: Savings/Assets at birth
	@unpack T,l,r,w,β,ρ = para
	X =  w*sum(l./((1+r).^collect(1:T)))
	Yi =  a0i
	Z =  sum((β*(1+r)).^((collect(1:T).-1)/ρ)./(1+r).^collect(1:T))
	ci1 = (X+Yi)/Z
	return ci1
end

# save an instance of a^i_0
a0i = 1.0

# let us call the function
c1i(para,a0i)

###########
# Part B: Solve for the consumption path over the life-ages_LifeCycle_Labor

# we start by vectorizing the Long-run Euler equation in (6)
LRE = (para.β*(1+para.r)).^((collect(1:para.T).-1)/para.ρ)



# Now, we can solve for the vector of the consumption path for an instance of a^i_0:
C = c1i(para,a0i).*LRE

# This vectorization is much faster and neater than writing a loop that would use the short-term Euler Equation (5):
Cloop = zeros(para.T) # Initialize the consumption vector
Cloop[1] = c1i(para,a0i)  # In Julia, indexing starts at 1
# Loop through consumption using the Euler equation
for i in 1:para.T-1
    Cloop[i+1] = (para.β * (1 + para.r))^(1 / para.ρ) * Cloop[i]
end
# check that both techniques give the same result: (we check whether the absolute difference is very very small)
is_equal = all(abs.(Cloop - C) .< 1e-12)
println("Both techniques give the same result: ", is_equal)


###########
# Part C: Solve for the savings path over the lifecycle
A = zeros(para.T+1) # Initialize savings vector
A[1] = a0i  # Savings agent i is born with
# Solve the whole savings path using the budget constraint
for t in 1:(para.T)  # Loop from the second period onward
    A[t+1] = para.w * para.l[t] + (1 + para.r) * A[t] - C[t]
end

###########
# Part D: Write a function that solve for the consumption/savings path over the lifecycle
function solveLCM(para,a0i)
	# This function solves for the life-cycle model for instance of assets at birth
	# a0i: Savings/Assets at birth
	@unpack T,l,r,w,β,ρ = para
		# we start by vectorizing the Long-run Euler equation in (6)
	LRE = (β*(1+r)).^((collect(1:T).-1)/ρ)
	# Now, we can solve for the vector of the consumption path for an instance of a^i_0:
	C = c1i(para,a0i).*LRE
	# Solve for the savings path over the lifecycle
	A = zeros(T+1) # Initialize savings vector
	A[1] = a0i  # Savings agent i is born with
	# Solve the whole savings path using the budget constraint
	for t in 1:(T)  # Loop from the second period onward
		A[t+1] = w * l[t] + (1 + r) * A[t] - C[t]
	end
	return C,A
end

# Call the function
C,A = solveLC(para,a0i)

# plot the result
plot(0:para.T,[NaN;C],label="Consumption")
plot!(0:para.T,A,label="Saving")
plot!(xlabel="Age", ylabel="Consumption / Saving")
plot!(legend=:topleft)


###########
# Part D: Write a function that simulated the model
function solveLCM(para,a0i)
	# This function solves for the life-cycle model for instance of assets at birth
	# a0i: Savings/Assets at birth
	@unpack T,l,r,w,β,ρ = para
		# we start by vectorizing the Long-run Euler equation in (6)
	LRE = (β*(1+r)).^((collect(1:T).-1)/ρ)
	# Now, we can solve for the vector of the consumption path for an instance of a^i_0:
	C = c1i(para,a0i).*LRE
	# Solve for the savings path over the lifecycle
	A = zeros(T+1) # Initialize savings vector
	A[1] = a0i  # Savings agent i is born with
	# Solve the whole savings path using the budget constraint
	for t in 1:(T)  # Loop from the second period onward
		A[t+1] = w * l[t] + (1 + r) * A[t] - C[t]
	end
	return C,A
end

# Setting a seed is important when you want to 
# ensure that the results of your random operations 
# (like random number generation) are reproducible.
Random.seed!(1) # set seed

# Call function and plot result
Nsim = 3
Csim,Asim = SimLCM(para,Nsim)

# plot the result
gr();
plot(0:para.T,hcat(fill(NaN, para.Nsim), Csim)',color=[:red :blue :green],label=nothing)
plot!(0:para.T,Asim',color=[:red :blue :green],line=:dash,label=nothing)
plot!(xlabel="Age", ylabel="Consumption / Saving")
plot!(legend=:topleft)

# Add dummy lines to simulate a separate legend for line style
# These dummy plots are invisible but appear in the legend
plot!(0:para.T, fill(NaN, para.T +1), label="Consumption", color=:black, linestyle=:solid)
plot!(0:para.T, fill(NaN, para.T +1), label="Saving", color=:black, linestyle=:dash)

# Add dummy lines to simulate a separate legend for colors (agents)
plot!(0:para.T, fill(NaN, para.T +1), label="Agent 1", color=:blue, linestyle=:solid)
plot!(0:para.T, fill(NaN, para.T +1), label="Agent 2", color=:red, linestyle=:solid)
plot!(0:para.T, fill(NaN, para.T +1), label="Agent 3", color=:green, linestyle=:solid)
plot!(dpi=600, size=(600,400))
savefig("figtabs/Simulation")

####
# Lecture 4: The Transition Path

##########
# Preamble

# start by installing and loading required packages
using Pkg
# List of required packages
packages = ["Plots", "Parameters", "Optim","LinearAlgebra"]
# Loop through and only add packages that are not installed
for pkg in packages
    if !haskey(Pkg.installed(), pkg)
        Pkg.add(pkg)
    end
end

using Plots, Parameters, Optim, LinearAlgebra 

# working directory is the current file
cd(dirname(@__FILE__))

##########
# Implementing the Transition path

# Calibrate the parameter and store them in a mutable construct
@with_kw mutable struct set_para
    # Timing parameters
    T::Int = 20                                        # Maximum age of life
    Tᵣ::Int = 15                                       # Retirement age
	l::Vector{Float16} = vcat(ones(Tᵣ), zeros(T-Tᵣ))     # Exogenous labour supply
	L::Float64 = sum(l)                                  # Aggreage labour supply

    # Preferences
    β::Float64 = 0.96                                  # Patience
    ρ::Float64 = 2.0                                   # Relative Risk Aversion (RRA) / Inverse IES
	
	# Prodcution technology
	α::Float64 = 1/3                                   # Output elasticities of capital
    δ::Float64 = 0.07                                   # yearly depreciation rate
end


# Create an instance of the struct
para = set_para()
# unpack parameters
@unpack T,Tᵣ,l,L,β,ρ,α,δ = para # unpack parameters

# Load functions from lecture 3 that help solve for steady-states
include("Functions_Lecture3.jl")
include("Functions_Lecture4.jl")

# Timing of events
T_shock      = T     # Time of the shock
T_conv       = 100   # Convergence periods (T_conv>T to allow for extra time to convergence after all shocked cohorts die off)
T_perfect    = T     # Perfect foresight periods (People foresee SS prices T periods into the future when making choices)

# The full timeline
TP = T_shock + T_conv + T_perfect

# In period T_shock, the contribution rate is unexpectedly changed from 0 to 0.1
τ0 = 0.0
τ1 = 0.1

###########
# Solve for the pre and post shock steady state

# Discipline capital guess
r_lb = 0.01 - δ  # lower bound interest rate
K_ub = L * ((r_lb + δ) / α)^(1 / (α - 1))  # upper bound K(r_lb)

## calculate pre-MIT-shock K_ss
K_ss0 = optimize(K -> objective_paygo(K,τ0,para), 0.0,K_ub).minimizer

## calculate post-MIT-shock K_ss
K_ss1 = optimize(K -> objective_paygo(K,τ1,para), 0.0,K_ub).minimizer


##########
# Step 1: Guess a transition path of capital

# Interpolating guess for K
Kvec = vcat(fill(K_ss0, T_shock), range(K_ss0, stop=K_ss1, length=T_conv), fill(K_ss1, T_perfect))


##########
# Step 2: Compute associated factor prices and pension benefits
rvec = α .* (Kvec / L) .^ (α - 1) .- δ
wvec = (1 - α) .* (Kvec / L) .^ α
τvec = vcat(fill(τ0, T_shock), fill(τ1, T_conv + T_perfect))
bvec = τvec .* wvec * sum(l) / (T - Tᵣ)

# Preallocating
A = zeros(TP, T)
C = zeros(TP, T)

# Computing steady-state savings and consumption in the old steady state
A_pre, C_pre = savings_function(fill(rvec[1], T), fill(wvec[1], T), fill(bvec[1], T), τ0, para)

# Pre-shock consumption-saving paths (updated later)
for tp in 1:T_shock
    A[tp, :] = A_pre
    C[tp, :] = C_pre
end

# Iterating over guesses of the capital transition path until convergence of the entire path
stepsize = 10000  # Initially a large number that should converge to zero
tol = 1e-6      # Tolerance level for convergence
counter = 0     # Counter for iterations

storeconv = zeros(28,TP)

# Perform updates until the capital transition path converges
while stepsize > tol
    counter += 1  # Increment iteration counter
    ktemp = copy(Kvec)  # Save current guess of the transition path
    storeconv[counter,:] .= ktemp
    ##########
    # Step 3:
    # Adjusting consumption-saving paths for people aged 1 to T at the time of the shock (reacting to changes in prices and policy)
    for age in 1:T
        A[T_shock-age+1, :], C[T_shock-age+1, :] = update_savings(
            age,
            copy(A_pre),
            copy(C_pre),
            rvec[(T_shock-age+1):(T_shock+T-age)],
            wvec[(T_shock-age+1):(T_shock+T-age)],
            bvec[(T_shock-age+1):(T_shock+T-age)],
            τ0,
            para
        )
    end

    ##########
    # Step 4: Loop over periods after the shock
    # Giving birth to new cohorts (born after the shock) and updating capital, factor prices, and benefits
    for tp in (T_shock+1):(T_shock + T_conv)
        ##########
        # Step 4 A:
        # Use inherited savings to produce a new guess for capital, factor prices, and pension benefits in the given period
        Kvec[tp] = sum(diag(reverse(A[(tp-T+1):tp, :],dims=1)))
        rvec[tp] = α * (Kvec[tp] / L)^(α-1) - δ
        wvec[tp] = (1 - α) * (Kvec[tp] / L)^α
        bvec[tp] = τvec[tp] * wvec[tp] * sum(l) / (T - Tᵣ)

        ##########
        # Step 4 B: For cohorts born after the shock, compute new consumption and savings decisions given future prices
        A[tp, :], C[tp, :] = savings_function(
            rvec[tp:(tp+T-1)],
            wvec[tp:(tp+T-1)],
            bvec[tp:(tp+T-1)],
            τ1,
            para
        )
    end

    # Update stepsize to check for convergence
    if counter > 0
        stepsize = maximum(abs.(Kvec - ktemp))
    end
end

println("Number of iterations until convergence: ", counter)

# Plot capital transition path
gr()
plot(1:TP, Kvec, label="Capital Transition Path")
ylims!(0, maximum(Kvec) + 5)
title!("Capital Transition Path")
xlabel!("Periods")
ylabel!("Capital Level")
savefig("figtabs/capital_transition_path.png")



################
# Functions Lecture 4

using Plots, Parameters, Optim

function update_savings(age, A, C, rvec, wvec, bvec, τ, para)
    """
    Update the savings path of agents already born at the time of the shock.
    
    Parameters:
    - age: The age of the agent in the shock period.
    - A: Pre-shock savings path (vector).
    - C: Pre-shock consumption path (vector).
    - rvec: Vector of interest rates (vector).
    - wvec: Vector of wage rates (vector).
    - bvec: Vector of pension benefits (vector).
    - τ: Pension contribution rate (scalar).
    - par: Dictionary containing parameters (T, ρ, β, l).
    
    Returns:
    - Updated savings path (A) and consumption path (C).
    """
    
    # Unpacking parameters
	@unpack T,l,β,ρ = para # unpack parameters

    # Determine savings from the period just before the shock
    if age == 0
        A_init = 0  # New generations do not have initial wealth (sanity check)
    else
        A_init = A[age]  # Generations born before the shock
    end

    # Solve for shock-period consumption (as in )
    numerator = A_init + sum(((1 - τ) .* wvec[age+1:T] .* l[age+1:T] .+ (1 .- l[age+1:T]) .* bvec[age+1:T]) ./ cumprod(1 .+ rvec[age+1:T]))
    denominator = sum((β .^ (0:(T-age-1)) .* cumprod(1 .+ rvec[age+1:T]) ./ (1 .+ rvec[age+1])) .^ (1 / ρ) ./ cumprod(1 .+ rvec[age+1:T]))
    C_age = numerator / denominator

    # Solve for the whole consumption path (as in L2.1)
    C[age+1:T] = C_age .* (β .^ (0:(T-age-1)) .* cumprod(1 .+ rvec[age+1:T]) ./ (1 .+ rvec[age+1])) .^ (1 / ρ)

    # Update savings using the period-by-period budget constraint
    if age == 0
        # Solve for first-period savings given no initial wealth
        A[age+1] = (1 - τ) * wvec[age+1] * l[age+1] + (1 - l[age+1]) * bvec[age+1] - C[age+1]
    end

    # Solve the whole savings path
    for x in age+1:T
        if x > age+1
            A[x] = (1 - τ) * wvec[x] * l[x] + (1 - l[x]) * bvec[x] + (1 + rvec[x]) * A[x-1] - C[x]
        end
    end

    return A, C
end

function savings_function(rvec, wvec, bvec, τ, para)
    """
    Calculate forward-looking problem of agents born after the shock.
    
    Parameters:
    - rvec: Vector of interest rates.
    - wvec: Vector of wage rates.
    - bvec: Vector of pension benefits.
    - τ: Pension contribution rate (scalar).
    - par: Dictionary containing parameters (T, ρ, β, l).
    
    Returns:
    - A: Savings path.
    - C: Consumption path.
    """
    
    # Unpacking parameters
	@unpack T,l,β,ρ = para # unpack parameters


    # Solve for first-period consumption (as in equation )
    c1_numerator = sum(((1 - τ) .* wvec .* l .+ (1 .- l) .* bvec) ./ cumprod(1 .+ rvec))
    c1_denominator = sum(((cumprod(1 .+ rvec) .* β .^ (0:T-1)) ./ (1 .+ rvec[1])) .^ (1 / ρ) ./ cumprod(1 .+ rvec))
    C1 = c1_numerator / c1_denominator

    # Solve for the whole consumption path using the Euler equation (L1.2)
    C = C1 .* (β .^ (0:T-1) .* cumprod(1 .+ rvec) ./ (1 .+ rvec[1])) .^ (1 / ρ)

    # Preallocate storage for savings path
    A = zeros(T)
    
    # Solve for first-period savings given no initial wealth
    A[1] = (1 - τ) * wvec[1] * l[1] - C[1]

    # Solve the whole savings path using the budget constraint
    for t in 2:T
        A[t] = (1 - τ) * wvec[t] * l[t] + (1 - l[t]) * bvec[t] + (1 + rvec[t]) * A[t-1] - C[t]
    end
    
    return A, C
end

################
# Functions Lecture 4

using Plots, Parameters, Optim

function update_savings(age::Int, A::Vector, C::Vector, rvec::Vector, wvec::Vector, bvec::Vector, τ::Float64, para)
    """
    Update the savings path of agents already born at the time of the shock.

    Parameters:
    - age: The age of the agent in the shock period (1-based in Julia).
    - A: Pre-shock savings path (vector).
    - C: Pre-shock consumption path (vector).
    - rvec: Vector of interest rates.
    - wvec: Vector of wage rates.
    - bvec: Vector of pension benefits.
    - τ: Pension contribution rate (scalar).
    - par: Dictionary containing parameters (T, ρ, β, l).

    Returns:
    - Updated savings path (A) and consumption path (C).
    """

    # Unpacking parameters
    @unpack T, l, β, ρ = para  # Unpack parameters

    # Determine savings from the period just before the shock
    if age == 1
        A_init = 0.0  # New generations do not have initial wealth (sanity check)
    else
        A_init = A[age - 1]  # Generations born before the shock
    end

    # Solve for shock-period consumption
    numerator = A_init + sum(((1 - τ) .* wvec[age:T] .* l[age:T] .+ (1 .- l[age:T]) .* bvec[age:T]) ./ cumprod(1 .+ rvec[age:T]))
    denominator = sum((β .^ (0:(T-age)) .* cumprod(1 .+ rvec[age:T]) ./ (1 .+ rvec[age])) .^ (1 / ρ) ./ cumprod(1 .+ rvec[age:T]))
    C_age = numerator / denominator

    # Solve for the whole consumption path
    C[age:T] .= C_age .* (β .^ (0:(T-age)) .* cumprod(1 .+ rvec[age:T]) ./ (1 .+ rvec[age])) .^ (1 / ρ)

    # Update savings using the period-by-period budget
    if age == 1
        # Solve for the first period savings given no initial wealth
        A[age] = (1 - τ) * wvec[age] * l[age] + (1 - l[age]) * bvec[age] - C[age]
    end

    # Solve the whole savings path
    for x in (age+1):T
        A[x] = (1 - τ) * wvec[x] * l[x] + (1 - l[x]) * bvec[x] + (1 + rvec[x]) * A[x - 1] - C[x]
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
    @unpack T, l, β, ρ = para  # Unpack parameters

    # Solve for first-period consumption (as in equation L1.3)
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


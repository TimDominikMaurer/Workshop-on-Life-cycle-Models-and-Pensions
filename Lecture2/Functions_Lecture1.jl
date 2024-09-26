# This file contains the functions from lecture 1
# that solve and simulate the life-cycle model

using Random, Parameters, Distributions

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

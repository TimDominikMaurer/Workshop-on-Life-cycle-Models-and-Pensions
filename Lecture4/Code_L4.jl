####
# Lecture 4: The Transition Path

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
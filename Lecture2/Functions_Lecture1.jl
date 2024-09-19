# This file contains the functions from lecture 1
# that solve and simulate the life-cycle model

using Random,Parameters, Distributions

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

function SimLCM(para,Nsim)
	# This function solves the life-cycle model and simulates Nsim agents/households
	@unpack μ,σ,T = para
	# Simulated initial wealth levels
	a0i_sim = rand(LogNormal(μ, σ), Nsim)
	# solve and simulate for all draws
	Csim = zeros((Nsim,T)) # storage
	Asim = zeros((Nsim,T+1)) # storage
	for i in 1:Nsim
		Csim[i,:],Asim[i,:] = solveLC(para,a0i_sim[i])
	end
	return Csim,Asim
end
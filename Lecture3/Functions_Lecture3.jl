# Functions from lecture 3

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
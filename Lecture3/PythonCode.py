# Import packages
import numpy as np
import scipy.optimize as optimize
import time
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [12, 8]
from IPython import display

# Define a dictionary for model parameters
par = dict() 
# Calibrate parameters
par['T'] = 20 # max age
par['R'] = 15 # retirement age
par['α'] = 1/3 # output elasticities of capital
par['ρ'] = 2 # risk aversion parameter
par['δ'] = 0.07 # yearly depreciation rate
par['β'] = 0.95 # agent's patience parameter
# vector exogenous labor supply given retirement
par['l'] = np.concatenate([np.ones(par['R']),np.zeros([par['T'] - par['R']])]) 
par['L'] = sum(par['l']) # aggregate labor supply

# unpack
T = par['T']
R = par['R']
α = par['α']
ρ = par['ρ']
δ = par['δ']
β = par['β']
l = par['l']
L = par['L']


# STEP 1: Guess a steady-state capital level
K = 35

# STEP 2: Calculate factor prices given the guessed captial level
r = α*(K/L)**(α-1) - δ
w = (1 - α)*(K/L)**α

# print factor prices
print('interest rate:',r) 
print('         wage:',w) 

# STEP 3: Solve for first periond consumption as in equation (L1.4)
C1 = w*sum(l/(1+r)**np.linspace(1,T,T))/sum((β*(1+r))**((np.linspace(1,T,T)-1)/ρ)/(1+r)**np.linspace(1,T,T))
print('C1:',C1) 

# STEP 4: Solve for the consumption path over the life-cycle

# using very similar techniques and the Long-run Euler Equation (L1.2),
# we can solve for the vector of the consumption path:
C = C1*(β*(1+r))**((np.arange(T))/ρ)
print('Long run Euler vector that is multiplied with C1:')
print((β*(1+r))**((np.arange(T))/ρ))
print('Vector of consumption path:')
print(C)

# define storage for consumption path
Cloop = np.zeros(T)
Cloop[0] = C1 # first period consumption is the first index (indexatin in python starts at 0)
for i in range(T-1): # loop consumption forward using the short-term Euler equation in (L1.1)
    Cloop[i+1] = (β*(1+r))**(1/ρ) * Cloop[i]
# check that both techniques give the same result: (we check whether the absolute difference is very very small)
abs(Cloop-C)<10e-12


# STEP 5: Solve for the whole savings path using the budget constraint
A = np.zeros(T) # preallocate storage
A[0] = w*l[0] - C[0] # solve for first period savings given no initial wealth
for t in range(1,T): # solve the whole savings path
    A[t] = w*l[t] + (1 + r)*A[t-1] - C[t]
    
# STEP 6: Compute implied aggregate capital
K_implied = sum(A)
print('Implied level of capital:' ,K_implied)
print('Initial guess:' ,K)


# Solve fore the consumption and savings plan given an interest rate
def solve(K_guess,par): # Solve for consumption, savings and aggregate capital given a guess for r

    # Unpacking Parameters
    T = par['T']
    ρ = par['ρ']
    δ = par['δ']
    β = par['β']
    l = par['l']
    L = par['L']

    # STEP 2: Solve for wage given guessed interest rate
    r = α*(K_guess/L)**(α-1) - δ
    w = (1 - α)*(K_guess/L)**α

    # STEP 3: Solve for first periond consumption as in equation (L1.4)
    C1 = w*sum(l/(1+r)**np.linspace(1,T,T))/sum((β*(1+r))**((np.linspace(1,T,T)-1)/ρ)/(1+r)**np.linspace(1,T,T))
    
    # STEP 4: Solve for the whole consumption path using the Long-Run Euler equation (L1.2)
    C = C1*(β*(1+r))**((np.arange(T))/ρ)

    # STEP 5: Solve for the whole saviongs path using the budget constraint
    A = np.zeros(T) # preallocate storage
    A[0] = w*l[0] - C[0] # solve for first period savings given no initial wealth
    for t in range(T): # solve the whole savings path
        if t>0:
            A[t] = w*l[t] + (1 + r)*A[t-1] - C[t]

    # STEP 6: Compute implied aggregate capital by summing over savings path
    K_implied = sum(A)
     
    return C,A,K_implied


# Define objective function that outputs the distances between K_guess and K_implied (STEP 8)
def objective(K_guess,par): # Solves the SS
   
    C,A,K_implied = solve(K_guess,par) 
    
    # STEP 8: Check distance between K_guess - K_implied (Note we define the loss as the squared difference)
    loss = (K_guess - K_implied)**2  
    return loss



# Define a model-consistent level of the lower bound
r_lb = 0.01 - δ # lower bound r
K_ub = L*((r_lb+δ)/α)**(1/(α-1)) # upper bound K(r_lb)
print('    Upper bound of K:',K_ub)

# STEP 9: Minimize loss

# using an initial guess
Kguess = 100
sol = optimize.minimize(objective,Kguess,args=(par))

# using bounds (constrained optimization)
# sol2 = optimize.minimize_scalar(objective,bounds=[0,K_ub],args=(par),method='bounded')

# Equilibrium capital stock
K_ss = sol.x
print('Steady-state capital:',K_ss[0])


C_ss,A_ss,K_ss = solve(K_ss,par)

# add zero wealth at birth
A_ss = np.concatenate([np.zeros(1),A_ss]) 

plt.plot(np.arange(T+1),A_ss)
plt.plot(np.linspace(1,T,T),C_ss)
plt.title('Consumption and Saving Path')
plt.xlabel("Age")
plt.gca().legend(('Saving','Consumption'))
plt.xticks(np.arange(T+1))
plt.show()


# define the grid of K^i
gridsize  = 100 
ss_jump = 20  # how far away from steady state do we jump

K = np.linspace(K_ss-ss_jump,K_ss+ss_jump,gridsize) # Define grid


# Compute associated K_{+}
K_plus = np.copy(K) # preallocate storage


for i in range(gridsize):
    C_temp,A_temp,K_temp = solve(K[i],par) 
    K_plus[i] = K_temp
    
## plot the maping function
plt.plot(K,K_plus) # mapping function
plt.plot(K,K) # 45 degree line
plt.scatter(K_ss,K_ss,s=200) # Point of steady state
plt.text(K_ss,K_ss-ss_jump/10, '$K_{ss}$', fontsize=16) # label of steady state point
plt.title('Mapping function') 
plt.xlabel('$K^i$')
plt.ylabel('$K^j_{+}$')
plt.gca().legend(('Mapping','45 degree line'))


## illustrate the contraction mapping
K_start = K_ss-ss_jump # starting point / jump off the steady state
niter = 6 # number of iterations

x = [] # storage for x-values of iteration points
y = [] # storage for z-values of iteration points
x.append([K_start]) # x-values of starting point
y.append([K_start]) # y-values of starting point'

K_iter = np.copy(K_start) # initialise starting value

for iteration in range(niter):
    # a. Iterate K forward
    C_iter_plus,A_iter_plus,K_iter_plus = solve(K_iter,par)  # solve for next K
    x.append([K_iter]) # x-values new K
    y.append([K_iter_plus]) # y-values new K
    
    # b. Plot the iteration
    display.display(plt.gcf()) 
    display.clear_output(wait=True) # clear plot after every iteration (so that plot looks interactive)
    
    plt.plot(K,K_plus) # mapping function
    plt.plot(K,K) # 45 degree line
    plt.plot(x,y) # K points of iteration
    plt.scatter(x,np.repeat(y[0],len(x))) # plot convergences along the x-axis 
    plt.scatter(K_ss,K_ss,s=200) # Point of steady state
    plt.text(K_ss,K_ss-ss_jump/10, '$K_{ss}$', fontsize=16) # label of steady state point
    plt.title('Contraction Mapping Plot')
    plt.xlabel('$K^i$')
    plt.ylabel('$K^j_{+}$')
    plt.gca().legend(('Mapping','45 degree line','Iteration'))
    plt.show()
    
    time.sleep(2.5) # low down iterations 
    
    # c. Update K for next iteration
    K_iter = np.copy(K_iter_plus)
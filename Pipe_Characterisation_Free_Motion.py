# =============================================================================
# This code is used to determine the gamma (young modulus) and alpha (E*) parameters
# from the pipe with the free vibration test
# =============================================================================

# Loading of the librairies
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

# Pipe parameters
L = 0.46
u = 0
alpha = 0.02
beta = 0
g = 9.81
dint = 6.35*10**-3
Dext = 15.875*10**-3
I = np.pi*(Dext**4-dint**4)/64
M_silicone = 1.34/1000*100**3
m = M_silicone*np.pi*(Dext**2-dint**2)/4
M = 0

# Free vibration frequency measured experimentally
freq = 0.914777638713204

# Range of gamma parameters on which we compute the eigenvalues
Gamma = np.linspace(0,500,501)

# Number of beam mode shapes used in the Galerkin decomposition
N = 10

# Function computing the value of a polynomial from its coefficients and its order
def f(x,b,order):
    ans = 0
    for i in range(order+1):
        ans += b[i]*x**(order-i)
    return ans


# Equation to solve to obtain the beam eigenfrequencies
def Eq_Cara(Lambda):
    return np.cos(Lambda)*np.cosh(Lambda)+1

# Use of the Newton method to solve the previous equation
LAMBDA = []
for i in range(N):
    LL_Guess = np.pi*(2*i+1)/2 # Initialization of the eigenfrequencies at an approached value
    x0 = LL_Guess + 0.1
    x1 = LL_Guess - 0.1
    while abs(x0-x1)>10**-16: # Loop of the Newton solver with the termination criterion
        xnew = x0 - (x0-x1)*Eq_Cara(x0)/(Eq_Cara(x0)-Eq_Cara(x1)) 
        x1 = x0
        x0 = xnew
    LAMBDA.append(x0)

# Function used to find the first positive value of an array
def first_Positiv(array):
    n = 0
    for i in range(len(array)):
        if array[i] > 0 and n == 0:
            n = i
    return n

# Function used to build the beam mode shapes
def sigma(r):
    return ((np.sinh(LAMBDA[r])-np.sin(LAMBDA[r]))/(np.cosh(LAMBDA[r])+np.cos(LAMBDA[r])))

# Functions used to build the B, C and D matrices
def bsr(s,r):
    if s == r:
        return 2
    else:
        return 4/((LAMBDA[s]/LAMBDA[r])**2+(-1)**(r+s))
    
def csr(s,r):
    if s == r:
        return LAMBDA[r]*sigma(r)*(2-LAMBDA[r]*sigma(r))
    else:
        return 4*(LAMBDA[r]*sigma(r)-LAMBDA[s]*sigma(s))/((-1)**(r+s)-(LAMBDA[s]/LAMBDA[r])**2)
    
def dsr(s,r):
    if s == r:
        return csr(s,r)/2
    else:
        return (4*(LAMBDA[r]*sigma(r)-LAMBDA[s]*sigma(s)+2)*(-1)**(r+s))/(1-(LAMBDA[s]/LAMBDA[r])**4)-((3+(LAMBDA[s]/LAMBDA[r])**4)/(1-((LAMBDA[s]/LAMBDA[r])**4)))*bsr(s,r)                                                                                                                                               

# Maximum assurance criterion function giving the correlation between two vectors
def MAC(X,Y):
    return ((np.dot(X,np.conj(Y)).real)**2+(np.dot(X,np.conj(Y)).imag)**2)/(np.dot(X,np.conj(X))*np.dot(Y,np.conj(Y))).real


# Construction of the B, C and D matrices
B = np.zeros((N,N))
C = np.zeros((N,N))
D = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        B[i,j] = bsr(i,j)  
        C[i,j] = csr(i,j)
        D[i,j] = dsr(i,j)

# Construction of the diagonal matrix
Delta = np.zeros((N,N))
for i in range(N):
    Delta[i,i] = LAMBDA[i]**4
MM = np.eye(N)

# This function returns the eigenvalues and eigenvectors of the time modes with respect to the flowrate u
def result(u, alpha, gamma, beta):
    ff = np.zeros((N,N))
    for i in range(N):
        ff[i,i] = alpha*LAMBDA[i]**4
    C_g = 2*(beta**0.5)*u*B + ff # Construction of the damping matrix
    K = Delta + gamma*B + (u**2-gamma)*C + gamma*D # Construction of the stifness matrix
    F = np.block([[np.zeros((N,N)),MM],[MM,C_g]]) # Reduced order matrix
    E = np.block([[-MM,np.zeros((N,N))],[np.zeros((N,N)),K]]) # Reduced order matrix
    eigenValues, eigenVectors = linalg.eig(-np.dot(np.linalg.inv(F),E))
    return eigenValues, eigenVectors

# Determination of Gamma

# First value of Gamma
gamma = Gamma[0]

# Computation of the eigenvalues and eigenvectors for gamma = 0
eigenValues, eigenVectors = result(u, alpha, gamma, 0)

# Sort the eigenvalues and eigenvectors
arg = np.argsort(np.array([(-1j*eigenValues).real]))[0]
IM_Omega = np.array([(-1j*eigenValues).imag[arg]])
RE_Omega = np.array([(-1j*eigenValues).real[arg]])

# Finds the index of the first mode
first = first_Positiv(RE_Omega[0])

Vectors = np.array([eigenVectors[:,arg]])

# Loop on the values of gamma
for i in range(1,len(Gamma)):
    
    # New value of gamma
    gamma = Gamma[i]   
    
    # Computation of the eigenvalues and eigenvectors for the new gamma value
    eigenValues, eigenVectors = result(u, alpha, gamma, 0)
    diff = np.zeros((2*N,2*N))
    
    # Association of the modes from gamma and gamma+1 with the MAC values
    for l in range(2*N):
        for k in range(2*N):
            diff[l,k] = MAC(Vectors[i-1,:,l],eigenVectors[:,k])
    
    # Sort of the eigenvalues and eigenvectors
    arg_Max = np.argmax(diff,1) 
    IM_Omega = np.append(IM_Omega,[(-1j*eigenValues[arg_Max]).imag],axis=0)
    RE_Omega = np.append(RE_Omega,[(-1j*eigenValues[arg_Max]).real],axis=0)
    Vectors = np.append(Vectors,np.array([eigenVectors[:,arg_Max]]),axis=0)


Y = RE_Omega[:,first] #Re(Omega1)
X1 = np.linspace(0,500,501) #Range of gamma
bd = np.polyfit(X1,Y,20)
Y_reg = f(X1,bd,20) #Construction of Re(Omega1)(gamma)

plt.plot(Gamma[:], Y, label="Results from the Galerkin method",color=(0.3,0.3,0.3))
plt.xlabel("$\gamma$")
plt.ylabel("$Re(\omega_1)(\gamma)$")
plt.show()

# Function to be solved to get the value of gamma from the polynomial fit on Re(Omega1)(gamma)
def to_be_solved_gamma(ga):
    return f(ga,bd,20)**2-4*np.pi**2*freq**2*(ga*L/g)

# Solution of the function and determination of gamma with a Newton solver
Gamma_Guess = 200
x0 = Gamma_Guess + 1 
x1 = Gamma_Guess - 1
while abs(x0-x1)>10**-10:
    xnew = x0 - (x0-x1)*to_be_solved_gamma(x0)/(to_be_solved_gamma(x0)-to_be_solved_gamma(x1))
    x1 = x0
    x0 = xnew
Gamma_sol = x0

plt.plot(Gamma[:],f(Gamma[:],bd,20)**2-4*np.pi**2*freq**2*(Gamma[:]*L/g),color=(0.3,0.3,0.3))
plt.plot([Gamma_sol,Gamma_sol,0],[(f(Gamma[:],bd,20)**2-4*np.pi**2*freq**2*(Gamma[:]*L/g))[-1],0,0],color=(0.01,0.01,0.01),linestyle="dashed")
plt.xlabel(r"$\gamma$")
plt.ylabel(r"$Re(\omega_1)(\gamma)^2-\frac{g\gamma}{L}(2\pi f_1)^2$")
plt.show()

E = m*g*L**3/(Gamma_sol*I) #Value of the young modulus from gamma
gamma = Gamma_sol #Value of gamma
RE = f(gamma,bd,20) #Value of the real part of the first eigenfrequency for the good value of gamma

# Determination of alpha
Alpha = np.linspace(0,0.05,501) #Range of value of alpha
delta = 0.016744044854346572 #Logarithmic amplitude decay measure with the free vibration experience
IM = RE*delta/(2*np.pi) #Imaginary part

# First value of alpha 
alpha = Alpha[0]

# Computation of the eigenvalues and eigenvectors with the first value of alpha
eigenValues, eigenVectors = result(u, alpha, gamma, 0)

# Sort of the modes
arg = np.argsort(np.array([(-1j*eigenValues).real]))[0]
IM_Omega = np.array([(-1j*eigenValues).imag[arg]])
RE_Omega = np.array([(-1j*eigenValues).real[arg]])

# Index of the first mode
first = first_Positiv(RE_Omega[0])
Vectors = np.array([eigenVectors[:,arg]])

# Loop on the range of values of alpha
for i in range(1,len(Alpha)):
    
    # New value of alpha
    alpha = Alpha[i]   
    
    # Computation of the eigenvalues and eigenvectors with the new value of alpha
    eigenValues, eigenVectors = result(u, alpha, gamma, 0)
    
    # Association of the modes from gamma and gamma+1 with the MAC values
    diff = np.zeros((2*N,2*N))
    for l in range(2*N):
        for k in range(2*N):
            diff[l,k] = MAC(Vectors[i-1,:,l],eigenVectors[:,k])    
            
    # Sort of the modes with the MAC values
    arg_Max = np.argmax(diff,1) 
    IM_Omega = np.append(IM_Omega,[(-1j*eigenValues[arg_Max]).imag],axis=0)
    RE_Omega = np.append(RE_Omega,[(-1j*eigenValues[arg_Max]).real],axis=0)
    Vectors = np.append(Vectors,np.array([eigenVectors[:,arg_Max]]),axis=0)
    

Y = IM_Omega[1:,first] #Im(omega1)
X1 = np.linspace(0.0001,0.05,500) #Range of alpha
bd = np.polyfit(X1,Y,20)
Y_reg = f(X1,bd,20) #Construction of Ie(Omega1)(alpha)

plt.plot(Alpha[1:], Y, label="Results from the Galerkin method",color=(0.3,0.3,0.3))
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$Im(\omega_1)(\alpha)$")
plt.show()

# Function to be solved to get the value of gamma from the polynomial fit on Im(Omega1)(alpha)
def to_be_solved_alpha(alpha_test):
    return f(alpha_test,bd,20) - IM

# Determination of the alpha value by solving the previous equation with a newton solver
Alpha_Guess = 0.02
x0 = Alpha_Guess + 0.01
x1 = Alpha_Guess - 0.01
while abs(x0-x1)>10**-16:
    xnew = x0 - (x0-x1)*to_be_solved_alpha(x0)/(to_be_solved_alpha(x0)-to_be_solved_alpha(x1))
    x1 = x0
    x0 = xnew
Alpha_sol = x0

plt.plot(Alpha[1:],f(Alpha[1:],bd,20) - IM*np.ones(Alpha[1:].shape[0]),color=(0.3,0.3,0.3))
plt.plot([Alpha_sol,Alpha_sol,0],[(f(Alpha[1:],bd,20)-IM)[0],0,0],color=(0.01,0.01,0.01),linestyle="dashed")
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$Im(\omega_1)(\alpha)-\frac{Re(\omega_1)\delta_1}{2\pi}$")
plt.show()

# Value of alpha
alpha = Alpha_sol
# Ucr_exp = 4*9.8/(60*1000*np.pi*dint**2)

# Value of the damping parameter
Estar = L**2*alpha*(E*m/I)**0.5
M = 996*np.pi*dint**2/4

# Non dimensional parameters with water
alpha_eau = (I/(E*(M+m)))**0.5*Estar/L**2
beta_eau = M/(M+m)
gamma_eau = (M+m)*L**3*g/(E*I)

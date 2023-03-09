# =============================================================================
# This code is used to generate data on the deleflection of the pipe after a 
# perturbation, before the critical speed and with the linear model
# =============================================================================

# Loading of the librairies
import numpy as np
import matplotlib.pyplot as plt

# Non dimensional parameters of the pipe
d = 6.35*10**-3
D = 15.875*10**-3
M_silicone = 1.34/1000*100**3
m = M_silicone*np.pi*(D**2-d**2)/4
M = 996*np.pi*d**2/4
Estar = 5333.901583749264
E = 225171.9440703812
I = np.pi*(D**4-d**4)/64
L = 46*10**-2
g = 9.81
alpha = (I/(E*(M+m)))**0.5*Estar/L**2
beta = M/(M+m)
gamma = (M+m)*L**3*g/(E*I)
tmax = 1

# Number of beam mode shapes used in the Galerkin decomposition
N = 10

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

# Construction of the B, C and D matrices
B = np.zeros((N,N))
C = np.zeros((N,N))
D = np.zeros((N,N))
M = np.eye(N)
for i in range(N):
    for j in range(N):
        B[i,j] = bsr(i,j)  
        C[i,j] = csr(i,j)
        D[i,j] = dsr(i,j)

# Construction of the diagonal matrices
Delta = np.zeros((N,N))
FF = np.zeros((N,N))
for i in range(N):
    Delta[i,i] = LAMBDA[i]**4
    FF[i,i] = alpha*LAMBDA[i]**4

# This function returns the eigenvalues and eigenvectors of the time modes with respect to the flowrate u
def result(u,beta,gamma):
    S = 2*beta**0.5*u*B # Construction of the damping matrix
    K = Delta + gamma*B + (u**2-gamma)*C + gamma*D # Construction of the stifness matrix
    F = np.block([[np.zeros((N,N)),M],[M,S]]) # Reduced order matrix
    E = np.block([[-M,np.zeros((N,N))],[np.zeros((N,N)),K]]) # Reduced order matrix
    eigenValues, eigenVectors = np.linalg.eig(-np.dot(np.linalg.inv(F),E)) # Solving of the eigenvalue problem
    return eigenValues, eigenVectors

# This function returns the deflection of the pipe for the (x,t) coordinates and for a given flowrate
def eta(x,t,u):
    # Compute the eigenvalues of the times functions
    eigenValues, eigenVectors = result(u,beta,gamma)
    # Use of the initial conditions to compute the constant parameters of the solution
    # At t=0 the deflection of the pipe is supposed to be the 0.3x the first beam mode shape
    CL = np.zeros(2*N)
    CL[0] = 0.3
    Constant = np.dot(np.linalg.inv(eigenVectors),CL)
    Sol = 0
    for r in range(N):
        qr = 0
        for i in range(2*N):
            qr += Constant[i]*np.exp(eigenValues[i]*t)*eigenVectors[r,i]
        phir = np.cosh(LAMBDA[r]*x)-np.cos(LAMBDA[r]*x)-sigma(r)*(np.sinh(LAMBDA[r]*x)-np.sin(LAMBDA[r]*x))
        Sol += qr*phir
    return Sol

u = np.array([3,6,9,12])

# Plotting of the results
t = np.linspace(0,1,1001)
plt.plot(t,eta(L,t,u[0]),color=(0.3,0.3,0.3),linestyle="solid",label="u="+str(u[0]))
plt.plot(t,eta(L,t,u[1]),color=(0.3,0.3,0.3),linestyle="dashed",label="u="+str(u[1]))
plt.plot(t,eta(L,t,u[2]),color=(0.3,0.3,0.3),linestyle="dashdot",label="u="+str(u[2]))
plt.plot(t,eta(L,t,u[3]),color=(0.3,0.3,0.3),linestyle="dotted",label="u="+str(u[3]))
plt.xlabel("Time tau")
plt.ylabel("Deflection eta")
plt.title("Deflection of the tip along time")
plt.legend()
plt.show()
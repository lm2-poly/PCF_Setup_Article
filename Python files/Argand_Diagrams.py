# =============================================================================
# This code is used to plot the evolution of the modes frequency and damping 
# with the evolution of the flowrate
# =============================================================================

# Loading of the librairies
import numpy as np
import matplotlib.pyplot as plt

# Size of the ploting window
xmin = 0
xmax = 70
ymin = -10
ymax = 40

# Non dimensional parameters of the pipe
alpha = 0.00
gamma = 10
beta = 0.65
umax = 20

# Step of evolution of the flowrate
du = 0.02

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

# Function computing the maximum assurance criterions between 2 vectors to acess their degree of similarity
def MAC(X,Y):
    return ((np.dot(X,np.conj(Y)).real)**2+(np.dot(X,np.conj(Y)).imag)**2)/(np.dot(X,np.conj(X))*np.dot(Y,np.conj(Y))).real

# Initialization of the modes for u = 0
u_array = np.array([0])
u = 0
eigenValues, eigenVectors = result(u,beta,gamma)

# Sorting of the modes by acending order of the real part of the eigenvalues
arg = np.argsort(np.array([(-1j*eigenValues).real]))[0]
IM_Omega = np.array([(-1j*eigenValues).imag[arg]])
RE_Omega = np.array([(-1j*eigenValues).real[arg]])
Vectors = np.array([eigenVectors[:,arg]])

# Start of the loop on the flowrate
i = 0 
while u < umax:
    i += 1 
    
    # New flowrate
    u_tempo = u + du  
    du_tempo = du  
    
    # Eigenvalues corresponding to this flowrate
    eigenValues, eigenVectors = result(u_tempo,beta,gamma)
    
    # Construction of the matrix contraining the MAC between each vectors from iteration i and i+1
    diff = np.zeros((2*N,2*N))
    for l in range(2*N):
        for k in range(2*N):
            diff[l,k] = MAC(Vectors[i-1,:,l],eigenVectors[:,k])
        
    # If the MAC is smaller than 0.99 we reduce the flowrate step by half
    while min(np.max(diff,1)) < 0.99 and du_tempo > 10**-10:
        u_tempo = u+du_tempo/2
        du_tempo = du_tempo/2 
        eigenValues, eigenVectors = result(u,beta,gamma)  
        for l in range(2*N):
            for k in range(2*N):
                diff[l,k] = MAC(Vectors[i-1,:,l],eigenVectors[:,k])  

    # Filling of the matrices containing the values of u, Omega...
    u = u_tempo
    u_array = np.append(u_array,u)
    arg_Max = np.argmax(diff,1) 
    IM_Omega = np.append(IM_Omega,[(-1j*eigenValues[arg_Max]).imag],axis=0)
    RE_Omega = np.append(RE_Omega,[(-1j*eigenValues[arg_Max]).real],axis=0)
    Vectors = np.append(Vectors,np.array([eigenVectors[:,arg_Max]]),axis=0)

# Index of the 4 first modes
mode1 = N
mode2 = N+1
mode3 = N+2
mode4 = N+3

# Find the indexes of the flowrate values close to the integer for plotting
index = []
for i in range(0,umax+1):
    index.append(np.argmin(np.abs(u_array-i)))
    
# Selection of the first 4 modes for ploting
x_mode_1 = RE_Omega[index,mode1]
y_mode_1 = IM_Omega[index,mode1]
x_mode_2 = RE_Omega[index,mode2]
y_mode_2 = IM_Omega[index,mode2]
x_mode_3 = RE_Omega[index,mode3]
y_mode_3 = IM_Omega[index,mode3]
x_mode_4 = RE_Omega[index,mode4]
y_mode_4 = IM_Omega[index,mode4]

# Plot of the results
plt.plot(RE_Omega[:,mode1],IM_Omega[:,mode1],label="Mode 1",linestyle="solid",color=(0.3,0.3,0.3))
plt.plot(RE_Omega[:,mode2],IM_Omega[:,mode2],label="Mode 2",linestyle="dashdot",color=(0.3,0.3,0.3))
plt.plot(RE_Omega[:,mode3],IM_Omega[:,mode3],label="Mode 3",linestyle="dashed",color=(0.3,0.3,0.3))
plt.plot(RE_Omega[:,mode4],IM_Omega[:,mode4],label="Mode 4",linestyle="dotted",color=(0.3,0.3,0.3))
plt.plot(x_mode_1,y_mode_1,".",color=(0.3,0.3,0.3),markersize=8)
plt.plot(x_mode_2,y_mode_2,".",color=(0.3,0.3,0.3),markersize=8)
plt.plot(x_mode_3,y_mode_3,".",color=(0.3,0.3,0.3),markersize=8)
plt.plot(x_mode_4,y_mode_4,".",color=(0.3,0.3,0.3),markersize=8)
plt.xlim((xmin,xmax))
plt.ylim((ymin,ymax))
plt.xlabel("Re(Omega)")
plt.ylabel("Im(Omega)")
plt.title("Evolution de Omega en faisant varier la vitesse u pour beta = " + str(beta) + " et gamma = " + str(gamma))
plt.legend()
plt.show()
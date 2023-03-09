# =============================================================================
# This code is used to extract data (frequency and logarithmic amplitude decay)
# from the free vibration videos 
# =============================================================================

# Loading of the needed librairies
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

# Function to compute the value of a polynomial function given its order and coefficients
def f(x,b,order):
    ans = 0
    for i in range(order+1):
        ans += b[i]*x**(order-i)
    return ans

# Loading of the position of the tip of the pipe through time from the free vibration videos
z,x = np.loadtxt("C:\\Users\Morgan\Videos\Videos_Pipe_0.25in_46cm\XYZT_tip_0Hz1.txt")

# Use of the 2000 first frames
x1 = x[2000:]

# Construction of an array containing the timestamps of each frames
t = np.linspace(0,x1.shape[0]/150,x1.shape[0])

# Extraction of the peaks from the x deflection
peaks1 = scipy.signal.find_peaks(x1,height=[0,0.5],distance=10)[0][:-1]
peaks1_n = scipy.signal.find_peaks(-x1,height=[0,0.5],distance=10)[0][:-1]

# Fitting of the polynomial function of order one on the log of the peaks to obtain the logarithmic decay
b1 = np.polyfit(peaks1/150,np.log(x1[peaks1]),1)

# Fast fourier transform to obtain the free vibration frequency
fft1 = np.fft.rfft(x1)

# Plotings
fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot()
ax.plot(t[::10],x1[::10],label = "Position of the tip of the pipe",color=(0.6,0.6,0.6),linewidth = 1)
ax.plot(peaks1/150,x1[peaks1], label = "Enveloppe of the amplitude",color=(0.1,0.1,0.1),linestyle="dashed",linewidth = 3)
ax.plot(peaks1_n/150,x1[peaks1_n],color=(0.1,0.1,0.1),linestyle="dashed",linewidth = 3)
ax.set_xlabel("Time in seconds")
ax.set_ylabel("Deflection in meters")
ax.set_ylim((-0.2, 0.2))
ax.legend()
plt.show()

plt.plot(np.linspace(150/16000,150*200/16000,199),(np.real(fft1[1:200])**2+np.imag(fft1[1:200])**2)**0.5,color=(0.3,0.3,0.3))
plt.xlabel("Frequency in Hz")
plt.ylabel("Amplitude of the FFT")
plt.show()
f1 = 150*(peaks1.shape[0]-1)/(peaks1[-1]-peaks1[0])



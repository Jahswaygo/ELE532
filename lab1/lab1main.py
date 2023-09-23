#Lab1 ELE532
#Jahmil Ally (501045419)

#Imports
import time
import numpy as np
import scipy.io as sci
import matplotlib.pyplot as plt
import sounddevice as sd

#Defining a generic function for plotting
def plot(f_t, t, newGraph=True, figsize=(12.0, 6.0), title='', plotLabel='', xLabel='t', yLabel='f(t)', color='g'):
    if newGraph:
        plt.figure(figsize=figsize)
    
    #Configurable titles/ Labels
    plt.plot(t, f_t,color= color, label=plotLabel)
    if title !='':
        plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    
    #Visuals
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

# A. Anonymous functions and plotting continuous functions

# Problem A.1: 
# Figure 1.46: Plotting f(t) = e^(-t) * cos(2πt)
t = np.linspace(-2, 2, 8)
f_t = np.exp(-t) * np.cos(2 * np.pi * t)
plot(f_t, t, title='A.1 Part A Generated Plot', plotLabel='e^(-t) * cos(2πt)')

# Figure 1.47: Additional Points
t = np.linspace(-2, 2, 100)
f_t = np.exp(-t) * np.cos(2 * np.pi * t)
plot(f_t, t, title='A.1 Part B Generated Plot', plotLabel='e^(-t) * cos(2πt)')

# Problem A.2:
t = np.linspace(-2, 2, 5)
f_t = np.exp(t)
plot(f_t, t, title='A.2 Generated Plot', plotLabel='e^(-t)')
plt.xticks(np.arange(-2, 2.01, 1))

# Problem A.3:
# A.2 plot
t = np.linspace(-2, 2, 5)
f_t = np.exp(t)
plot(f_t, t, title='A.2 Generated Plot Vs Figure 1.46', plotLabel='e^(-t)')
plt.xticks(np.arange(-2, 2.01, 1))
# SuperImpose Figure 1.46 from A,1
t = np.linspace(-2, 2, 8)
f_t = np.exp(-t) * np.cos(2 * np.pi * t)
plot(f_t, t,newGraph=False, color='r' , plotLabel='e^(-t) * cos(2πt)')

# Display all plots
plt.show()
# End of code
#Lab1 ELE532
#Jahmil Ally (501045419)

#Imports
import time
import numpy as np
import scipy.io as sci
import matplotlib.pyplot as plt
import sounddevice as sd

#Defining a generic function for plotting
def plot(f_t, t, newGraph=True, figsize=(12.0, 6.0), title='', functionLabel='', xLabel='t', yLabel='f(t)', color='g'):
    if newGraph:
        plt.figure(figsize=figsize)
    
    #Configurable titles/ Labels
    plt.plot(t, f_t, color=color, label=functionLabel)
    if title !='':
        plt.title(title)
    if xLabel !='':
        plt.xlabel(xLabel)
    if yLabel !='':
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
plot(f_t, t, title='A.1: Part A Generated Plot', functionLabel='e^(-t) * cos(2πt)')

# Figure 1.47: Additional Points
t = np.linspace(-2, 2, 100)
f_t = np.exp(-t) * np.cos(2 * np.pi * t)
plot(f_t, t, title='A.1: Part B Generated Plot', functionLabel='e^(-t) * cos(2πt)')

# Problem A.2:
t = np.linspace(-2, 2, 5)
f_t = np.exp(t)
plot(f_t, t, title='A.2: Generated Plot', functionLabel='e^(-t)')
plt.xticks(np.arange(-2, 2.01, 1))

# Problem A.3:
# A.2 plot
t = np.linspace(-2, 2, 5)
f_t = np.exp(t)
plot(f_t, t, title='A.2: Generated Plot Vs Figure 1.46', functionLabel='e^(-t)')
plt.xticks(np.arange(-2, 2.01, 1))
# SuperImpose Figure 1.46 from A,1
t = np.linspace(-2, 2, 8)
f_t = np.exp(-t) * np.cos(2 * np.pi * t)
plot(f_t, t,newGraph=False, color='r' , functionLabel='e^(-t) * cos(2πt)')


#B. Time shifting and time scaling

#Problem B.1:
t = np.linspace(-1, 2, 1000)
p_t = np.heaviside(t, 1) - np.heaviside(t - 1, 1)
plot(p_t, t, title='B.1: Unit Step Function Plotted', functionLabel='u(t) - u(t-1)', yLabel='p(t)')

#Problem B.2:
def r(t):
    return t * p_t

def n(t):
    return r(t) + r(-t + 2)

r_t = r(t)
n_t = n(t)

plot(r_t, t, title='B.2: r(t) VS n(t) Plotted', functionLabel='r(t) = tp(t)', yLabel='r(t)/n(t)')
plot(n_t, t,newGraph=False,color='r' , functionLabel='n(t) = r(t) + r(-t + 2).')

#Problem B.3:
#n1(t)
t = np.linspace(-1, 1, 1000)  # Reduce time values for n1 and n2
t = 0.5*t
n1_t = n(t)
plot(n1_t, t, title='B.3: n1(t) Vs n2(t) Plotted', functionLabel='n1(t) = n(1/2 t)', yLabel='n1(t)/n2(t)')
#n2(t)
t = np.linspace(-1, 1, 1000) 
t = 0.5*(t + (1/2))
n2_t = n(t)  # Adjust the time values for n2
plot(n2_t, t, newGraph=False, color='r', functionLabel='n2(t) = n1(t + 1/2)')

#Problem B.4:
#n3(t)
t = np.linspace(-1, 1, 1000)
t = t+(1/4)
n3_t = n(t)
plot(n3_t, t, title='B.4: n3(t) Vs n4(t) Plotted', functionLabel='n3(t) = n(t + 1/4)', yLabel='n3(t)/n4(t)')
#n4(t)
t = np.linspace(-1, 1, 1000)  
t = 1/2*(t + 1/4)
n4_t = n(t)
plot(n4_t, t, newGraph=False, color='r', functionLabel='n4(t) = n3(1/2 t)')

#Problem B.5:
#n2(t)
t = np.linspace(-1, 1, 1000) 
t = 0.5*(t + (1/2))
n2_t = n(t)  # Adjust the time values for n2
plot(n2_t, t, title='B.5: n2(t) Vs n4(t) Plotted', functionLabel='n2(t) = n1(t + 1/2)', yLabel='n2(t)/n4(t)')
#n4(t)
t = np.linspace(-1, 1, 1000)  
t = 1/2*(t + 1/4)
n4_t = n(t)
plot(n4_t, t, newGraph=False, color='r', functionLabel='n4(t) = n3(1/2 t)')

# Display all plots
plt.show()
# End of code
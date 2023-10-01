#Lab1 ELE532
#Jahmil Ally (501045419)

#Imports
import time
import numpy as np
import scipy.io as sci
import matplotlib.pyplot as plt
import sounddevice as sd

#Variable Declaration
color='g'

#Defining a generic function for plotting
def plot(f_t, t, newGraph=True, figsize=(12.0, 6.0), title='', functionLabel='', xLabel='t', yLabel='f(t)'):
    global color
    if newGraph:
        plt.figure(figsize=figsize)
        color='g'
    
    #Configure Colour
    if color == 'g' and  newGraph==False:
        color='r'
    elif color == 'r' and  newGraph==False:
        color='y'
    elif color == 'y' and  newGraph==False:
        color='b'
    elif color == 'b' and  newGraph==False:
        color='o'
    elif color == 'o' and  newGraph==False:
        color='p'
    
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
t = np.arange(-2, 2, 0.01)
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
plot(f_t, t,newGraph=False, functionLabel='e^(-t) * cos(2πt)')


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
plot(n_t, t, newGraph=False, functionLabel= 'n(t) = r(t) + r(-t + 2).')

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
plot(n2_t, t, newGraph=False, functionLabel='n2(t) = n1(t + 1/2)')

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
plot(n4_t, t, newGraph=False, functionLabel='n4(t) = n3(1/2 t)')

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
plot(n4_t, t, newGraph=False, functionLabel='n4(t) = n3(1/2 t)')


#C. Visualizing operations on the independent variable and algorithm vectorization

#Problem C.1 : g(t) = u(t) * f(t)
t = np.linspace(-2, 2, 1000)
f_t = np.exp(-2 * t) * np.cos(4 * np.pi * t)
u_t = np.heaviside(t, 1)
g_t = u_t * f_t
plot(g_t, t, title='C.1: g(t) Plotted', functionLabel= "g(t) = f(t) * u(t)", xLabel='t', yLabel='g(t)')

#Problem C.2: Plotting s(t)
t = np.arange(-2, 4, 0.01)
t = t + 1
f_t = np.exp(-2 * t) * np.cos(4 * np.pi * t)
u_t = np.heaviside(t, 1)
g_t = u_t * f_t
plot(g_t,t, title='C.2: s(t) Plotted', functionLabel= "s(t) = g(t+1)", xLabel='t', yLabel='s(t)')

#Problem C.3: g(t) for different alpha values
t = np.arange(0, 4, 0.01)
u_t = np.heaviside(t, 1)
alpha = [1, 3, 5, 7]

newGraph = True
for i in range(0, 4):
    if(i > 0):
        newGraph = False
    f_t = np.exp(-2) * np.exp(-1 * alpha[i] * t) * np.cos(4 * np.pi * t)
    s_t = f_t * u_t
    functionLabel = "alpha: " + str(alpha[i])
    plot(s_t, t, newGraph=newGraph, title='C.3: sα(t) Plotted', functionLabel= functionLabel , xLabel='t', yLabel='s(t)')

#Problem C.4: Array Length of Alpha 
length=len(alpha)
print("\nThe Length of the Matrix is " + str(length) + " Cells\n")


#D. Array indexing
#Read MATLAB data file
data = sci.loadmat('lab1\ELE532_Lab1_Data.mat')

#Load data
A = data['A']
B = data['B']
array_x_audio = data['x_audio']

#Problem D.1 : Modifications to Array A
#(a) A(:)
a = A.flatten()
print("(a) Flattened A to 1D Array:")
print(a)

#(b) A([ 2 4 7 ])
b_indexes = np.array([1, 3, 6])
b = A.flatten()[b_indexes]
print("\n(b) Extracted elements based on index:")
print(b)

#(c)
c_mask = A >= 0.2
print("\n(c) Boolean mask for all elements >= 0.2:")
print(c_mask)

#(d)
d = A[c_mask]
print("\n(d) Extracted elements where A >= 0.2:")
print(d)

#(e)
A[c_mask] = 0
print("\n(e) Setting elements >= 0.2 to 0:")
print(A)

#Problem D.2
rows, cols = B.shape
#(a) Set values below 0.01 to zero
#(c) Check execution time
start = time.time()
for i in range(rows):
    for j in range(cols):
        if abs(B[i, j]) < 0.01:
            B[i, j] = 0
end = time.time()
nestedTime = end - start

#(b) Indexing approach
start = time.time()
B[B < 0.01] = 0
end = time.time()
IndexTime = end - start

print("\nNested Time: " + str(nestedTime) + "\nIndex time:" + str(IndexTime) + "\n")

#Problem D.3
#Create a Copy of the data
audio_X = np.copy(array_x_audio)

#Threshold for compression
threshold = 0.1

#Initialize counter for zero-valued samples
zero_samples = 0

#Nested Loop to iterate through the array and apply the compression
for i in range(len(audio_X)):
    if abs(audio_X[i]) < threshold:
        audio_X[i] = 0
        zero_samples += 1

#Print the number of zero-valued samples
print(f"\nNumber of zero-valued samples: {zero_samples}")

#Play the original audio
sd.play(array_x_audio, 8000)
sd.wait()

#Play the compressed audio
sd.play(audio_X, 8000)
sd.wait()


#EOF CONTENT 

#Display all plots
plt.show()

#End of code

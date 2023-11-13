#Lab2 ELE532
#Jahmil Ally (501045419)

#Imports
import os
import time
import numpy as np
import sympy as sp
import scipy.io as sci
import matplotlib.pyplot as plt
import matlab.engine

#Variable Declaration
color="g"
eng = matlab.engine.start_matlab()

#Defining a generic function for plotting
def plot(f_t, t, newGraph=True, figsize=(12.0, 6.0), title="", functionLabel="", xLabel="t", yLabel="f(t)"):
    global color
    if newGraph:
        plt.figure(figsize=figsize)
        color="g"
    
    #Configure Colour
    if  color == "g" and  newGraph==False:
        color="r"
    elif color == "r" and  newGraph==False:
        color="y"
    elif color == "y" and  newGraph==False:
        color="b"
    elif color == "b" and  newGraph==False:
        color="o"
    elif color == "o" and  newGraph==False:
        color="p"
    
    #Configurable titles/ Labels
    plt.plot(t, f_t, color=color, label=functionLabel)
    if title !="":
        plt.title(title)
    if xLabel !="":
        plt.xlabel(xLabel)
    if yLabel !="":
        plt.ylabel(yLabel)
    
    #Visuals
    plt.grid(True)
    plt.legend()
    plt.tight_layout()



#Part A:
#Problem A.1
#Define values
R = [1e4, 1e4, 1e4]
C = [1e-6, 1e-6]

#Coefficients for the characteristic equation
A1 = [1, (1/R[0] + 1/R[1] + 1/R[2]) / C[1], 1 / (R[0] * R[1] * C[0] * C[1])]

#Characteristic roots
lambda_values = np.roots(A1)

print("\nA.1) Roots:" + str(lambda_values[0]) + "," + str(lambda_values[1]) + "\n")

#Problem A.2
#Time vector
t = np.arange(0, 0.1, 0.0005)

#Unit step function: u(t)
u = lambda t: 1.0 * (t >= 0)

#h(t) using the characteristic roots
h = lambda t: (C[0] * np.exp(lambda_values[0] * t) + C[1] * np.exp(lambda_values[1] * t)) * (u(t))

#h(t) Plotted
plot(h(t), t, title="Problem A2: Impulse Response", functionLabel="h(t)", xLabel="Time [s]", yLabel="Amplitude")


#Problem A.3   
def CH2MP2(R, C):
    #Coefficients for the characteristic equation
    A = [1, (1/R[0] + 1/R[1] + 1/R[2]) / C[1], 1 / (R[0] * R[1] * C[0] * C[1])]
    
    #Characteristic roots
    roots = np.roots(A)
    
    return roots

lambda_ = CH2MP2([1e4, 1e4, 1e4],[1e-9, 1e-6])

print("\nA.3) Roots:" + str(lambda_[0]) + "," + str(lambda_[1]) + "\n")

plt.show()



#Part B:
#Problem B.1
#Specify the primary and alternate file paths
primary_script_path = "/Users/jah/Documents/GitHub/ELE532/lab2/CH2MP4.m"
alternate_script_path = "C:\\Users\\Jahmil\\Desktop\\Coding_Projects\\ELE532\\lab2\\CH2MP4.m"

#Check if the file exists in the primary location
if os.path.exists(primary_script_path):
    script_path = primary_script_path
else:
    #If not, use the alternate location
    script_path = alternate_script_path

eng = matlab.engine.start_matlab()
eng.eval(f"run('{script_path}')", nargout=0)


#Problem B.2
#Defining functions
x = lambda t: np.heaviside(t , 1) - np.heaviside(t - 2, 1)
h = lambda t: (t+1) * (np.heaviside(t + 1, 1) - np.heaviside(t, 1))

#Defining time vector
t = np.arange(-2, 5, 0.01)

x_t = x(t)
h_t = h(t)

y_t = np.convolve(x_t, h_t, "same") * 0.01

plt.figure(figsize=(6, 14))

#Subplot for x(t)
plt.subplot(3, 1, 1)
plt.plot(t, x_t, label="x(t)")
plt.title("x(t)")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)

#Subplot for h(t)
plt.subplot(3, 1, 2)
plt.plot(t, h_t, label="h(t)")
plt.title("h(t)")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)

#Subplot for y(t)
plt.subplot(3, 1, 3)
plt.plot(t, y_t)
plt.title("y(t)")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplots_adjust(hspace=0.5)

plt.show()


#Problem B.3    
#Part a)
t = np.linspace(-10, 10, 1000)
x1 = lambda t: (np.heaviside(t - 4, 0.5) - np.heaviside(t - 6, 0.5))
x2 = lambda t: (np.heaviside(t + 5, 0.5) - np.heaviside(t + 4, 0.5))

x1_t = x1(t)
x2_t = x2(t)

convolution = np.convolve(x1_t, x2_t, "same") * (t[1]-t[0])  # Multiply by dt for integration

plt.figure(figsize=(6, 14))

#Subplot for x1(t)
plt.subplot(3, 1, 1)
plt.plot(t, x1_t)
plt.title("x1(t)")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)

#Subplot for x2(t)
plt.subplot(3, 1, 2)
plt.plot(t, x2_t)
plt.title("x2(t)")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)

#Subplot for convolution
plt.subplot(3, 1, 3)
plt.plot(t, convolution)
plt.title("Convolution of x1(t) and x2(t)")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplots_adjust(hspace=0.5)

#Part b)
t = np.linspace(-10, 10, 1000)
x1 = lambda t: (np.heaviside(t - 3, 0.5) - np.heaviside(t - 5, 0.5))
x2 = lambda t: (np.heaviside(t + 5, 0.5) - np.heaviside(t + 3, 0.5))

x1_t = x1(t)
x2_t = x2(t)

convolution = np.convolve(x1_t, x2_t, "same") * (t[1]-t[0])  # Multiply by dt for integration

plt.figure(figsize=(6, 14))

#Subplot for x1(t)
plt.subplot(3, 1, 1)
plt.plot(t, x1_t)
plt.title("x1(t)")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)

#Subplot for x2(t)
plt.subplot(3, 1, 2)
plt.plot(t, x2_t)
plt.title("x2(t)")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)

#Subplot for convolution
plt.subplot(3, 1, 3)
plt.plot(t, convolution)
plt.title("Convolution of x1(t) and x2(t)")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)
plt.subplots_adjust(hspace=0.5)

#Part h)
t = np.linspace(-4, 3, 1000)  #Extended to capture both functions
x1 = lambda t: np.exp(t) * (np.heaviside(t + 2, 0.5) - np.heaviside(t, 0.5))
x2 = lambda t: np.exp(-2 * t) * (np.heaviside(t, 0.5) - np.heaviside(t - 1, 0.5))

x1_t = x1(t)
x2_t = x2(t)

convolution = np.convolve(x1_t, x2_t, "same") * (t[1]-t[0])  # Multiply by dt for integration

plt.figure(figsize=(6, 14))

# Subplot for x1(t)
plt.subplot(3, 1, 1)
plt.plot(t, x1_t)
plt.title("x1(t)")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)

# Subplot for x2(t)
plt.subplot(3, 1, 2)
plt.plot(t, x2_t)
plt.title("x2(t)")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)

# Subplot for convolution
plt.subplot(3, 1, 3)
plt.plot(t, convolution)
plt.title("Convolution of x1(t) and x2(t)")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)
plt.subplots_adjust(hspace=0.5)

plt.show()


#Part C:
#Problem C.1
#Defining Functions 
t = np.arange(-1, 5, 0.001)
h1 = lambda t: np.exp(t/5) * np.heaviside(t, 1)
h2 = lambda t: 4*np.exp(-t/5) * np.heaviside(t, 1)
h3 = lambda t: 4*np.exp(-t) * np.heaviside(t, 1)
h4 = lambda t: 4*(np.exp(-t/5) - np.exp(-t)) * np.heaviside(t, 1)


h1_t= h1(t)
h2_t= h1(t)
h3_t= h1(t)
h4_t= h1(t)


plt.figure(figsize=(8, 16))

# Subplot for h1(t)
plt.subplot(4, 1, 1)
plt.plot(t, h1_t, label="e^(t/5) * u(t)")
plt.title("S1: h(t)")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)

# Subplot for h2(t)
plt.subplot(4, 1, 2)
plt.plot(t, h2_t, label="4e^(-t/5) * u(t)")
plt.title("S2: h2(t)")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)

# Subplot for h3(t)
plt.subplot(4, 1, 3)
plt.plot(t, h3_t, label="4e^-t * u(t)")
plt.title("S3: h3(t)")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)

# Subplot for h4(t)
plt.subplot(4, 1, 4)
plt.plot(t, h4_t, label="4(e^(-t/5) - e^(-t)) * u(t)")
plt.title("S4: h4(t)")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplots_adjust(hspace=0.5)

plt.show()


#Problem C.2
#S1:
eigenvalue_1 = 1/5

#S2:
eigenvalue_2 = -(1/5)

#S3:
eigenvalue_3 = -1

#S4:
eigenvalue_4a = -(1/5)
eigenvalue_4b = -1


#Problem C.3
#Specify the primary and alternate file paths
primary_script_path = "/Users/jah/Documents/GitHub/ELE532/lab2/C3.m"
alternate_script_path = "C:\\Users\\Jahmil\\Desktop\\Coding_Projects\\ELE532\\lab2\\C3.m"

#Check if the file exists in the primary location
if os.path.exists(primary_script_path):
    script_path = primary_script_path
else:
    #If not, use the alternate location
    script_path = alternate_script_path

eng = matlab.engine.start_matlab()
eng.eval(f"run('{script_path}')", nargout=0)



#Part D:
#Problem D.1
#Too long for 2 marks.

#Problem D.2
#When two signals are convolved, the resulting signal"s duration is equal to the sum of the durations of the original functions.

#Show Remaining plots
plt.show()
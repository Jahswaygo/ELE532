#Lab3 ELE532
#Jahmil Ally (501045419)

#Imports
import os
import time
import numpy as np
import scipy.io as sci
import matplotlib.pyplot as plt
import matlab.engine
from scipy.integrate import quad
import cmath
from sympy import symbols, cos, pi, exp, integrate

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

def Dn(x, n, period=20):
    t = symbols("t")
    w0_1 = 2 * np.pi / period
    
    if x < 0 or x > 2:
        return None
    
    n_array = np.array(n)
    Dn_array = np.zeros_like(n_array, dtype=complex)
    
    if x == 0:
        Dn_array[n_array == 1] = 1/4
        Dn_array[n_array == -1] = 1/4
        Dn_array[n_array == 3] = 1/2
        Dn_array[n_array == -3] = 1/2
        
        return Dn_array

    elif x == 1 or x == 2:
        # Handle dividing by zero
        n_non_zero = np.where(n_array != 0, n_array, np.nan)  # Replace 0 with nan
        Dn_array = 1 / (n_non_zero * np.pi)
        
        if x == 1:
            Dn_array *= np.sin((n_non_zero * np.pi) / 2)
        elif x == 2:
            Dn_array *= np.sin((n_non_zero * np.pi) / 4)
        
        # Handle the case when n = 0 (replace with the correct value if needed)
        Dn_array[np.isnan(Dn_array)] = 0  # Replace nan with 0
        return Dn_array

def plot_spectra(D_n, n_range, title):
    plt.figure(figsize=(12, 7))
    
    #Magnitude spectrum
    plt.subplot(1, 2, 1)
    plt.stem(n_range, np.abs(D_n), "k", markerfmt="ok")
    plt.xlabel("n")
    plt.ylabel("|D_n|")
    plt.title(f"Magnitude Spectrum of {title}")
    
    #Phase spectrum
    plt.subplot(1, 2, 2)
    plt.stem(n_range, np.angle(D_n), "k", markerfmt="ok")
    plt.xlabel("n")
    plt.ylabel("∠ D_n [rad]")
    plt.title(f"Phase Spectrum of {title}")
    
    plt.tight_layout()

# Part A.4
ranges = [list(range(-5, 6)), list(range(-20, 21)), list(range(-50, 51)), list(range(-500, 501))]
    
for i in range(0, 4):
    currentRange = ranges[i]
    for j in range(0, 3):
        Dn_x = Dn(j, currentRange)
        
        if Dn_x is not None:
            n = np.array(currentRange)
            title = f"x{j+1} ({n[0]} ≤ n ≤ {n[-1]})"
            plot_spectra(Dn_x, n, title)
            plt.show()

# Part A.5 & A.6

def reconstruct_signal(Dn, n_range, t, period= 20 ):# Assuming T0 is 20 for x1(t), adjust as needed for other signals
    w0 = 2 * np.pi / period  
    x_reconstructed = np.zeros_like(t, dtype=complex)
    
    for n, D in zip(n_range, Dn):
        x_reconstructed += D * np.exp(1j * n * w0 * t)
    
    return x_reconstructed.real

# Define the time vector for reconstruction
t = np.arange(-300, 301, 1)  # t from -300 to 300

# Reconstruct and plot signals for different ranges of n
for i, n_range in enumerate(ranges):
    plt.figure(figsize=(16, 12))
    
    for j in range(3):
        Dn_x = Dn(j, currentRange)
        
        if Dn_x is not None:
            x_reconstructed = reconstruct_signal(Dn_x, n_range, t)
            print(x_reconstructed)
            plt.subplot(3, 1, j+1)
            plt.plot(t, x_reconstructed, label=f"x"+str(j+1)+"(t) Reconstructed")
            plt.xlabel("t (sec)")
            plt.ylabel(f"x{j+1}(t)")
            plt.title(f"x{j+1}(t) Reconstructed from Fourier Coefficients n={n_range[0]} to {n_range[-1]})")
            plt.grid()
            plt.legend()
            
    plt.subplots_adjust(hspace=0.5)
    plt.show()
    plt.tight_layout()

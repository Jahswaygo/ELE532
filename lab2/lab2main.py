#Lab1 ELE532
#Jahmil Ally (501045419)

#Imports
import time
import numpy as np
import scipy.io as sci
import matplotlib.pyplot as plt
import sounddevice as sd
import matlab.engine

#Variable Declaration
color='g'
eng = matlab.engine.start_matlab()

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

# A. 

plt.show
import matlab.engine
eng.quit()
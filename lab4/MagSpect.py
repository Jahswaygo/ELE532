import numpy as np
import matplotlib.pyplot as plt
"""
Jahmil Ally
501045419
This Function is the MagSpect.m converted into Python. 

Credits:   Author = M. Zeytinoglu 
                Department of Electrical & Computer Engineering
                Ryerson University
                Toronto, Ontario, CANADA
"""


def MagSpect(x, Fs=32000):
    """
    MagSpect ... Utility function to simplify plotting the magnitude spectrum.

    MagSpect(x) plots the double-sided magnitude spectrum of x using 
    a 1024-point FFT; the frequency axis labels are generated based 
    on the sampling frequency Fs (default is 32 kHz). Spectral magnitude
    values are plotted in dB.
    """
    
    Nfft = 1024  # Default FFT size

    # Set up the frequency vector
    ff = np.fft.fftshift(np.fft.fftfreq(Nfft, 1 / Fs)) 

    # Compute the spectrum of x(t) using Nfft-point FFT
    Xspect = np.fft.fftshift(np.fft.fft(x, Nfft)) 

    # Plot the magnitude spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(ff, 20 * np.log10(np.abs(Xspect)))
    plt.xlim([-Fs / 2, Fs / 2])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude [dB]')
    plt.title('Magnitude Spectrum')
    plt.grid(True)
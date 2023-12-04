import os
import time
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from MagSpect import MagSpect
from numpy import fft

def saveTomachine(savepath= 'lab4\\Figures_for_B', name = "", xlabel ='Samples', ylabel='Amplitude', title= ""):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    os.makedirs(savepath, exist_ok=True)
    plt.savefig(f'{savepath}\\{name}.png')

# Path to your MATLAB file
file_path = "lab4\Lab4_Data.mat"

# Load the MATLAB file
mat_data  = scipy.io.loadmat(file_path)

# Extracting the data
xspeech = np.squeeze(mat_data['xspeech'])
hLPF2500 = np.squeeze(mat_data['hLPF2500'])
hChannel = np.squeeze(mat_data['hChannel'])
Fs = mat_data['Fs'].item() 


# Analyzing HChannel
# Frequency Response
plt.figure()
MagSpect(hChannel)
saveTomachine(savepath='lab4\Initial Analysis', name='hChannel_frequency_response', title='Frequency Response of hChannel')

# Analyzing hLPF2500
# Frequency Response
plt.figure()
MagSpect(hLPF2500)
saveTomachine(savepath='lab4\Initial Analysis', name='hLPF2500_frequency_response', title='Frequency Response of hChannel: 2500' )

# Analyzing xSpeech
plt.figure()
MagSpect(xspeech)
saveTomachine(savepath='lab4\Initial Analysis', name='Speechx_frequency_response', title='Frequency Response of xSpeech')


# Encoding Process
# Filtering 
xspeech_filtered = np.convolve(xspeech, hLPF2500, mode='same')
MagSpect(xspeech_filtered)
saveTomachine(savepath='lab4\Encoding', name='Filtered_Signal', title ='Magnitude Spectrum of Filtered Signal')

# Modulating (Amplitude Modulation)
shift_frequency = 6000  # Frequency shift in Hz
modulation_index = 20   # Adjust the modulation index as needed
t = np.arange(len(xspeech_filtered)) / Fs
print(t)

carrier_wave = np.cos(2 * np.pi * shift_frequency * t)
xspeech_am = (1 + modulation_index * xspeech_filtered) * carrier_wave
MagSpect(xspeech_am)
saveTomachine(savepath='lab4\Encoding', name='Modulated_&_Shift_Signal', title='Magnitude Spectrum of Modulated & Shifted Signal')

# Transmitting
transmitted_signal = np.convolve(xspeech_am, hChannel, mode='same')
MagSpect(transmitted_signal)
saveTomachine(savepath='lab4\Encoding', name='Transmited_Signal', title= 'Magnitude Spectrum of Transmited Signal')


# Decoding Process
# Frequency Demodulation (removing removing shift)
t = np.arange(len(transmitted_signal)) / Fs 
received_signal_demodulated = transmitted_signal * carrier_wave
MagSpect(received_signal_demodulated)
saveTomachine(savepath='lab4\Decoding', name='Demodulated_output_signal',title='Magnitude Spectrum of Demodulated Signal')

# Post-filtering (applying a filter and removing excess noise picked up from channel)
recovered_signal_filtered = np.convolve(received_signal_demodulated, hLPF2500, mode='same')
MagSpect(recovered_signal_filtered)
saveTomachine(savepath='lab4\Decoding', name='Filtered_output_signal', title= 'Magnitude Spectrum of Post-Filtered Signal')

# Normalization (Scaling makes it so it peaks @1
recovered_signal_normalized = recovered_signal_filtered / np.max(np.abs(recovered_signal_filtered))
MagSpect(recovered_signal_normalized)
saveTomachine(savepath='lab4\Decoding', name='Normalized_output_signal', title= 'Magnitude Spectrum of Post-Normalized Signal')

print('Playing 1st Audio')
sd.play(np.real(xspeech),Fs)
sd.wait()

print('Playing 2nd Audio')
sd.play(np.real(recovered_signal_normalized),Fs)
sd.wait()

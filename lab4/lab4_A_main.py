import os
import numpy as np
import matplotlib.pyplot as plt

str_omega = "\u03C9"

# Create the directory for saving figures if it doesn't exist
save_path = 'lab4\\Figures_for_A'
os.makedirs(save_path, exist_ok=True)

def plot_signal(time, signal, title, x_label, y_label, subplot_position=None, label=None, color=None):
    if subplot_position:
        plt.subplot(subplot_position)

    plt.plot(time, signal, label=label, color='red')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    if label:
        plt.legend()

def perform_convolution(signal, time):
    convolved_signal = np.convolve(signal, signal, 'full')
    conv_time = np.linspace(2*time[0], 2*time[-1], 2*len(time)-1)
    return convolved_signal, conv_time


# Problem A.1
question = "A.1"

x = lambda t: np.where((t >= 0) & (t < 10), 1, 0)
t = np.arange(-10, 30, 0.001)
x_t = x(t)  
z_t, t_conv = perform_convolution(x_t, t)

plt.figure(figsize=(14, 6))
plot_signal(t, x_t, f'{question} - Original Signal x(t)', 'Time', 'Amplitude', subplot_position=211, label='x(t)')
plot_signal(t_conv, z_t, f'{question} - Convolution of x(t) with itself', 'Time', 'Amplitude', subplot_position=212, label='z(t)')
plt.tight_layout()

# Save the figures
plt.savefig(f'{save_path}\\{question}_Original_Signal_x_t.png_&_Convolution_x_t_with_itself.png')


# Problem A.2
question = "A.2"

N, PulseWidth = 100,10
t = np.linspace(-10, 30, N)
x_t = x = np.concatenate((np.ones(PulseWidth), np.zeros(N - PulseWidth)))
x_w = np.fft.fft(x_t)
z_w = np.abs(x_w)**2

plt.figure(figsize=(14, 6))
plot_signal(t, x_t, f'{question} - Signal x(t)', 'Time', 'Amplitude', subplot_position=211, label='x(t)')
plot_signal(t, z_w, f'{question} - Signal |X({str_omega})|', 'Time', 'Amplitude', subplot_position=212, label='z(t)')
plt.tight_layout()

plt.savefig(f'{save_path}\\{question}_Signal_X_omega_&__Signal_X_omega_abs.png')


# Problem A.3
question = "A.3"

# Generate the frequency axis

f = np.fft.fftfreq(N)
f = np.fft.fftshift(f)

z_w_abs = np.abs(z_w)
z_w_ang = np.angle(z_w)

plt.figure(figsize=(14, 6))
plot_signal(f, np.fft.fftshift(z_w), f'{question} - Original Signal Z({str_omega})', 'Frequency', 'Amplitude', subplot_position=311, label=f'w({str_omega})')
plot_signal(f, np.fft.fftshift(z_w_abs), f'{question} - Signal |Z({str_omega})|', 'Frequency', 'Amplitude', subplot_position=312, label=f'w({str_omega})')
plot_signal(f, np.fft.fftshift(z_w_ang), f'{question} - Original Signal ∠Z({str_omega})', 'Frequency', 'Amplitude', subplot_position=313, label=f'w({str_omega})')
plt.tight_layout()

# Save the figures with problem numbers
plt.savefig(f'{save_path}\\{question}_Signal_Z_omega_&_Signal_Z_omega_abs_&_Signal_Z_omega_angle.png')


# Problem A.4
question = "A.4"

z_w_ifft = np.fft.ifftn(z_w)
t = np.linspace(-10, 30, len(z_w_ifft))

plt.figure(figsize=(14, 6))
plot_signal(t_conv, z_t, f'{question} - z(t) -> conv.', 'Time', 'Amplitude', subplot_position=211, label='z(t)')
plot_signal(t , np.fft.ifftshift(z_w_ifft), f'{question} - z(t) -> ifft', 'Time', 'Amplitude', subplot_position=212, label='z(t)')
plt.tight_layout()

plt.savefig(f'{save_path}\\{question}_z_t_conv_&_z_t_ifft.png')


# Problem A.5
question = "A.5"
# Pulse Widths
PulseWidths=[5, 10, 25]
N=100
for i in range(0, 3):
    PulseWidth = PulseWidths[i]
    x_t = np.concatenate((np.ones(PulseWidth), np.zeros(N - PulseWidth)))
    x_w = np.fft.fft(x_t)

    # Generate the frequency
    f = np.fft.fftfreq(N)
    f = np.fft.fftshift(f)

    x_w_abs = np.abs(x_w)
    x_w_ang = np.angle(x_w)
    

    plt.figure(figsize=(14, 6))
    plot_signal(f, np.fft.fftshift(x_w), f'{question} - Original Signal X({str_omega}), Pulse Width: {PulseWidths[i]}', 'Frequency', 'Amplitude', subplot_position=311, label=f'w({str_omega})')
    plot_signal(f, np.fft.fftshift(x_w_abs), f'{question} - Signal |X({str_omega})|, Pulse Width: {PulseWidths[i]}', 'Frequency', 'Amplitude', subplot_position=312, label=f'w({str_omega})')
    plot_signal(f, np.fft.fftshift(x_w_ang), f'{question} - Original Signal ∠X({str_omega}), Pulse Width: {PulseWidths[i]}', 'Frequency', 'Amplitude', subplot_position=313, label=f'w({str_omega})')
    plt.tight_layout()
    
    plt.savefig(f'{save_path}\\{question}_Signal_X_omega_{PulseWidths[i]}.png')
    
    
# Problem A.6
question = "A.6"
N = 100
PulseWidth = 10
t = np.arange(0, N)
x = np.concatenate((np.ones(PulseWidth), np.zeros(N - PulseWidth)))

wx = x * np.exp(1j * (np.pi/3) * t)
wy = x * np.exp(-1j * (np.pi/3) * t)
wz = x * np.cos((np.pi/3) * t)

Xf = np.fft.fft(wx)
Yf = np.fft.fft(wy)
Zf = np.fft.fft(wz)

f = np.fft.fftfreq(N, 1/N)
f_shifted = np.fft.fftshift(f)

plt.figure(figsize=(14, 8))
plot_signal(f_shifted, np.fft.fftshift(Xf), f'{question} - X({str_omega})', f'Frequency ({str_omega})', 'Amplitude', subplot_position=311)
plot_signal(f_shifted, np.fft.fftshift(np.abs(Xf)), f'{question} - |X({str_omega})|', f'Frequency ({str_omega})', 'Magnitude', subplot_position=312)
plot_signal(f_shifted, np.fft.fftshift(np.angle(Xf)), f'{question} - Angle X({str_omega})', f'Frequency ({str_omega})', 'Phase (radians)', subplot_position=313)
plt.tight_layout()
plt.savefig(f'{save_path}/{question}_Xf.png')

plt.figure(figsize=(14, 8))
plot_signal(f_shifted, np.fft.fftshift(Yf), f'{question} - Y({str_omega})', f'Frequency ({str_omega})', 'Amplitude', subplot_position=311)
plot_signal(f_shifted, np.fft.fftshift(np.abs(Yf)), f'{question} - |Y({str_omega})|', f'Frequency ({str_omega})', 'Magnitude', subplot_position=312)
plot_signal(f_shifted, np.fft.fftshift(np.angle(Yf)), f'{question} - Angle Y({str_omega})', f'Frequency ({str_omega})', 'Phase (radians)', subplot_position=313)
plt.tight_layout()
plt.savefig(f'{save_path}/{question}_Yf.png')

plt.figure(figsize=(14, 8))
plot_signal(f_shifted, np.fft.fftshift(Zf), f'{question} - Z({str_omega})', f'Frequency ({str_omega})', 'Amplitude', subplot_position=311)
plot_signal(f_shifted, np.fft.fftshift(np.abs(Zf)), f'{question} - |Z({str_omega})|', f'Frequency ({str_omega})', 'Magnitude', subplot_position=312)
plot_signal(f_shifted, np.fft.fftshift(np.angle(Zf)), f'{question} - Angle Z({str_omega})', f'Frequency ({str_omega})', 'Phase (radians)', subplot_position=313)
plt.tight_layout()
plt.savefig(f'{save_path}/{question}_Zf.png')


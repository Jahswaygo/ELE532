import numpy as np
import matplotlib.pyplot as plt

str_omega = "\u03C9"
def plot_signal(time, signal, title, x_label, y_label, subplot_position=None, label=None, color=None):
    """
    Function to plot a signal with a title and labels. Can be used for both individual plots and subplots.
    """
    if subplot_position:
        plt.subplot(subplot_position)

    plt.plot(time, signal, label=label, color='red')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    if label:
        plt.legend()

def step_signal(time, signal, title, x_label, y_label, subplot_position=None, label=None, color=None):
    """
    Function to plot a signal with a title and labels. Can be used for both individual plots and subplots.
    """
    if subplot_position:
        plt.subplot(subplot_position)

    plt.step(time, signal, label=label, color='blue')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    if label:
        plt.legend()

def perform_convolution(signal, time):
    """
    Perform convolution on the given signal and return the convolved signal and time vector.
    """
    convolved_signal = np.convolve(signal, signal, 'full')
    conv_time = np.linspace(2*time[0], 2*time[-1], 2*len(time)-1)
    return convolved_signal, conv_time

# Problem A.1
# Define the signal using a lambda function
x = lambda t: np.where((t >= 0) & (t < 10), 1, 0)

# Create a time vector from -10 to 30 with steps of 0.001
t = np.arange(-10, 30, 0.001)

# Generate the signal using the lambda function
x_t = x(t)  # Apply the lambda function to generate the signal

# Perform the convolution
z_t, t_conv = perform_convolution(x_t, t)

# Plot the original signal and the convolution result
plt.figure(figsize=(14, 6))
plot_signal(t, x_t, 'Original Signal x(t)', 'Time', 'Amplitude', subplot_position=211, label='x(t)')
plot_signal(t_conv, z_t, 'Convolution of x(t) with itself', 'Time', 'Amplitude', subplot_position=212, label='z(t)')
plt.tight_layout()

# Problem A.2
# Create a time vector from -10 to 30 with steps of 0.001
N, PulseWidth = 100,10

x_t = x = np.concatenate((np.ones(PulseWidth), np.zeros(N - PulseWidth)))
t = np.linspace(-10, 30, N)

x_w = np.fft.fft(x_t)

z_w = np.abs(x_w)**2

# Problem A.3

# Generate the frequency axis

f = np.fft.fftfreq(N)
f = np.fft.fftshift(f)

z_w_abs = np.abs(z_w)
z_w_ang = np.angle(z_w)

plt.figure(figsize=(14, 6))
plot_signal(f, np.fft.fftshift(z_w), f'Original Signal Z({str_omega})', 'Frequency', 'Amplitude', subplot_position=311, label=f'w({str_omega})')
plot_signal(f, np.fft.fftshift(z_w_abs), f'Signal |Z({str_omega})|', 'Frequency', 'Amplitude', subplot_position=312, label=f'w({str_omega})')
plot_signal(f, np.fft.fftshift(z_w_ang), f'Original Signal ∠Z({str_omega})', 'Frequency', 'Amplitude', subplot_position=313, label=f'w({str_omega})')
plt.tight_layout()

# Problem A.4
z_w_ifft = np.fft.ifftn(z_w)
t = np.linspace(-10, 30, len(z_w_ifft))


plt.figure(figsize=(14, 6))
plot_signal(t_conv, z_t, 'z(t) -> conv.', 'Time', 'Amplitude', subplot_position=211, label='z(t)')
plot_signal(2.5*(t - 6.2), np.fft.ifftshift(z_w_ifft)*1000, 'z(t) -> ifft', 'Time', 'Amplitude', subplot_position=212, label='z(t)')
plt.tight_layout()

plt.show()

# Problem A.5

# Pulse Widths
PulseWidths=[5, 10, 25]
N=100
for i in range(0, 3):
    PulseWidth = PulseWidths[i]
    x_t = np.concatenate((np.ones(PulseWidth), np.zeros(N - PulseWidth)))
    x_w = np.fft.fft(x_t)

    # Generate the frequency axis
    f = np.fft.fftfreq(N)
    f = np.fft.fftshift(f)

    x_w_abs = np.abs(x_w)
    x_w_ang = np.angle(x_w)
    

    plt.figure(figsize=(14, 6))
    plot_signal(f, np.fft.fftshift(x_w), f'Original Signal X({str_omega}), Pulse Width: {PulseWidths[i]}', 'Frequency', 'Amplitude', subplot_position=311, label=f'w({str_omega})')
    plot_signal(f, np.fft.fftshift(x_w_abs), f'Signal |X({str_omega})|, Pulse Width: {PulseWidths[i]}', 'Frequency', 'Amplitude', subplot_position=312, label=f'w({str_omega})')
    plot_signal(f, np.fft.fftshift(x_w_ang), f'Original Signal ∠X({str_omega}), Pulse Width: {PulseWidths[i]}', 'Frequency', 'Amplitude', subplot_position=313, label=f'w({str_omega})')
    plt.tight_layout()

plt.show()


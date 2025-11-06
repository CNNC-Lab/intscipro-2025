# FFT (scipy.fft)
from scipy import fft
import numpy as np
import matplotlib.pyplot as plt
import os

# Setup for saving figures
output_dir = os.path.dirname(__file__)

# Generate signal
t = np.linspace(0, 1, 1000)
signal_data = np.sin(2*np.pi*5*t)

# Compute FFT
spectrum = fft.fft(signal_data)
frequencies = fft.fftfreq(len(signal_data), 1/1000)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Time domain
ax1.plot(t, signal_data)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Amplitude')
ax1.set_title('Signal (Time Domain)')
ax1.grid(True, alpha=0.3)

# Frequency domain
ax2.plot(frequencies[:len(frequencies)//2], 
         np.abs(spectrum[:len(spectrum)//2]))
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Magnitude')
ax2.set_title('FFT (Frequency Domain)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig20_fft.png'), dpi=150, bbox_inches='tight')
plt.show()
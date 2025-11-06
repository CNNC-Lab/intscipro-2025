# Signal processing (scipy.signal)
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import os

# Setup for saving figures
output_dir = os.path.dirname(__file__)

# Generate a noisy signal with multiple frequency components
fs = 1000  # sampling frequency
t = np.linspace(0, 1, fs)
np.random.seed(42)

# Create signal: 5 Hz + 25 Hz + 100 Hz + noise
signal_clean = (np.sin(2*np.pi*5*t) + 
                0.5*np.sin(2*np.pi*25*t) + 
                0.3*np.sin(2*np.pi*100*t))
noise = 0.5*np.random.randn(len(t))
signal_noisy = signal_clean + noise

# Design a bandpass filter
lowcut = 10
highcut = 50
b, a = signal.butter(4, [lowcut, highcut], 
                     btype='band', fs=fs)

# Apply filter
filtered = signal.filtfilt(b, a, signal_noisy)

# Compute frequency response of filter
w, h = signal.freqz(b, a, worN=2000, fs=fs)

# Compute power spectral density
freqs_orig, psd_orig = signal.welch(signal_noisy, fs=fs, nperseg=256)
freqs_filt, psd_filt = signal.welch(filtered, fs=fs, nperseg=256)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Time domain - original and filtered signals
axes[0, 0].plot(t[:200], signal_noisy[:200], 'r-', alpha=0.5, linewidth=1, label='Noisy')
axes[0, 0].plot(t[:200], filtered[:200], 'b-', linewidth=2, label='Filtered')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Amplitude')
axes[0, 0].set_title('Signal Filtering (Time Domain)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Filter frequency response
axes[0, 1].plot(w, 20*np.log10(abs(h)), 'b-', linewidth=2)
axes[0, 1].axvline(lowcut, color='r', linestyle='--', alpha=0.7, label='Cutoff frequencies')
axes[0, 1].axvline(highcut, color='r', linestyle='--', alpha=0.7)
axes[0, 1].set_xlabel('Frequency (Hz)')
axes[0, 1].set_ylabel('Magnitude (dB)')
axes[0, 1].set_title('Bandpass Filter Frequency Response')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xlim(0, 150)

# Plot 3: Power spectral density
axes[1, 0].semilogy(freqs_orig, psd_orig, 'r-', alpha=0.7, linewidth=2, label='Original')
axes[1, 0].semilogy(freqs_filt, psd_filt, 'b-', linewidth=2, label='Filtered')
axes[1, 0].axvline(5, color='g', linestyle=':', alpha=0.5, label='Signal frequencies')
axes[1, 0].axvline(25, color='g', linestyle=':', alpha=0.5)
axes[1, 0].axvline(100, color='g', linestyle=':', alpha=0.5)
axes[1, 0].set_xlabel('Frequency (Hz)')
axes[1, 0].set_ylabel('Power Spectral Density')
axes[1, 0].set_title('Frequency Content')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xlim(0, 150)

# Plot 4: Spectrogram
f, t_spec, Sxx = signal.spectrogram(signal_noisy, fs=fs, nperseg=128)
im = axes[1, 1].pcolormesh(t_spec, f, 10*np.log10(Sxx), shading='gouraud', cmap='viridis')
axes[1, 1].set_ylabel('Frequency (Hz)')
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_title('Spectrogram (Original Signal)')
axes[1, 1].set_ylim(0, 150)
plt.colorbar(im, ax=axes[1, 1], label='Power (dB)')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig19_scipy_signal.png'), dpi=150, bbox_inches='tight')
plt.show()
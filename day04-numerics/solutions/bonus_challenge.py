# Bonus Challenge: Integrated Analysis
import numpy as np
from scipy import signal, stats
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# 1. Generate synthetic sensor data
fs = 100  # Sampling rate (Hz)
duration = 10  # seconds
t = np.linspace(0, duration, fs * duration)

# True signal: combination of two frequencies
freq1, freq2 = 2, 8  # Hz
true_signal = (np.sin(2*np.pi*freq1*t) + 
               0.5*np.sin(2*np.pi*freq2*t))

# 2. Add realistic noise
noise = 0.5 * np.random.randn(len(t))
noisy_signal = true_signal + noise

# 3. Spectral analysis
freqs, psd = signal.welch(noisy_signal, fs=fs, nperseg=256)

# Find dominant frequencies
peak_indices = signal.find_peaks(psd, height=0.01)[0]
dominant_freqs = freqs[peak_indices]
print(f"Dominant frequencies detected: {dominant_freqs[:5]} Hz")

# 4. Design filter (bandpass: 1-10 Hz)
sos = signal.butter(4, [1, 10], btype='band', fs=fs, output='sos')
filtered_signal = signal.sosfiltfilt(sos, noisy_signal)

# 5. Calculate performance metrics
mse_noisy = np.mean((true_signal - noisy_signal)**2)
mse_filtered = np.mean((true_signal - filtered_signal)**2)
snr_improvement = 10 * np.log10(mse_noisy / mse_filtered)

print(f"\nPerformance Metrics:")
print(f"MSE (noisy): {mse_noisy:.4f}")
print(f"MSE (filtered): {mse_filtered:.4f}")
print(f"SNR improvement: {snr_improvement:.2f} dB")

# 6. Comprehensive visualization
fig = plt.figure(figsize=(15, 10))

# Time domain comparison
plt.subplot(2, 2, 1)
plt.plot(t[:200], true_signal[:200], 'g-', linewidth=2, label='True signal', alpha=0.7)
plt.plot(t[:200], noisy_signal[:200], 'r-', linewidth=0.5, label='Noisy signal', alpha=0.5)
plt.plot(t[:200], filtered_signal[:200], 'b-', linewidth=2, label='Filtered signal')
plt.xlabel('Time (s)', fontsize=11, fontweight='bold')
plt.ylabel('Amplitude', fontsize=11, fontweight='bold')
plt.title('Signal Comparison', fontsize=13, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Power spectral density
plt.subplot(2, 2, 2)
freqs_noisy, psd_noisy = signal.welch(noisy_signal, fs=fs, nperseg=256)
freqs_filtered, psd_filtered = signal.welch(filtered_signal, fs=fs, nperseg=256)
plt.semilogy(freqs_noisy, psd_noisy, 'r-', alpha=0.7, label='Noisy')
plt.semilogy(freqs_filtered, psd_filtered, 'b-', linewidth=2, label='Filtered')
plt.axvline(freq1, color='g', linestyle='--', alpha=0.7, label=f'{freq1} Hz')
plt.axvline(freq2, color='g', linestyle='--', alpha=0.7, label=f'{freq2} Hz')
plt.xlabel('Frequency (Hz)', fontsize=11, fontweight='bold')
plt.ylabel('PSD', fontsize=11, fontweight='bold')
plt.title('Power Spectral Density', fontsize=12, fontweight='bold')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)
plt.xlim(0, 20)

# Filter frequency response
plt.subplot(2, 2, 3)
w, h = signal.sosfreqz(sos, worN=2000, fs=fs)
plt.plot(w, 20*np.log10(abs(h)), 'b-', linewidth=2)
plt.axvline(1, color='r', linestyle='--', alpha=0.7, label='Cutoffs')
plt.axvline(10, color='r', linestyle='--', alpha=0.7)
plt.xlabel('Frequency (Hz)', fontsize=11, fontweight='bold')
plt.ylabel('Magnitude (dB)', fontsize=11, fontweight='bold')
plt.title('Filter Frequency Response', fontsize=12, fontweight='bold')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)
plt.xlim(0, 20)

# Error analysis
plt.subplot(2, 2, 4)
error_noisy = noisy_signal - true_signal
error_filtered = filtered_signal - true_signal
plt.hist(error_noisy, bins=50, alpha=0.5, label='Noisy error', color='red', density=True)
plt.hist(error_filtered, bins=50, alpha=0.5, label='Filtered error', color='blue', density=True)
plt.xlabel('Error', fontsize=11, fontweight='bold')
plt.ylabel('Density', fontsize=11, fontweight='bold')
plt.title('Error Distribution', fontsize=12, fontweight='bold')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3, axis='y')

fig.suptitle('Comprehensive Signal Analysis and Filtering', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# 7. Statistical report
print("\n" + "="*60)
print("STATISTICAL ANALYSIS REPORT")
print("="*60)
print(f"\nSignal Parameters:")
print(f"  Duration: {duration} seconds")
print(f"  Sampling rate: {fs} Hz")
print(f"  Number of samples: {len(t)}")
print(f"  True frequencies: {freq1} Hz, {freq2} Hz")

print(f"\nNoise Characteristics:")
print(f"  Noise std: {noise.std():.4f}")
print(f"  Signal std: {true_signal.std():.4f}")
print(f"  Original SNR: {10*np.log10(np.var(true_signal)/np.var(noise)):.2f} dB")

print(f"\nFilter Performance:")
print(f"  Filter type: Butterworth bandpass")
print(f"  Order: 4")
print(f"  Passband: 1-10 Hz")
print(f"  SNR improvement: {snr_improvement:.2f} dB")

print(f"\nAccuracy Metrics:")
print(f"  Noisy signal:")
print(f"    MSE: {mse_noisy:.4f}")
print(f"    RMSE: {np.sqrt(mse_noisy):.4f}")
print(f"    Correlation with true: {np.corrcoef(true_signal, noisy_signal)[0,1]:.4f}")
print(f"  Filtered signal:")
print(f"    MSE: {mse_filtered:.4f}")
print(f"    RMSE: {np.sqrt(mse_filtered):.4f}")
print(f"    Correlation with true: {np.corrcoef(true_signal, filtered_signal)[0,1]:.4f}")
print(f"    Improvement: {(1 - mse_filtered/mse_noisy)*100:.1f}%")

print("\n" + "="*60)

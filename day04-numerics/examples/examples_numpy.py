import numpy as np
import matplotlib.pyplot as plt

# Basic Signal Processing
# Convolution (filtering, smoothing)
signal = np.random.randn(1000)
kernel = np.ones(10) / 10  # moving average
smoothed = np.convolve(signal, kernel, mode='same')
# plot
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(signal, 'k', alpha=0.5, linewidth=0.8, label='Original Signal')
ax.plot(smoothed, 'r', linewidth=2, label='Smoothed (MA-10)')
ax.set_xlim(0, len(signal) - 1)
ax.set_xlabel('Sample Index')
ax.set_ylabel('Amplitude')
ax.set_title('Signal Smoothing via Moving Average')
ax.legend(frameon=True, loc='upper right')
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('01_signal_smoothing.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: 01_signal_smoothing.png')

# Correlation
signal1 = np.random.randn(100)
signal2 = np.random.randn(100)
correlation = np.correlate(signal1, signal2, mode='full')
# plot
fig, ax = plt.subplots(figsize=(10, 4))
lags = np.arange(-len(signal1) + 1, len(signal1))
ax.plot(lags, correlation, 'b', linewidth=1.5)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
ax.set_xlim(lags[0], lags[-1])
ax.set_xlabel('Lag')
ax.set_ylabel('Cross-correlation')
ax.set_title('Cross-correlation between Signal 1 and Signal 2')
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('02_cross_correlation.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: 02_cross_correlation.png')

# Correlation coefficient matrix
data_matrix = np.random.randn(5, 100)  # 5 channels
corr_matrix = np.corrcoef(data_matrix)
# plot
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
ax.set_xlabel('Channel')
ax.set_ylabel('Channel')
ax.set_title('Correlation Coefficient Matrix')
ax.set_xticks(range(len(corr_matrix)))
ax.set_yticks(range(len(corr_matrix)))
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Correlation', rotation=270, labelpad=15)
plt.tight_layout()
plt.savefig('03_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: 03_correlation_matrix.png')

# FFT (Fast Fourier Transform)
# Note: use scipy.fft for more advanced features
sampling_rate = 1000  # Hz
dt = 1 / sampling_rate
time = np.arange(len(signal)) * dt
fft_result = np.fft.fft(signal)
frequencies = np.fft.fftfreq(len(signal), d=dt)
power_spectrum = np.abs(fft_result)**2

# plot: 3-panel figure
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

# Panel 1: Time series
axes[0].plot(time, signal, 'k', linewidth=0.5, alpha=0.7)
axes[0].set_xlim(0, time[-1])
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Amplitude')
axes[0].set_title('Original Signal (Time Domain)')
axes[0].grid(True, alpha=0.3, linestyle='--')

# Panel 2: Power spectrum
# Only plot positive frequencies
pos_mask = frequencies >= 0
axes[1].plot(frequencies[pos_mask], power_spectrum[pos_mask], 'g', linewidth=1.5)
axes[1].set_xlim(0, 50)
axes[1].set_xlabel('Frequency (Hz)')
axes[1].set_ylabel('Power')
axes[1].set_title('Power Spectrum')
axes[1].grid(True, alpha=0.3, linestyle='--')

# Panel 3: Spectrogram (manual implementation with numpy)
nperseg = 256
noverlap = nperseg // 2
hop = nperseg - noverlap
n_segments = (len(signal) - noverlap) // hop
Sxx = np.zeros((nperseg // 2 + 1, n_segments))
t_spec = np.zeros(n_segments)
f_spec = np.fft.rfftfreq(nperseg, d=dt)

for i in range(n_segments):
    start = i * hop
    segment = signal[start:start + nperseg]
    if len(segment) == nperseg:
        windowed = segment * np.hanning(nperseg)
        fft_seg = np.fft.rfft(windowed)
        Sxx[:, i] = np.abs(fft_seg)**2
        t_spec[i] = (start + nperseg // 2) * dt

Sxx_db = 10 * np.log10(Sxx + 1e-10)  # Add small value to avoid log(0)
im = axes[2].pcolormesh(t_spec, f_spec, Sxx_db, shading='gouraud', cmap='viridis')
axes[2].set_ylim(0, 50)
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Frequency (Hz)')
axes[2].set_title('Spectrogram')
cbar = plt.colorbar(im, ax=axes[2])
cbar.set_label('Power (dB)', rotation=270, labelpad=15)

plt.tight_layout()
plt.savefig('04_fft_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: 04_fft_analysis.png')

# Neuroscience example: connectivity analysis
eeg_channels = np.random.randn(64, 1000)
connectivity = np.corrcoef(eeg_channels)

# plot: 2-panel figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Sample of EEG channels
n_channels_to_plot = 10
time_eeg = np.arange(eeg_channels.shape[1]) / sampling_rate
offset = 4  # vertical offset between channels for visibility

for i in range(n_channels_to_plot):
    axes[0].plot(time_eeg, eeg_channels[i, :] + i * offset, linewidth=0.8, 
                 label=f'Ch {i+1}' if i < 5 else '')  # Only label first 5 to avoid clutter

axes[0].set_xlim(0, time_eeg[-1])
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Channel (offset for visibility)')
axes[0].set_title(f'EEG Channels (showing {n_channels_to_plot}/{eeg_channels.shape[0]})')
axes[0].grid(True, alpha=0.3, linestyle='--')
axes[0].set_yticks([])

# Panel 2: Connectivity matrix
im = axes[1].imshow(connectivity, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
axes[1].set_xlabel('Channel')
axes[1].set_ylabel('Channel')
axes[1].set_title(f'Connectivity Matrix ({eeg_channels.shape[0]} channels)')
cbar = plt.colorbar(im, ax=axes[1])
cbar.set_label('Correlation', rotation=270, labelpad=15)

plt.tight_layout()
plt.savefig('05_eeg_connectivity.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: 05_eeg_connectivity.png')
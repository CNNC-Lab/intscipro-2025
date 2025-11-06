# Colors, line styles, markers
import matplotlib.pyplot as plt
import numpy as np
import os

# Setup for saving figures
output_dir = os.path.dirname(__file__)

x = np.linspace(0, 10, 100)
y = np.sin(x)

# Figure size and DPI
plt.figure(figsize=(10, 6), dpi=100)

# Color options: 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w'
# Or RGB: (0.5, 0.2, 0.8) or hex: '#FF5733'
plt.plot(x, y, color='#80203c', linewidth=2)

# Line styles: '-', '--', '-.', ':'
plt.plot(x, y+0.5, linestyle='--', linewidth=2)

# Markers: 'o', 's', '^', 'v', '*', '+', 'x', 'D'
plt.plot(x[::10], y[::10], 'o', markersize=8,
         markerfacecolor='red', 
         markeredgecolor='black',
         markeredgewidth=1.5)

# Transparency
plt.plot(x, y-0.5, alpha=0.5)

# Labels and title with font size
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Amplitude (mV)', fontsize=14)
plt.title('Customized Plot', fontsize=16, 
          fontweight='bold')

# Axis limits
plt.xlim(0, 10)
plt.ylim(-2, 2)

# Grid and spines
plt.grid(True, linestyle='--', alpha=0.7)
ax = plt.gca()  # get current axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend customization
# plt.legend(loc='upper right', 
#            frameon=True,
#            shadow=True,
#            fontsize=12)

# Annotations
plt.annotate('Peak', xy=(1.5, 1), 
             xytext=(3, 1.3),
             arrowprops=dict(arrowstyle='->'))

# Tight layout (prevents label cutoff)
plt.tight_layout()

# Save figure
plt.savefig(os.path.join(output_dir, 'fig8.png'), dpi=300, bbox_inches='tight')

# Style sheets
# plt.style.use('seaborn-v0_8-darkgrid')
# Options: 'ggplot', 'seaborn', 'bmh', etc.
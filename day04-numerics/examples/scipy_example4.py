# Interpolation (scipy.interpolate)
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
import os

# Setup for saving figures
output_dir = os.path.dirname(__file__)

# 1D interpolation
x = np.linspace(0, 10, 10)
y = np.sin(x)
f = interpolate.interp1d(x, y, kind='cubic')
x_new = np.linspace(0, 10, 100)
y_new = f(x_new)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'o', markersize=8, label='Original data points', color='red')
plt.plot(x_new, y_new, '-', linewidth=2, label='Cubic interpolation', color='blue')
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('1D Interpolation Example', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig21_interpolation.png'), dpi=150, bbox_inches='tight')
plt.show()

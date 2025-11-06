# Optimization (scipy.optimize)
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
import os

# Setup for saving figures
output_dir = os.path.dirname(__file__)

# Curve fitting
def func(x, a, b):
    return a * np.exp(-b * x)

np.random.seed(42)
x_data = np.linspace(0, 4, 50)
y_data = func(x_data, 2.5, 1.3) + 0.2*np.random.randn(50)
params, covariance = optimize.curve_fit(func, x_data, y_data)

# Find minimum of function
def parabola(x):
    return (x-2)**2

result = optimize.minimize(parabola, x0=0)

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Curve fitting
x_fit = np.linspace(0, 4, 200)
y_fit = func(x_fit, *params)
ax1.scatter(x_data, y_data, alpha=0.6, s=30, label='Data with noise')
ax1.plot(x_fit, y_fit, 'r-', linewidth=2, 
         label=f'Fitted: a={params[0]:.2f}, b={params[1]:.2f}')
ax1.plot(x_fit, func(x_fit, 2.5, 1.3), 'g--', linewidth=2, 
         label='True: a=2.5, b=1.3', alpha=0.7)
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_title('Curve Fitting (Exponential Decay)', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Function minimization
x_range = np.linspace(-1, 5, 200)
y_range = parabola(x_range)
ax2.plot(x_range, y_range, 'b-', linewidth=2, label='f(x) = (x-2)²')
ax2.plot(result.x, result.fun, 'ro', markersize=12, 
         label=f'Minimum at x={result.x[0]:.3f}')
ax2.axvline(result.x, color='r', linestyle='--', alpha=0.5)
ax2.axhline(result.fun, color='r', linestyle='--', alpha=0.5)
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('f(x)', fontsize=12)
ax2.set_title('Function Minimization', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig22_optimization.png'), dpi=150, bbox_inches='tight')
plt.show()
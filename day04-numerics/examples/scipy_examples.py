# Statistics (scipy.stats)
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import os

# Setup for saving figures
output_dir = os.path.dirname(__file__)

# T-test
np.random.seed(42)
data1 = np.random.randn(100)
data2 = np.random.randn(100) + 0.5
t_stat, p_value = stats.ttest_ind(data1, data2)

# Normal distribution
x = np.linspace(-3, 3, 100)
pdf = stats.norm.pdf(x, loc=0, scale=1)

# Correlation
r, p = stats.pearsonr(data1[:50], data2[:50])

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: T-test comparison
axes[0, 0].hist(data1, bins=20, alpha=0.5, label=f'Group 1 (μ={data1.mean():.2f})')
axes[0, 0].hist(data2, bins=20, alpha=0.5, label=f'Group 2 (μ={data2.mean():.2f})')
axes[0, 0].set_xlabel('Value')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title(f'T-test: t={t_stat:.3f}, p={p_value:.4f}')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Normal distribution PDF
axes[0, 1].plot(x, pdf, 'b-', linewidth=2, label='PDF')
axes[0, 1].fill_between(x, pdf, alpha=0.3)
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('Probability Density')
axes[0, 1].set_title('Normal Distribution N(0,1)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Correlation scatter plot
axes[1, 0].scatter(data1[:50], data2[:50], alpha=0.6)
axes[1, 0].set_xlabel('Data 1')
axes[1, 0].set_ylabel('Data 2')
axes[1, 0].set_title(f'Correlation: r={r:.3f}, p={p:.4f}')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Multiple distributions
x_range = np.linspace(-5, 5, 200)
axes[1, 1].plot(x_range, stats.norm.pdf(x_range, 0, 1), label='N(0,1)')
axes[1, 1].plot(x_range, stats.norm.pdf(x_range, 0, 0.5), label='N(0,0.5)')
axes[1, 1].plot(x_range, stats.norm.pdf(x_range, 1, 1), label='N(1,1)')
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('Probability Density')
axes[1, 1].set_title('Different Normal Distributions')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig18_scipy_stats.png'), dpi=150, bbox_inches='tight')
plt.show()
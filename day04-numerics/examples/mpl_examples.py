# Simple line plot
import matplotlib.pyplot as plt
import numpy as np
import os

# Setup for saving figures
output_dir = os.path.dirname(__file__)

# Generate data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create plot
plt.plot(x, y)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Sine Wave')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'fig1.png'), dpi=150, bbox_inches='tight')
plt.show()

# Multiple lines
y2 = np.cos(x)
plt.plot(x, y, label='sin(x)')
plt.plot(x, y2, label='cos(x)')
plt.legend()
plt.savefig(os.path.join(output_dir, 'fig2.png'), dpi=150, bbox_inches='tight')
plt.show()

# Line styles and markers
plt.plot(x, y, 'r--', label='red dashed')
plt.plot(x, y2, 'b.', label='blue dots')
plt.plot(x, y+0.5, 'go-', label='green line+circles')
plt.legend()
plt.savefig(os.path.join(output_dir, 'fig3.png'), dpi=150, bbox_inches='tight')
plt.show()


# Scatter plot
x = np.random.randn(100)
y = np.random.randn(100)
plt.scatter(x, y, alpha=0.5)
plt.xlabel('Variable 1')
plt.ylabel('Variable 2')
plt.savefig(os.path.join(output_dir, 'fig4.png'), dpi=150, bbox_inches='tight')
plt.show()

# Bar plot
categories = ['A', 'B', 'C', 'D']
values = [23, 45, 56, 78]
plt.bar(categories, values)
plt.ylabel('Count')
plt.savefig(os.path.join(output_dir, 'fig5.png'), dpi=150, bbox_inches='tight')
plt.show()

# Histogram
data = np.random.randn(1000)
plt.hist(data, bins=30, edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.savefig(os.path.join(output_dir, 'fig6.png'), dpi=150, bbox_inches='tight')
plt.show()

# Subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].plot(x, y)
axes[0, 1].scatter(x, y)
axes[1, 0].hist(data, bins=20)
axes[1, 1].bar(categories, values)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig7.png'), dpi=150, bbox_inches='tight')
plt.show()
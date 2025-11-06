import matplotlib.pyplot as plt
import numpy as np
import os

# Setup for saving figures
output_dir = os.path.dirname(__file__)

# Generate sample data
np.random.seed(42)
data = np.random.normal(15, 10, 1000)

# Basic histogram
plt.hist(data)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Basic Histogram')
plt.savefig(os.path.join(output_dir, 'fig9_basic_hist.png'), dpi=150, bbox_inches='tight')
plt.show()

# Specify number of bins
plt.hist(data, bins=30)
plt.savefig(os.path.join(output_dir, 'fig10_bins.png'), dpi=150, bbox_inches='tight')
plt.show()

# Specify bin edges
bins = np.arange(0, 50, 2)
plt.hist(data, bins=bins)
plt.savefig(os.path.join(output_dir, 'fig11_bin_edges.png'), dpi=150, bbox_inches='tight')
plt.show()

# Normalized histogram (density)
plt.hist(data, bins=30, density=True)
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.savefig(os.path.join(output_dir, 'fig12_density.png'), dpi=150, bbox_inches='tight')
plt.show()

# Customize appearance
plt.hist(data, bins=25, 
         edgecolor='black',
         facecolor='skyblue',
         alpha=0.7)
plt.savefig(os.path.join(output_dir, 'fig13_custom.png'), dpi=150, bbox_inches='tight')
plt.show()

# Multiple histograms (overlaid)
data1 = np.random.normal(15, 10, 1000)
data2 = np.random.normal(20, 8, 1000)

plt.hist(data1, bins=30, alpha=0.5, label='Group 1')
plt.hist(data2, bins=30, alpha=0.5, label='Group 2')
plt.legend()
plt.savefig(os.path.join(output_dir, 'fig14_multiple.png'), dpi=150, bbox_inches='tight')
plt.show()

# Cumulative histogram
plt.hist(data, bins=30, cumulative=True)
plt.ylabel('Cumulative Count')
plt.savefig(os.path.join(output_dir, 'fig15_cumulative.png'), dpi=150, bbox_inches='tight')
plt.show()

# Horizontal histogram
plt.hist(data, bins=30, orientation='horizontal')
plt.savefig(os.path.join(output_dir, 'fig16_horizontal.png'), dpi=150, bbox_inches='tight')
plt.show()

# 2D histogram (heatmap)
x = np.random.randn(1000)
y = np.random.randn(1000)
plt.hist2d(x, y, bins=30, cmap='Blues')
plt.colorbar(label='Count')
plt.xlabel('X Variable')
plt.ylabel('Y Variable')
plt.savefig(os.path.join(output_dir, 'fig17_2d_hist.png'), dpi=150, bbox_inches='tight')
plt.show()
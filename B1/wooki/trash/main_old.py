import sys
import os

# Add folder to path using absolute path
data_folder = os.path.abspath('A2')

print(f"print {data_folder}")
sys.path.insert(0, data_folder)

# Import dictionary
from A2 import all_data

# print(f"Data loaded from A2.py:{all_data}")

from B1.wooki.esn_old import *
# Use directly
# esn = EchoStateNetwork()
# nrmse = calculate_nrmse(y_true, y_pred)

# Run hyperparameter analysis
print("Running hyperparameter analysis...")
fig = hyperparameter_analysis_plot(all_data, n_trials=2)

# Also show sample time series
fig2, axes = plt.subplots(2, 2, figsize=(12, 8))
fig2.suptitle('Sample Time Series Data', fontsize=14)

for idx, (name, series) in enumerate(all_data.items()):
    ax = axes[idx//2, idx%2]
    ax.plot(series[:1000], linewidth=1)  # Plot first 1000 points
    ax.set_title(name)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Time steps')
    ax.set_ylabel('Value')

plt.tight_layout()
plt.show()

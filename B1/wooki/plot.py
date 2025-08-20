import matplotlib.pyplot as plt
from main2 import *
import os

base_dir = os.path.dirname(os.path.abspath(__file__))  # script folder

# Plot
for label in problems:
    # Create a folder for this label if it doesn't exist
    label_dir = os.path.join(base_dir, f"Regime_{label}")
    os.makedirs(label_dir, exist_ok=True)

    for param in ['Nres', 'p', 'alpha', 'rho']:
        x_vals, y_vals = results[label][param]
        plt.figure()
        plt.plot(x_vals, y_vals, marker='o')
        plt.xlabel(param)
        plt.ylabel("NRMSE")
        plt.title(f"NRMSE vs {param} (Regime {label})")
        plt.grid(True)
        plt.tight_layout()

        # Save plots
        # script_dir = os.path.dirname(os.path.abspath(__file__))  # folder of the script
        # plt.savefig(os.path.join(script_dir, f"Figure_B1_{param}.png"), dpi=300)

        # Save the plot inside the label folder
        filename = f"{param}.png"
        plt.savefig(os.path.join(label_dir, filename), dpi=300)
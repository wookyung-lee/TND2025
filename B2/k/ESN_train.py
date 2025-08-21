import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from ESN import *
import pickle



# Import dictionary
data_folder = os.path.abspath('A2')
sys.path.insert(0, data_folder)
from A2 import all_data

# Hyperparameters
Nres_list = np.linspace(300, 1000, 10, dtype=int)
p_list = np.linspace(0.1, 1.0, 10)
alpha_list = np.linspace(0.1, 1.0, 10)
rho_list = np.linspace(0.1, 1.5, 10)

problems = ["a", "b", "c", "e"]
transient_pts = 40000  # discard transient
results = {}
optimized_params = {}  # To store optimized hyperparameters

for label in problems:
    data = all_data[label]['x']

    # Remove transient and normalize
    u = data[transient_pts:]
    u = (u - np.mean(u)) / np.std(u)

    # Split 50% for training
    N_train = len(u) // 2
    u_train = u[:N_train]
    y_train = u[:N_train]

    results[label] = {}
    optimized_params[label] = {}

    # NRMSE for Nres
    nrmse_list = []
    for Nres in Nres_list:
        esn = ESN(Nres=Nres, p=0.75, alpha=0.5, rho=0.85, random_state=42)
        nrmse = esn.train(u_train, y_train)
        nrmse_list.append(nrmse)
    results[label]['Nres'] = (Nres_list, nrmse_list)
    # Store optimized Nres for this label
    opt_idx = np.argmin(nrmse_list)
    optimized_params[label]['Nres'] = Nres_list[opt_idx]
    print(f"Nres done, {label}, best: {optimized_params[label]['Nres']}")

    # NRMSE for p
    nrmse_list = []
    for p in p_list:
        esn = ESN(Nres=300, p=p, alpha=0.5, rho=0.85, random_state=42)
        nrmse = esn.train(u_train, y_train)
        nrmse_list.append(nrmse)
    results[label]['p'] = (p_list, nrmse_list)
    # Store optimized p for this label
    opt_idx = np.argmin(nrmse_list)
    optimized_params[label]['p'] = p_list[opt_idx]
    print(f"p done, {label}, best: {optimized_params[label]['p']}")

    # NRMSE for alpha
    nrmse_list = []
    for alpha in alpha_list:
        esn = ESN(Nres=300, p=0.75, alpha=alpha, rho=0.85, random_state=42)
        nrmse = esn.train(u_train, y_train)
        nrmse_list.append(nrmse)
    results[label]['alpha'] = (alpha_list, nrmse_list)
    # Store optimized alpha for this label
    opt_idx = np.argmin(nrmse_list)
    optimized_params[label]['alpha'] = alpha_list[opt_idx]
    print(f"alpha done, {label}, best: {optimized_params[label]['alpha']}")

    # NRMSE for rho
    nrmse_list = []
    for rho in rho_list:
        esn = ESN(Nres=300, p=0.75, alpha=0.5, rho=rho, random_state=42)
        nrmse = esn.train(u_train, y_train)
        nrmse_list.append(nrmse)
    results[label]['rho'] = (rho_list, nrmse_list)
    # Store optimized rho for this label
    opt_idx = np.argmin(nrmse_list)
    optimized_params[label]['rho'] = rho_list[opt_idx]
    print(f"rho done, {label}, best: {optimized_params[label]['rho']}")

# script folder
base_dir = os.path.dirname(os.path.abspath(__file__))

# Plot
for label in problems:
    # Create a folder for this label if it doesn't exist
    label_dir = os.path.join(base_dir, f"A2_{label}")
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

        # Save the plot inside the label folder
        filename = f"{param}.png"
        plt.savefig(os.path.join(label_dir, filename), dpi=300)

# Save optimized parameters to a file
with open("optimized_params.pkl", "wb") as f:
    pickle.dump(optimized_params, f)
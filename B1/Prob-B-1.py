import sys, os
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from ESN import ESN
import torch
import pickle

# Preparation
seed = 42
np.random.seed(seed)
train_fraction = 0.5
dt = 0.005
Tt = 40000  # transient points to discard

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


# Import dataset
data_folder = os.path.abspath('A2')
sys.path.insert(0, data_folder)
from A2 import all_data

# Function to compute NRMSE for a given hyperparameter
def compute_nrmse(param_name, param_value, u_train, y_train):
    # kwargs = {'Nres': 600, 'p': 0.7, 'alpha': 0.5, 'rho': 0.9}
    kwargs = {'Nres': 850, 'p': 0.7, 'alpha': 0.1, 'rho': 0.85}
    kwargs[param_name] = param_value
    esn = ESN(**kwargs)
    return esn.train(u_train, y_train)

# Hyperparameter ranges
Nres_list = np.linspace(300, 1000, 10, dtype=int)
p_list = np.linspace(0.1, 1.0, 10)
alpha_list = np.linspace(0.1, 1.0, 10)
rho_list = np.linspace(0.1, 1.5, 10)

problems = ["a", "b", "c", "e"]
transient_pts = Tt
results = {}

# Loop through problems and hyperparameters
for label in problems:
    data = all_data[label]['x']

    # Remove transient and normalize
    u = data[transient_pts:]
    u = (u - np.mean(u)) / np.std(u)

    # Split 50% for training
    N_train = len(u) // 2
    u_train = u[:N_train-1]
    y_train = u[1:N_train]

    results[label] = {}

    for param_name, param_list in zip(['Nres', 'p', 'alpha', 'rho'], [Nres_list, p_list, alpha_list, rho_list]):
        # Evaluate NRMSE for all values in parallel
        nrmse_list = Parallel(n_jobs=2)(
            delayed(compute_nrmse)(param_name, val, u_train, y_train) for val in param_list
        )
        results[label][param_name] = (param_list, nrmse_list)
        print(f"{param_name} done, {label}")

# Plotting
base_dir = os.path.dirname(os.path.abspath(__file__))
for label in problems:
    label_dir = os.path.join(base_dir, f"A-2{label}")
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
        filename = f"{param}.png"
        plt.savefig(os.path.join(label_dir, filename), dpi=300)
        plt.close()

optimal_params = {}
for label in problems:
    optimal_params[label] = {}
    for param in ['Nres', 'p', 'alpha', 'rho']:
        x_vals, y_vals = results[label][param]
        min_idx = np.argmin(y_vals)
        optimal_params[label][param] = (x_vals[min_idx], y_vals[min_idx])

# Save to pickle
with open("optimal_hyperparams.pkl", "wb") as f:
    pickle.dump(optimal_params, f)

# Load back and print
with open("optimal_hyperparams.pkl", "rb") as f:
    loaded_optimal = pickle.load(f)

print("\nOptimal hyperparameters (with min NRMSE) for each regime:")
print(loaded_optimal)


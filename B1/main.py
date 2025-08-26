import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from ESN import * 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import dictionary
data_folder = os.path.abspath('A2')
sys.path.insert(0, data_folder)
from A2 import all_data

# For parallel computing
def compute_nrmse(param_name, param_value, u_train, y_train):
    kwargs = {'Nres': 300, 'p': 0.75, 'alpha': 0.5, 'rho': 0.9, 'random_state': 42} # rho in range [0.8,1.2]
    kwargs[param_name] = param_value
    esn = ESN(**kwargs)    

    return esn.train(u_train, y_train)

print(f"device: {device}")

# Hyperparameters
Nres_list = np.linspace(300, 1000, 10, dtype=int)
p_list = np.linspace(0.1, 1.0, 10)
alpha_list = np.linspace(0.1, 1.0, 10)
rho_list = np.linspace(0.1, 1.5, 10)

problems = ["a", "b", "c", "e"]
transient_pts = 40000  # discard transient
results = {}

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
        # Parallel evaluation of NRMSE for all values
        nrmse_list = Parallel(n_jobs=2)(delayed(compute_nrmse)(param_name, val, u_train, y_train) for val in param_list) # 2x faster than not using Parallel 
        # nrmse_list = [compute_nrmse(param_name, val, u_train, y_train) for val in param_list] # took 10 min from a to b
        results[label][param_name] = (param_list, nrmse_list)
        print(f"{param_name} done, {label}")
              
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

# prompt to open a shell session into the server (cip3a0.cip.cs.fau) and land directly in that directory (/proj/ciptmp/qi24jovo/tnd2025)
# ssh qi24jovo@cip3a0.cip.cs.fau.de
# cd /proj/ciptmp/qi24jovo/tnd2025
# conda activate myenv
# source myenv_proj/bin/activate
# python main.py

# prompt to copy files from local directory to server
# scp C:\Users\prizl\Documents\GitHub\TND2025\B1\hyperparameter_optimization.py qi24jovo@cip3a0.cip.cs.fau.de:/proj/ciptmp/qi24jovo/tnd2025/

# prompt to copy plots generated inside the cip-pool server to local directory
# scp -r qi24jovo@cip3a0.cip.cs.fau.de:/proj/ciptmp/qi24jovo/tnd2025/A2_* "C:/Users/prizl/Documents/GitHub/TND2025/B1/plots"


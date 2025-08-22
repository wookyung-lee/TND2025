import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from ESN import * 

# prompt to copy files from local directory to server
# scp C:\Users\prizl\Documents\GitHub\TND2025\B1\main.py qi24jovo@cip3a0.cip.cs.fau.de:/proj/ciptmp/qi24jovo/tnd2025/

# prompt to copy plots generated inside the cip-pool server to local directory
# scp -r qi24jovo@cip3a0.cip.cs.fau.de:/proj/ciptmp/qi24jovo/tnd2025/A2_* "C:/Users/prizl/Documents/GitHub/TND2025/B1/plots"

# Import dictionary
data_folder = os.path.abspath('A2')
sys.path.insert(0, data_folder)
from A2 import all_data

# For parallel computing
def compute_nrmse(param_name, param_value, u_train, y_train):
    kwargs = {'Nres': 300, 'p': 0.75, 'alpha': 0.5, 'rho': 0.85, 'random_state': 111}
    kwargs[param_name] = param_value
    kwargs['random_state'] = 111
    esn = ESN(**kwargs)    

    return esn.train(u_train, y_train)

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
    u_train = u[:N_train]
    y_train = u[:N_train]

    results[label] = {}

    for param_name, param_list in zip(['Nres', 'p', 'alpha', 'rho'], [Nres_list, p_list, alpha_list, rho_list]):
        # Parallel evaluation of NRMSE for all values
        nrmse_list = Parallel(n_jobs=2)(delayed(compute_nrmse)(param_name, val, u_train, y_train) for val in param_list) # 2x faster than not using Parallel 
        # nrmse_list = [compute_nrmse(param_name, val, u_train, y_train) for val in param_list] # took 10 min from a to b
        results[label][param_name] = (param_list, nrmse_list)
        print(f"{param_name} done, {label}")
              
    # # NRMSE for Nres
    # nrmse_list = []
    # for Nres in Nres_list:
    #     esn = ESN(Nres=Nres, p=0.75, alpha=0.5, rho=0.85, random_state=111)
    #     nrmse = esn.train(u_train, y_train)
    #     nrmse_list.append(nrmse)
    # results[label]['Nres'] = (Nres_list, nrmse_list)
    # print(f"Nres done, {label}")

    # # NRMSE for p
    # nrmse_list = []
    # for p in p_list:
    #     esn = ESN(Nres=300, p=p, alpha=0.5, rho=0.85, random_state=111)
    #     nrmse = esn.train(u_train, y_train)
    #     nrmse_list.append(nrmse)
    # results[label]['p'] = (p_list, nrmse_list)
    # print(f"p done, {label}")

    # # NRMSE for alpha
    # nrmse_list = []
    # for alpha in alpha_list:
    #     esn = ESN(Nres=300, p=0.75, alpha=alpha, rho=0.85, random_state=111)
    #     nrmse = esn.train(u_train, y_train)
    #     nrmse_list.append(nrmse)
    # results[label]['alpha'] = (alpha_list, nrmse_list)
    # print(f"alpha done, {label}")

    # # NRMSE for rho
    # nrmse_list = []
    # for rho in rho_list:
    #     esn = ESN(Nres=300, p=0.75, alpha=0.5, rho=rho, random_state=111)
    #     nrmse = esn.train(u_train, y_train)
    #     nrmse_list.append(nrmse)
    # results[label]['rho'] = (rho_list, nrmse_list)
    # print(f"rho done, {label}")

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

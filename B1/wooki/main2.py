import sys
import os
import numpy as np
from ESN2 import *

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

    # NRMSE for Nres
    nrmse_list = []
    for Nres in Nres_list:
        esn = ESN(Nres=Nres, p=0.75, alpha=0.5, rho=0.85, random_state=42)
        nrmse = esn.train(u_train, y_train)
        nrmse_list.append(nrmse)
    results[label]['Nres'] = (Nres_list, nrmse_list)

    # NRMSE for p
    nrmse_list = []
    for p in p_list:
        esn = ESN(Nres=300, p=p, alpha=0.5, rho=0.85, random_state=42)
        nrmse = esn.train(u_train, y_train)
        nrmse_list.append(nrmse)
    results[label]['p'] = (p_list, nrmse_list)

    # NRMSE for alpha
    nrmse_list = []
    for alpha in alpha_list:
        esn = ESN(Nres=300, p=0.75, alpha=alpha, rho=0.85, random_state=42)
        nrmse = esn.train(u_train, y_train)
        nrmse_list.append(nrmse)
    results[label]['alpha'] = (alpha_list, nrmse_list)

    # NRMSE for rho
    nrmse_list = []
    for rho in rho_list:
        esn = ESN(Nres=300, p=0.75, alpha=0.5, rho=rho, random_state=42)
        nrmse = esn.train(u_train, y_train)
        nrmse_list.append(nrmse)
    results[label]['rho'] = (rho_list, nrmse_list)


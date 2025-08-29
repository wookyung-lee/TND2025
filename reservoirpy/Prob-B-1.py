import sys, os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from reservoirpy.nodes import Reservoir, Ridge
from sklearn.metrics import mean_squared_error
from copy import deepcopy
import pickle

# Preparation
seed = 42
np.random.seed(seed)
train_fraction = 0.5
dt = 0.005
Tt = 40000  # transient points to discard

def create_esn(Nres, p, alpha, rho):
    Win = np.random.uniform(-0.5, 0.5, (Nres, 1))
    
    graph = nx.erdos_renyi_graph(Nres, p, directed=True)
    for node in graph.nodes():
        if np.random.random() < p:
            graph.add_edge(node, node)
    Wres = nx.to_numpy_array(graph)
    Wres = np.where(Wres > 0, np.random.uniform(-1, 1, Wres.shape), Wres)
    eigenvalues = np.linalg.eigvals(Wres)
    spectral_radius = np.max(np.abs(eigenvalues))
    Wres = (Wres / spectral_radius) * rho
    
    reservoir = Reservoir(Win=Win, W=Wres, bias=np.zeros(Nres), lr=alpha, input_dim=1, seed=seed)
    readout = Ridge(ridge=1e-6, fit_bias=False)
    esn = reservoir >> readout
    
    return esn

def load_data():
    with open("../timeseries_data.pkl", "rb") as f:
        all_data = pickle.load(f)
        return all_data
all_data = load_data()

# Function to compute NRMSE for a given hyperparameter
def compute_nrmse(param_name, param_value, u_train, y_train):
    kwargs = {'Nres': 850, 'p': 0.7, 'alpha': 0.1, 'rho': 0.85}
    kwargs[param_name] = param_value
    esn = create_esn(**kwargs)
    esn.fit(u_train, y_train, warmup=0)
    esn.nodes[0].reset()
    pred = esn.predict(u_train)
    mse = mean_squared_error(y_train, pred)
    return np.sqrt(mse) / np.std(y_train)

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
    u_train = u[:N_train-1].reshape(-1, 1)
    y_train = u[1:N_train].reshape(-1, 1)

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

'''
Optimal hyperparameters (with min NRMSE) for each regime:
{'a': {'Nres': (922, 0.0009453201023228991), 'p': (0.9, 0.0009716100001276195), 'alpha': (0.7000000000000001, 1.8781744864847318e-06), 'rho': (1.188888888888889, 0.000758900054331064)}, 
'b': {'Nres': (1000, 0.000527449182586811), 'p': (0.5, 0.0005269772331512974), 'alpha': (0.7000000000000001, 2.36607479841743e-06), 'rho': (1.188888888888889, 0.0004067732903537365)}, 
'c': {'Nres': (844, 0.0007346363801683339), 'p': (0.4, 0.0007455191571105316), 'alpha': (0.7000000000000001, 2.1350917129243833e-06), 'rho': (1.0333333333333334, 0.0006178028985463741)}, 
'e': {'Nres': (1000, 0.000268069970590192), 'p': (0.5, 0.0002708632464014993), 'alpha': (0.5, 9.89688134234625e-06), 'rho': (1.188888888888889, 0.00022912772705475505)}}
'''
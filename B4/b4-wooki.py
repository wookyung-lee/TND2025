import torch
import matplotlib.pyplot as plt
import pickle
from torchdiffeq import odeint

# Checking if GPU device is used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

from ESN import *
# Add path for ESN class (use this to run on your device, e.g. laptop)
# sys.path.append(r"C:/Users/prizl/Documents/GitHub/TND2025")
# from ESN import ESN  # Make sure ESN is modified for PyTorch tensors

# Load optimized parameters
with open("optimized_params_rho.pkl", "rb") as f:
    optimized_params_rho = pickle.load(f)

# Placeholder warmup times 
T2_times = {'a': 10000, 'b': 10000, 'c': 10000, 'e': 10000}

# HR neuron model (PyTorch ver.)
def hr_neuron_model(state, I, r):
    x, y, z = state
    dxdt = y + 3*x**2 - x**3 - z + I
    dydt = 1 - 5*x**2 - y
    dzdt = r * (4*(x + 8/5) - z)
    return torch.stack([dxdt, dydt, dzdt])

# ODE solver 
def simulate_hr(initial_state, t, I, r):
    def f(t, state):
        return hr_neuron_model(state, I, r)
    sol = odeint(f, initial_state, t)
    return sol

# Setup 
initial_state = torch.tensor([-1.0, 2.0, 0.5], device=device)
dt = 0.02
t = torch.arange(0, 1500, dt, device=device)
r = 0.003

# Create ESN models 
esn_models = {}
for key, val in optimized_params_rho.items():
    params = val['best_params']
    esn_models[key] = ESN(
        params['Nres'],
        params['p'],
        params['alpha'],
        params['rho']
    )  # already GPU-ready

# Spike interval finder
def compute_isis(time_series, threshold=1.0):
    spikes = torch.where((time_series[:-1] < threshold) & (time_series[1:] >= threshold))[0]
    if len(spikes) < 2:  # need at least 2 spikes to compute ISI
        return torch.tensor([], device=time_series.device)
    isis = torch.diff(spikes).float() * dt
    return isis

# Control parameter range
# I_values = torch.arange(2.5, 3.5 + 0.005, 0.005, device=device)
I_values = torch.arange(2.5, 3.5, 0.5, device=device)
log_isis_results_all = {key: [] for key in esn_models.keys()}

# Main loop
for key, esn in esn_models.items():
    Tt = T2_times[key]
    for I in I_values:
        print(f"Processing ESN {key} with I={I.item():.3f}")
        # HR neuron simulation
        solution = simulate_hr(initial_state, t, I, r)
        x = solution[:,0]

        # normalize
        mean = x.mean()
        std = x.std()
        x_norm = (x - mean) / std

        # train/test split
        training_data = x_norm[Tt:]
        split_idx = len(training_data) // 2
        train_input = training_data[:split_idx-1]
        train_target = training_data[1:split_idx]

        # TRAIN
        esn.train(train_input, train_target, transient_steps=Tt)

        # PREDICT
        warmup_input = train_input[-Tt:]
        n_pred_steps = len(training_data[split_idx-1:])
        predicted = esn.predict(warmup_input, n_pred_steps)
        
        # print(type(predicted), type(std), type(mean), flush=True)
        predicted = torch.tensor(predicted, device=device, dtype=torch.float32)
        
        # denormalize
        predicted_denorm = predicted * std + mean

        # ISIs
        isis = compute_isis(predicted_denorm)
        log_isis = torch.log(isis[isis>0]) if len(isis) > 0 else torch.tensor([float('nan')], device=device)
        log_isis_results_all[key].append(log_isis.cpu())  # move to CPU for plotting

# Plotting
plt.figure(figsize=(12,8))
colors = {'a':'magenta', 'b':'blue', 'c':'green', 'e':'orange'}
for key in esn_models.keys():
    plt.figure(figsize=(12, 8))
    for i, I in enumerate(I_values.cpu()):
        if len(log_isis_results_all[key][i]) > 0:
            plt.scatter([I.item()]*len(log_isis_results_all[key][i]),
                        log_isis_results_all[key][i].numpy(),
                        color=colors[key], s=4, label=key if i==0 else "")
            
    plt.xlabel('Control Parameter I')
    plt.ylabel('log(ISI)')
    plt.title(f'Bifurcation Diagram - ESN variant {key}')
    plt.grid(True, which='minor', linestyle='-', linewidth=0.5)
    plt.minorticks_on()
    plt.tick_params(which='minor', length=5, color='gray')
    plt.savefig(f'Prob-B-4_{key}.png')
    plt.close()

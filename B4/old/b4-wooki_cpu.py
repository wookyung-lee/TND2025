import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.preprocessing import StandardScaler
import pickle
import sys

# Add path for ESN class
sys.path.append(r"C:/Users/prizl/Documents/GitHub/TND2025")
from ESN_old import ESN

# --- Load optimized parameters ---
with open("optimized_params_rho.pkl", "rb") as f:
    optimized_params_rho = pickle.load(f)

# --- Load warmup times ---
# with open("warmup_times.pkl", "rb") as f:
#     warmup_data = pickle.load(f)

# Extract T2 for each key
# T2_times = {key: val['T2'] for key, val in warmup_data.items()} # 200k

# placeholder
T2_times = {'a': 1000, 'b': 1000, 'c': 1000, 'e': 1000}

# --- HR neuron model ---
def hr_neuron_model(state, t, I, r):
    x, y, z = state
    dxdt = y + 3 * x**2 - x**3 - z + I
    dydt = 1 - 5 * x**2 - y
    dzdt = r * (4 * (x + 8/5) - z)
    return [dxdt, dydt, dzdt]

# --- General setup ---
initial_state = [-1.0, 2.0, 0.5]
dt = 0.02
t = np.arange(0, 1500, dt) # len = 75k
r = 0.003

# --- Create ESN models ---
esn_models = {}
for key, val in optimized_params_rho.items():
    params = val['best_params']
    esn_models[key] = ESN(
        params['Nres'],
        params['p'],
        params['alpha'],
        params['rho']
    )

# --- Spike interval finder ---
def compute_isis(time_series, threshold=1.0):
    spikes = np.where((time_series[:-1] < threshold) & (time_series[1:] >= threshold))[0]
    isis = np.diff(spikes) * dt
    return isis

# --- Control parameter range ---
I_values = np.arange(2.5, 3.5 + 0.005, 0.005)

# --- Store log(ISI) results ---
log_isis_results_all = {key: [] for key in esn_models.keys()}

# --- Loop over ESNs and I values ---
for key, esn in esn_models.items():
    Tt = T2_times[key]  # warmup
    for I in I_values:
        # HR neuron simulation
        solution = odeint(hr_neuron_model, initial_state, t, args=(I, r))
        x = solution[:,0]
        
        # normalize
        scaler = StandardScaler()
        x_norm = scaler.fit_transform(x.reshape(-1,1)).flatten()
        
        # train/test split
        training_data = x_norm[Tt:]
        split_idx = len(training_data) // 2
        train_input = training_data[:split_idx-1]
        train_target = training_data[1:split_idx]

        # train ESN
        # transient_steps = min(Tt, len(train_input)-1)
        esn.train(train_input, train_target, transient_steps=Tt)
        
        # predict
        warmup_input = train_input[-Tt:]
        n_pred_steps = len(training_data[split_idx-1:])
        predicted = esn.predict(warmup_input, n_pred_steps)
        predicted_denorm = scaler.inverse_transform(predicted.reshape(-1,1)).flatten()
        
        # ISIs
        isis = compute_isis(predicted_denorm)
        log_isis = np.log(isis[isis>0]) if len(isis) > 0 else [np.nan]
        log_isis_results_all[key].append(log_isis)

# --- Plotting ---
plt.figure(figsize=(12, 8))
colors = {'a': 'magenta', 'b': 'blue', 'c': 'green', 'e': 'orange'}

for key in esn_models.keys():
    for i, I in enumerate(I_values):
        if len(log_isis_results_all[key][i]) > 0:
            plt.scatter(
                [I] * len(log_isis_results_all[key][i]),
                log_isis_results_all[key][i],
                color=colors[key], s=4, label=key if i == 0 else ""
            )

plt.xlabel('Control Parameter I')
plt.ylabel('log(ISI)')
plt.title('Prob-B-4: Bifurcation Diagram (ESN variants)')
plt.grid(True, which='minor', linestyle='-', linewidth=0.5)
plt.minorticks_on()
plt.tick_params(which='minor', length=5, color='gray')
plt.legend()
plt.savefig('Prob-B-4.png')
plt.show()

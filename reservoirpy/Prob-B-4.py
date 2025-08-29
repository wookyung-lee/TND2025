import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from reservoirpy.nodes import Reservoir, Ridge
import networkx as nx


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

def predict_autonomous(esn, warmup, n_steps):
    esn.nodes[0].reset()
    warumup_preds = esn.run(warmup)
    
    predictions = []
    ut = warumup_preds[-1]
    
    for _ in range(n_steps):
        pred = esn(ut)
        predictions.append(pred)
        ut = pred
    
    return np.array(predictions)

# HR neuron system
def hr_neuron_model(state, t, I, r):
    x, y, z = state
    dxdt = y + 3 * x**2 - x**3 - z + I
    dydt = 1 - 5 * x**2 - y
    dzdt = r * (4 * (x + 8/5) - z)
    return [dxdt, dydt, dzdt]


# General setup
initial_state = [-1.0, 2.0, 0.5]
dt = 0.02
t = np.arange(0, 1500, dt)
r = 0.003
Tt = 40000
optimized_params = {'N_res': 875, 'p': 0.8, 'alpha': 0.5, 'rho': 0.9}
N_warmup2 = 200000
seed = 42

# Create ESN
esn_optimized = create_esn(
    optimized_params['N_res'],
    optimized_params['p'],
    optimized_params['alpha'],
    optimized_params['rho']
)

# Spike interval finder
def compute_isis(time_series, threshold=1.0):
    spikes = np.where((time_series[:-1] < threshold) & (time_series[1:] >= threshold))[0]
    isis = np.diff(spikes) * dt
    return isis

# Loop across control parameter I
I_values = np.arange(2.5, 3.5, 0.005)
log_isis_results = []

for I in I_values:
    solution = odeint(hr_neuron_model, initial_state, t, args=(I, r))
    x = solution[:, 0]

    scaler = StandardScaler()
    x_normalized = scaler.fit_transform(x.reshape(-1, 1)).flatten()

    usable_data = x_normalized[Tt:]
    split_idx = len(usable_data) // 2
    train_data = usable_data[:split_idx].reshape(-1, 1)
    warmup = usable_data[:N_warmup2].reshape(-1, 1)

    esn_optimized.fit(train_data[:-1], train_data[1:], warmup=0)
    predicted = predict_autonomous(esn_optimized, warmup, len(usable_data[N_warmup2:-1]))
    predicted_denormalized = scaler.inverse_transform(predicted.reshape(-1, 1)).flatten()

    isis = compute_isis(predicted_denormalized)
    log_isis = np.log(isis) if len(isis) > 0 else [np.nan]
    log_isis_results.append(log_isis)

# Draw diagram
plt.figure(figsize=(12, 8))
for i, I in enumerate(I_values):
    if len(log_isis_results[i]) > 0:
        plt.scatter([I] * len(log_isis_results[i]), log_isis_results[i], color='#ff00ff', s=4)

plt.xlabel('Control Parameter I')
plt.ylabel('log(ISI)')
plt.title('Prob-B-4: Bifurcation Diagram')
plt.grid(True, which='minor', linestyle='-', linewidth=0.5)
plt.minorticks_on()
plt.tick_params(which='minor', length=5, color='gray')
plt.savefig('Prob-B-4.png')
plt.show()


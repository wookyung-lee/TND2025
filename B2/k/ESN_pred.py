import numpy as np
import matplotlib.pyplot as plt
import torch
from ESN import ESN
from A2 import all_data
import pickle

with open("optimized_params.pkl", "rb") as f:
    optimized_params = pickle.load(f)

transient_pts = 40000  # Washed-out transient points to discard
dt = 0.005  # Time step used in integration
warmup_points_list = np.linspace(40001, 130000, 20, dtype=int)  # Warm-up points to test (adjust upper bound <= data length)

def compute_prediction_nrmse(esn, u, y, Nwarmup):
    """
    Train ESN on input u, y then compute prediction NRMSE with Nwarmup warm-up points.
    """
    # Train ESN
    esn.train(u, y)

    T = len(u)
    r = torch.zeros(esn.Nres, device=esn.Win.device, dtype=torch.float32)

    predicted = []
    # Warm-up phase with true input
    for t in range(min(Nwarmup, T)):
        u_t = torch.tensor(u[t], device=esn.Win.device, dtype=torch.float32)
        r = (1 - esn.alpha) * r + esn.alpha * torch.tanh(esn.Wres @ r + esn.Win * u_t)

    # Prediction phase autonomous
    with torch.no_grad():
        for t in range(Nwarmup, T):
            x_out = esn.Wout @ r
            predicted.append(x_out.cpu().item())
            r = (1 - esn.alpha) * r + esn.alpha * torch.tanh(esn.Wres @ r + esn.Win * x_out)

    predicted = np.array(predicted)
    y_true = y[min(Nwarmup, len(y)):]

    error = np.sqrt(np.mean((predicted - y_true) ** 2)) / np.std(y_true)
    return error

# Run Problem B2 for each regime
results_prediction = {}

for label in ['a', 'b', 'c', 'e']:
    data = all_data[label]['x']
    u = data[transient_pts:]
    # Normalize input with zero mean and unit variance
    u = (u - np.mean(u)) / np.std(u)

    N_train = len(u) // 2
    u_train = u[:N_train]
    y_train = u[:N_train]

    params = optimized_params[label]

    esn = ESN(Nres=params['Nres'], p=params['p'], alpha=params['alpha'], rho=params['rho'], lambda_reg=1e-6, random_state=42)

    # Ensure warmup points do not exceed training data size
    warmup_points = warmup_points_list[warmup_points_list < len(u_train)]

    nrmse_pred_list = []
    for Nwarmup in warmup_points:
        nrmse_pred = compute_prediction_nrmse(esn, u_train, y_train, Nwarmup)
        nrmse_pred_list.append(nrmse_pred)

    results_prediction[label] = (warmup_points * dt, nrmse_pred_list)  # Convert warm-up points to time

    # Plot NRMSE prediction vs warm-up time
    plt.figure()
    plt.plot(warmup_points * dt, nrmse_pred_list, marker='o')
    plt.xlabel('Warm-up Time (s)')
    plt.ylabel('NRMSE in Prediction')
    plt.title(f'NRMSE vs Warm-up Time - Regime {label}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

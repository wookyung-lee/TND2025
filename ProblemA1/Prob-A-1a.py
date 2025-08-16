import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import math




# Simulation batch size hyper parameter
batch_size = 200

# Run on GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32





def hr(X:torch.Tensor, I:torch.Tensor, r:float=0.0) -> torch.Tensor:
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    x_squared = x * x
    dx = y + 3.0 * x_squared - x * x_squared - z + I
    dy = 1.0 - 5.0 * x_squared - y
    dz = r * (4.0 * (x + 1.6) - z)
    return torch.stack([dx, dy, dz], dim=1)


def integrate(
    f:callable,
    X0:torch.Tensor,
    I:torch.Tensor,
    r:float,
    T:float,
    dt:float, 
    t0:float=0.0, 
    T_t:float=0.0
    ) -> tuple:
    # Device Conversion
    X0 = X0.to(device=device, dtype=dtype)
    I = I.to(device=device, dtype=dtype)
    batch_size = X0.shape[0]
    
    # Number of steps
    n_steps = int(math.ceil((T - t0) / dt)) if T > t0 else 0
    n_transient = int(math.ceil((T_t - t0) / dt)) if T_t > t0 else 0
    n_save = n_steps + 1 - n_transient
    if n_save <= 0:
        raise RuntimeError("More Transient steps than overall steps")
    
    # Allocate memory
    xs = torch.empty((n_save, batch_size), device=device, dtype=dtype)
    ts = torch.linspace(t0 + n_transient * dt, T, n_save, device=device, dtype=dtype)
    
    # Initial parameters
    t = t0
    x = X0.clone()
    idx = 0
    if n_transient == 0:
        xs[idx] = x[:, 0]
        idx = 1
    
    with torch.no_grad():
        for i in range(1, n_steps + 1):
            # Determine step size
            h = dt
            if i == n_steps:
                h = float(T) - float(t)
                if h <= 0.0:
                    break
            h_half = h / 2.0
            
            # RK 4
            k1 = f(x, I, r)
            k2 = f(x + h_half * k1, I, r)
            k3 = f(x + h_half * k2, I, r)
            k4 = f(x + h * k3, I, r)
            
            t += h
            x += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            
            # Store in memory
            if i >= n_transient:
                xs[idx] = x[:, 0]
                idx += 1
    
    if idx != n_save:
        xs = xs[:idx]
        ts = ts[:idx]
    
    return ts, xs


def get_logISIs(ts:np.ndarray, xs:np.ndarray, x_th:float) -> list:
    # Determine threshold crossing from below
    below_threshold = xs[:-1] < x_th
    above_threshold = xs[1:] >= x_th
    crossings = below_threshold & above_threshold
    
    batch_size = xs.shape[1]
    logISIs = []
    
    for i in range(batch_size):
        crossing_mask = crossings[:, i]
        if not crossing_mask.any():
            logISIs.append(np.array([]))
            continue
        
        spike_indices = np.nonzero(crossing_mask)[0] + 1
        spike_times = ts[spike_indices]
        
        if len(spike_times) < 2:
            logISIs.append(np.array([]))
            continue
        
        # Calculate log of inter-spike intervals
        intervals = np.diff(spike_times)
        valid_intervals = intervals[intervals > 0]
        if len(valid_intervals) == 0:
            logISIs.append(np.array([]))
        else:
            logISI = np.log(valid_intervals)
            logISIs.append(logISI)
    
    return logISIs


def plot(Is:np.ndarray, logISIs:list):
    x_points, y_points = [], []
    
    if len(Is) != len(logISIs):
        raise RuntimeError("Is and logISIs have different lengths")
    
    for I, logISI in zip(Is, logISIs):
        # No Spiking
        if logISI is None or logISI.size == 0:
            continue
        
        # Invalid interval values
        finite_values = np.isfinite(logISI)
        if not np.any(finite_values):
            continue
        
        x_points.append(np.full(np.count_nonzero(finite_values), I))
        y_points.append(logISI[finite_values])
    
    if len(x_points) == 0:
        raise RuntimeError("No ISIs found")
    
    x_points = np.concatenate(x_points)
    y_points = np.concatenate(y_points)
    
    plt.figure(figsize=(10, 7))
    plt.scatter(x_points, y_points, s=1, alpha=0.25)
    plt.title("Bifurcation Diagram I vs log(ISI)")
    plt.xlabel("I [mV]")
    plt.ylabel("log(ISI)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Prob-A-1a.png")
    plt.show()


def main(batch_size=50):
    print(f"Device: {device}")
    
    # Parameter I
    I_min = 2.5
    I_max = 3.5
    dI = 0.005
    Is = []
    I = I_min
    while I < I_max - 1e-10:
        Is.append(I)
        I += dI
    Is.append(I_max)
    Is = torch.tensor(Is, device=device, dtype=dtype)
    N = len(Is)
    
    # Initial state
    x0 = -1.0
    y0 = 2.0
    z0 = 0.5
    X0 = torch.tensor([x0, y0, z0], device=device, dtype=dtype).unsqueeze(0)
    x_th = 1.0
    
    # Simulation time
    T = 1500.0
    dt = 0.005
    T_t = 200.0
    
    # Parameter r
    r = 0.003
    
    logISIs = []
    
    # Batch processing
    for start in tqdm(range(0, N, batch_size)):
        # Prepare batch
        end = min(start + batch_size, N)
        Is_batch = Is[start:end]
        X0_batch = X0.repeat(len(Is_batch), 1)
        
        # Integrate Hindmarsh-Rose neuron model for batch
        ts_batch, xs_batch = integrate(hr, X0_batch, Is_batch, r, T, dt, T_t=T_t)
        ts_np = ts_batch.cpu().numpy()
        xs_np = xs_batch.cpu().numpy()
        
        # Calculate log inter-spike intervals for x crossing threshold from below
        logISIs_batch = get_logISIs(ts_np, xs_np, x_th)
        logISIs.extend(logISIs_batch)
    
    # Plot results and save as image
    Is_cpu = Is.cpu().numpy()
    plot(Is_cpu, logISIs)


if __name__ == "__main__":
    main(batch_size)
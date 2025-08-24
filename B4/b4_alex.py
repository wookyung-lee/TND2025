from tqdm import tqdm
import torch
import networkx as nx
import math

# ------------------------------
# Device
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

# ------------------------------
# Hindmarsh-Rose neuron model (same as in A1a)
# ------------------------------
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
    X0 = X0.to(device=device, dtype=dtype)
    I = I.to(device=device, dtype=dtype)
    n_steps = int(math.ceil((T - t0) / dt))
    n_transient = int(math.ceil((T_t - t0) / dt))
    n_save = n_steps + 1 - n_transient
    xs = torch.empty((n_save, X0.shape[0]), device=device, dtype=dtype)
    ts = torch.linspace(t0 + n_transient * dt, T, n_save, device=device, dtype=dtype)
    t = t0
    x = X0.clone()
    idx = 0
    if n_transient == 0:
        xs[idx] = x[:, 0]
        idx = 1
    with torch.no_grad():
        for i in range(1, n_steps + 1):
            h = dt
            if i == n_steps:
                h = float(T) - float(t)
                if h <= 0.0: break
            h_half = h / 2.0
            k1 = f(x, I, r)
            k2 = f(x + h_half * k1, I, r)
            k3 = f(x + h_half * k2, I, r)
            k4 = f(x + h * k3, I, r)
            t += h
            x += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            if i >= n_transient:
                xs[idx] = x[:, 0]
                idx += 1
    return ts, xs

# ------------------------------
# ESN class for B4
# ------------------------------
class ESN:
    def __init__(self, Nres=300, p=0.8, alpha=0.6, rho=0.9, lambda_reg=1e-6, random_state=42):
        self.Nres = Nres
        self.p = p
        self.alpha = alpha
        self.rho = rho
        self.lambda_reg = lambda_reg
        np.random.seed(random_state)
        # Input weights
        self.Win = torch.tensor(np.random.uniform(-0.5, 0.5, size=(self.Nres,)),
                                dtype=torch.float32, device=device)
        # Reservoir weights
        G = nx.erdos_renyi_graph(self.Nres, self.p, directed=True, seed=random_state)
        for node in G.nodes():
            if np.random.random() < self.p:
                G.add_edge(node, node)
        Wres = nx.to_numpy_array(G)
        Wres[Wres != 0] = np.random.uniform(-1, 1, size=np.count_nonzero(Wres))
        eigvals = np.linalg.eigvals(Wres)
        max_eig = max(abs(eigvals))
        if max_eig > 0:
            Wres *= self.rho / max_eig
        self.Wres = torch.tensor(Wres, dtype=torch.float32, device=device)
        self.Wout = None

    def _reservoir_states(self, u):
        T = len(u)
        u = torch.tensor(u, dtype=torch.float32, device=device)
        X_res = torch.zeros((self.Nres, T), device=device)
        x = torch.zeros(self.Nres, device=device)
        for t in range(T):
            x = (1 - self.alpha) * x + self.alpha * torch.tanh(self.Wres @ x + self.Win * u[t])
            X_res[:, t] = x
        return X_res

    def train(self, u_train, y_train):
        X_res = self._reservoir_states(u_train)
        y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
        I = torch.eye(self.Nres, device=device)
        self.Wout = torch.linalg.solve(X_res @ X_res.T + self.lambda_reg * I, X_res @ y_train)

    def predict(self, u_init, steps, warmup=200000):
        u = torch.tensor([u_init], dtype=torch.float32, device=device)
        x = torch.zeros(self.Nres, device=device)
        preds = []
        for t in range(steps + warmup):
            x = (1 - self.alpha) * x + self.alpha * torch.tanh(self.Wres @ x + self.Win * u[-1])
            y = (x @ self.Wout).unsqueeze(0)
            u = torch.cat([u, y])
            if t >= warmup:
                preds.append(y.item())
        return np.array(preds)

# ------------------------------
# Compute ISIs
# ------------------------------
def compute_isis(x, threshold=1.0):
    spikes = np.where((x[:-1] < threshold) & (x[1:] >= threshold))[0]
    if len(spikes) < 2:
        return []
    times = spikes
    isis = np.diff(times)
    return np.log(isis[isis > 0])

# ------------------------------
# Main
# ------------------------------
def main():
    print(f"Device: {device}")
    I_min, I_max, dI = 2.5, 3.5, 0.005
    I_values = np.arange(I_min, I_max + 1e-10, dI)
    r = 0.003
    T, dt, T_t = 1500.0, 0.005, 200.0
    X0 = torch.tensor([[-1.0, 2.0, 0.5]], device=device, dtype=dtype)

    all_I, all_logISI = [], []

    for I in tqdm(I_values):
        ts, xs = integrate(hr, X0, torch.tensor([I], device=device, dtype=dtype), r, T, dt, T_t=T_t)
        x_data = xs.cpu().numpy().flatten()
        split = len(x_data) // 2
        u_train, y_train = x_data[:split], x_data[1:split+1]

        esn = ESN()  # placeholder hyperparams are inside the class
        esn.train(u_train, y_train)

        u_init = x_data[split]
        steps = len(x_data) - split
        x_pred = esn.predict(u_init, steps=steps, warmup=200000)  # placeholder warmup

        log_isis = compute_isis(x_pred)
        if len(log_isis) > 0:
            all_I.extend([I] * len(log_isis))
            all_logISI.extend(log_isis)

    # Save results for later plotting
    np.savez("b4_results.npz", I=np.array(all_I), logISI=np.array(all_logISI))
    print("Saved results to b4_results.npz")

    # Plot immediately (optional)
    plt.figure(figsize=(10, 6))
    plt.scatter(all_I, all_logISI, s=1, c="red", alpha=0.5)
    plt.xlabel("Control parameter I")
    plt.ylabel("log(ISI)")
    plt.title("Problem B4: ESN-based Bifurcation Diagram")
    plt.grid(True)
    plt.savefig("Prob-B-4.png", dpi=600, bbox_inches='tight', pad_inches=0.01)
    plt.show()


if __name__ == "__main__":
    main()

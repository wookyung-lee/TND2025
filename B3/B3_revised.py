# ============================
# B3_revised.py  (single file)
# ============================

# ---- SAFE HEADER: prevent OpenMP/MKL crashes & pickle issues ----
import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")



# ---- Imports (light; no heavy work here) ----
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch

torch.set_num_threads(1)
device = torch.device("cpu")  # keep CPU to avoid extra native libs

# -------------------------------------------------
# If you already have A2's all_data, you can import:
#   from A2 import all_data
# and skip the "Generate A2 data" block below.
# -------------------------------------------------

# ---- Hindmarsh–Rose (HR) utilities ----
def hr_model(state, I, r):
    x, y, z = state
    dx = y + 3*x**2 - x**3 - z + I
    dy = 1 - 5*x**2 - y
    dz = r * (4*(x + 8/5) - z)
    return np.array([dx, dy, dz], dtype=float)

def rk4_step(func, state, dt, I, r):
    k1 = func(state, I, r)
    k2 = func(state + 0.5*dt*k1, I, r)
    k3 = func(state + 0.5*dt*k2, I, r)
    k4 = func(state + dt*k3, I, r)
    return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def integrate_hr(I, r, T=1500.0, dt=0.005, init_state=(-1.0, 2.0, 0.5)):
    n_steps = int(T/dt)
    state = np.array(init_state, dtype=float)
    x_vals = np.zeros(n_steps, dtype=float)
    t_vals = np.linspace(0.0, T, n_steps, endpoint=False)
    for i in range(n_steps):
        x_vals[i] = state[0]
        state = rk4_step(hr_model, state, dt, I, r)
    return t_vals, x_vals

# ---- Generate A2 data (self-contained) ----
cases = {
    "a": (3.5, 0.003),
    "b": (3.34, 0.003),
    "c": (1.67, 0.003),
    # "d": (3.2, 0.003),  # not needed for B
    "e": (3.29, 0.003),
}
all_data = {}
for label, (I, r) in cases.items():
    t, x = integrate_hr(I, r)
    all_data[label] = {"time": t, "x": x}

# ---- ESN (Torch, no networkx, stable & minimal) ----
class ESN:
    def __init__(self, Nres=300, p=0.02, alpha=0.5, rho=0.9,
                 lambda_reg=1e-6, random_state=42):
        self.Nres = int(Nres)
        self.p = float(p)
        self.alpha = float(alpha)
        self.rho = float(rho)
        self.lambda_reg = float(lambda_reg)
        self.random_state = random_state
        self.Wres = None
        self.Win = None
        self.Wout = None
        self._init_weights()

    def _init_weights(self):
        rng = np.random.default_rng(self.random_state)

        # Input weights: Nres x 1 (scalar input)
        self.Win = torch.tensor(
            rng.uniform(-0.5, 0.5, size=(self.Nres,)),
            dtype=torch.float32, device=device
        )

        # Sparse Erdos–Renyi mask
        mask = rng.random((self.Nres, self.Nres)) < self.p
        W = np.zeros((self.Nres, self.Nres), dtype=np.float32)
        W[mask] = rng.uniform(-1.0, 1.0, size=mask.sum()).astype(np.float32)

        # Spectral radius scaling
        if self.Nres <= 1200:  # eig is OK at this size
            eigvals = np.linalg.eigvals(W)
            max_eig = np.max(np.abs(eigvals)) if eigvals.size else 0.0
            if max_eig > 0:
                W *= (self.rho / max_eig)
        else:
            # Fallback: conservative scaling by row-norm
            row_norm = max(np.linalg.norm(W, ord=2), 1e-6)
            W *= (self.rho / row_norm)

        self.Wres = torch.tensor(W, dtype=torch.float32, device=device)

    def _reservoir_states(self, input_u, washout=0):
        """
        Returns (X, y) where:
          X: (Nres, T-washout) reservoir states
          y: (T-washout,) target signal aligned with X columns
        """
        u = torch.tensor(input_u, dtype=torch.float32, device=device)
        T = u.shape[0]
        x = torch.zeros(self.Nres, dtype=torch.float32, device=device)
        X = torch.zeros(self.Nres, T, dtype=torch.float32, device=device)
        for t in range(T):
            x = (1.0 - self.alpha) * x + self.alpha * torch.tanh(self.Wres @ x + self.Win * u[t])
            X[:, t] = x
        if washout > 0:
            X = X[:, washout:]
            u = u[washout:]
        return X, u

    def train(self, u_train, washout=500):
        """
        Ridge regression readout fitting with washout.
        Target is the same as the (normalized) input (1D prediction).
        """
        X, y = self._reservoir_states(u_train, washout=washout)
        I = torch.eye(self.Nres, dtype=torch.float32, device=device)
        self.Wout = torch.linalg.solve(X @ X.T + self.lambda_reg * I, X @ y)
        # Optional: compute NRMSE on the training slice
        y_pred = (X.T @ self.Wout)
        nrmse = torch.sqrt(torch.mean((y_pred - y)**2)) / (torch.std(y) + 1e-12)
        return float(nrmse)

    @torch.no_grad()
    def predict_with_warmup(self, u_full, start_idx, warmup_steps, end_idx):
        """
        Teacher-forcing warm-up on u_full[start_idx : start_idx+warmup_steps],
        then closed-loop until end_idx (exclusive).
        Returns: xr (np.array), start_aut_idx (int)
        """
        u = torch.tensor(u_full, dtype=torch.float32, device=device)
        warmup_steps = int(max(0, warmup_steps))
        start_idx = int(start_idx)
        end_idx = int(end_idx)
        if start_idx + warmup_steps >= end_idx:
            warmup_steps = max(0, end_idx - start_idx - 1)

        x = torch.zeros(self.Nres, dtype=torch.float32, device=device)

        # Teacher forcing for warm-up
        for n in range(start_idx, start_idx + warmup_steps):
            x = (1.0 - self.alpha) * x + self.alpha * torch.tanh(self.Wres @ x + self.Win * u[n])

        # Autonomous rollout
        xr = []
        for n in range(start_idx + warmup_steps, end_idx):
            y_hat = torch.dot(self.Wout, x)  # scalar
            xr.append(float(y_hat))
            x = (1.0 - self.alpha) * x + self.alpha * torch.tanh(self.Wres @ x + self.Win * y_hat)

        return np.array(xr, dtype=float), (start_idx + warmup_steps)

# ---- Configs ----
dt = 0.005
t_wash_hr = 200.0
i_wash_hr = int(t_wash_hr / dt)  # 40_000
T_end = 1500.0
warmups_sec = {"a": (50.0, 1000.0), "b": (50.0, 1000.0), "c": (50.0, 1000.0), "e": (50.0, 1000.0)}
outdir = "B3_figs"
os.makedirs(outdir, exist_ok=True)

# ---- Load optimized params (prefer JSON; fallback to PKL; fallback to defaults) ----
DEFAULT_OPT = {
    "a": {"Nres": 600, "p": 0.02, "alpha": 0.30, "rho": 0.95},
    "b": {"Nres": 800, "p": 0.02, "alpha": 0.25, "rho": 0.98},
    "c": {"Nres": 1000, "p": 0.02, "alpha": 0.20, "rho": 0.97},
    "e": {"Nres": 1000, "p": 0.02, "alpha": 0.20, "rho": 0.99},
}

def load_optimized_params():
    for candidate in ("optimized_params.json", "../optimized_params.json"):
        if os.path.exists(candidate):
            with open(candidate) as jf:
                return json.load(jf)
    for candidate in ("optimized_params.pkl", "../optimized_params.pkl"):
        if os.path.exists(candidate):
            with open(candidate, "rb") as f:
                obj = pickle.load(f)
            # cast to plain types
            return {
                lbl: {"Nres": int(v["Nres"]), "p": float(v["p"]),
                      "alpha": float(v["alpha"]), "rho": float(v["rho"])}
                for lbl, v in obj.items()
            }
    return DEFAULT_OPT

opt = load_optimized_params()

# ---- Training helper (consistent normalization & washout) ----
def train_readout_for_regime(x_full, params):
    """
    Normalizes after 200s like your training code, trains readout on first half.
    Returns trained ESN and (mu, sd) for de-normalization.
    """
    esn = ESN(
        Nres=int(params["Nres"]),
        p=float(params["p"]),
        alpha=float(params["alpha"]),
        rho=float(params["rho"]),
        lambda_reg=1e-6,
        random_state=42,
    )

    # Match your training normalization: after 200s segment
    u = x_full[i_wash_hr:]
    mu, sd = float(np.mean(u)), float(np.std(u) + 1e-12)
    u_norm = (u - mu) / sd

    N_train = len(u_norm) // 2
    esn.train(u_norm[:N_train], washout=500)
    return esn, mu, sd

# ---- Plot helper ----
def plot_overlay(label, t, x, esn, mu, sd, T_warm, linestyle, tag):
    start_idx = i_wash_hr
    end_idx = len(x)
    # normalize entire x with same stats (OK even for t<200s; we only use >=200s)
    u_full_norm = (x - mu) / sd

    warm_steps = int(T_warm / dt)
    xr_norm, start_aut = esn.predict_with_warmup(u_full_norm, start_idx, warm_steps, end_idx)
    t_pred = t[start_aut:end_idx]
    xr = xr_norm * sd + mu

    plt.figure(figsize=(10, 4))
    plt.plot(t, x, lw=0.8, label="Ground truth x(t)", color="blue")
    plt.plot(t_pred, xr, lw=0.8, label="ESN output x_r(t)", color="red")
    plt.axvline(x=t_wash_hr + T_warm, color="k", linestyle=linestyle, lw=1.1,
                label=f"Warm-up {T_warm:.0f}s")
    plt.xlim([200, 1500])
    plt.xlabel("Time t")
    plt.ylabel("Membrane potential x(t)")
    plt.title(f"B3 – Regime {label} ({tag})")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"B3_{label}_{int(T_warm)}s.png"), dpi=300)
    plt.close()

# ---- Run B3 for regimes a, b, c, e ----
for label in ["a", "b", "c", "e"]:
    t = all_data[label]["time"]
    x = all_data[label]["x"]

    params = opt.get(label, DEFAULT_OPT[label])
    esn, mu, sd = train_readout_for_regime(x, params)
    T1, T2 = warmups_sec[label]
    plot_overlay(label, t, x, esn, mu, sd, T1, "-",  f"T1={T1:.0f}s")
    plot_overlay(label, t, x, esn, mu, sd, T2, "--", f"T2={T2:.0f}s")

print(f"Done. Figures saved to: {os.path.abspath(outdir)}")

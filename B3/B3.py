# =========================


import numpy as np
import matplotlib.pyplot as plt
import os

import matplotlib.pyplot as plt

# Hindmarsh-Rose model equations
def hr_model(state, I, r):
    x, y, z = state
    dx = y + 3 * x ** 2 - x ** 3 - z + I
    dy = 1 - 5 * x ** 2 - y
    dz = r * (4 * (x + 8 / 5) - z)
    return np.array([dx, dy, dz])


# Runge–Kutta 4th order integrator
def rk4_step(func, state, dt, I, r):
    k1 = func(state, I, r)
    k2 = func(state + 0.5 * dt * k1, I, r)
    k3 = func(state + 0.5 * dt * k2, I, r)
    k4 = func(state + dt * k3, I, r)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# Integrate HR model
def integrate_hr(I, r, T=1500, dt=0.005, init_state=(-1.0, 2.0, 0.5)):
    n_steps = int(T / dt)
    state = np.array(init_state, dtype=float)
    x_vals = np.zeros(n_steps)
    t_vals = np.linspace(0, T, n_steps)
    for i in range(n_steps):
        x_vals[i] = state[0]
        state = rk4_step(hr_model, state, dt, I, r)
    return t_vals, x_vals


# cases = {
#     "2(a) Periodic spikes (I=3.5, r=0.003)": (3.5, 0.003),
#     "2(b) Chaotic spikes (I=3.34, r=0.003)": (3.34, 0.003),
#     "2(c) Periodic bursts (3 spikes/burst, I=1.67, r=0.003)": (1.67, 0.003),
#     "2(d) Periodic bursts (9 spikes/burst, I=3.2, r=0.003)": (3.2, 0.003),
#     "2(e) Chaotic bursts (I=3.29, r=0.003)": (3.29, 0.003),
# }

cases = {
    "a": (3.5, 0.003),
    "b": (3.34, 0.003),
    "c": (1.67, 0.003),
    "d": (3.2, 0.003),
    "e": (3.29, 0.003),
}

# Used in problem B
all_data = {}

# Plots & Collect data
for label, (I, r) in cases.items():
    t, x = integrate_hr(I, r)

    all_data[label] = {'time': t, 'x': x}

    # plt.figure(figsize=(10, 4))
    # plt.plot(t, x, lw=0.5, color="blue")
    # plt.title(f"Problem A.{label}")
    # plt.xlabel("Time t (ms)")
    # plt.ylabel("Membrane potential x(t)")
    # plt.xlim([200, 1500])
    # plt.tight_layout()

    # Save plots
    # script_dir = os.path.dirname(os.path.abspath(__file__))  # folder of the script
    # plt.savefig(os.path.join(script_dir, f"Figure_A2_{label}.png"), dpi=300)

    # plt.show()

# === B3: ESN overlays ====
# =========================
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

# -------------------------
# Utility: simple ESN class
# -------------------------
class ESN:
    def __init__(self, n_res=800, p=0.02, alpha=0.3, rho=0.95, in_scale=1.0, seed=7, ridge=1e-6):
        rng = np.random.default_rng(seed)
        self.n_res = n_res
        self.alpha = alpha
        self.rho = rho
        self.in_scale = in_scale
        self.ridge = ridge

        # sparse reservoir
        W = sparse.random(n_res, n_res, density=p, random_state=rng, data_rvs=rng.standard_normal).toarray()
        # scale spectral radius
        eigvals = np.linalg.eigvals(W)
        W *= (rho / np.max(np.abs(eigvals)))
        self.W = W

        # input weights (1D input x plus bias)
        self.Win = rng.standard_normal((n_res, 2)) * in_scale

        # readout (to be trained)
        self.Wout = None
        self.state = np.zeros(n_res)

    def reset(self):
        self.state.fill(0.0)

    def _update(self, u):
        # u is scalar; use tanh reservoir with leak
        u_vec = np.array([u, 1.0])  # [input; bias]
        pre = self.W @ self.state + self.Win @ u_vec
        r = np.tanh(pre)
        self.state = (1 - self.alpha) * self.state + self.alpha * r
        return self.state

    def collect_states(self, u_seq, washout_steps=0):
        """Teacher forcing over u_seq (numpy array), returns X (augmented states), Y (targets)."""
        self.reset()
        X = []
        Y = []
        for t, u in enumerate(u_seq):
            r = self._update(u)
            if t >= washout_steps:
                X.append(np.hstack([r, 1.0]))  # augment with bias
                Y.append(u_seq[t])            # predict x(t)
        return np.asarray(X), np.asarray(Y)[:, None]

    def fit_readout(self, u_seq, washout_steps=0):
        X, Y = self.collect_states(u_seq, washout_steps=washout_steps)
        # ridge regression: Wout = Y X^T (X X^T + λI)^-1
        XXt = X.T @ X
        regI = self.ridge * np.eye(XXt.shape[0])
        self.Wout = (np.linalg.solve(XXt + regI, X.T @ Y)).T  # shape (1, n_res+1)
        return X, Y

    def predict_with_warmup(self, u_true, warmup_steps, start_idx, end_idx):
        """
        Teacher-force for warmup_steps starting at start_idx; then close the loop until end_idx (exclusive).
        Returns xr (length end_idx-start_idx), and the time index at which autonomous prediction starts.
        """
        self.reset()
        # teacher forcing warmup
        for n in range(start_idx, start_idx + warmup_steps):
            self._update(u_true[n])
        # closed loop rollout
        xr = []
        # start closed loop from the last true value (common choice) or model's own next step; we use model output loop
        y_prev = u_true[start_idx + warmup_steps]  # seed with ground truth once
        for n in range(start_idx + warmup_steps, end_idx):
            # compute output from current reservoir state
            aug = np.hstack([self.state, 1.0])[None, :]               # (1, n_res+1)
            y = float(self.Wout @ aug.T)                               # scalar
            xr.append(y)
            # feed back model output
            self._update(y)
        return np.array(xr), start_idx + warmup_steps

# -------------------------------------------------------
# B3 config: hyperparams (reuse your best from B2 here!)
# You can customize per regime if you tuned separately.
# -------------------------------------------------------
b2_hparams = {
    "a": dict(n_res=800, p=0.02, alpha=0.3, rho=0.95, in_scale=1.0, ridge=1e-6, seed=7),
    "b": dict(n_res=1000, p=0.02, alpha=0.25, rho=0.98, in_scale=1.0, ridge=1e-6, seed=7),
    "c": dict(n_res=1200, p=0.02, alpha=0.2, rho=0.97, in_scale=1.0, ridge=1e-6, seed=7),
    "d": dict(n_res=1200, p=0.02, alpha=0.2, rho=0.97, in_scale=1.0, ridge=1e-6, seed=7),
    "e": dict(n_res=1200, p=0.02, alpha=0.2, rho=0.99, in_scale=1.0, ridge=1e-6, seed=7),
}

# ------------------------------------------
# Global timings
# ------------------------------------------
dt = 0.005
T_total = 1500.0
washout_time_hr = 200.0               # HR system initial transient to ignore in plots/metrics
washout_steps_hr = int(washout_time_hr / dt)

# Choose two prediction warm-ups (seconds) per regime for B3:
# T1 should correspond to a short/worse warm-up; T2 a longer/cleaner one.
# If you already have the NRMSE_P vs warmup curve from B2, replace these with those exact choices.
warmups_sec = {
    "a": (50.0, 1000.0),
    "b": (50.0, 1000.0),
    "c": (50.0, 1000.0),
    "d": (50.0, 1000.0),
    "e": (50.0, 1000.0),
}

# ------------------------------------------
# Train ESN readout once per regime (B2)
# ------------------------------------------
def train_esn_for_regime(label, x_full, params):
    """
    Minimal stand-in for B2 training: use teacher forcing on [200s, 1200s] to fit Wout.
    Replace with your actual B2 training if you have it; keep hyperparams identical.
    """
    esn = ESN(**params)
    n = len(x_full)
    # training window
    t_train_start = washout_steps_hr                     # 200s
    t_train_end = int(1200.0 / dt)                       # up to 1200s
    u_train = x_full[t_train_start:t_train_end]
    # internal washout of reservoir before fitting readout
    esn.fit_readout(u_train, washout_steps=500)          # discard first 500 steps of teacher-forced states
    return esn

# ------------------------------------------
# Plotting helper for B3 overlays
# ------------------------------------------
def plot_b3_overlays(label, t, x, esn, T1, T2):
    """
    For a regime label, plot two figures:
    - plot 1: warm-up T1 with solid vline
    - plot 2: warm-up T2 with dashed vline
    x_r(t) is shown only after 200s in both.
    """
    n = len(x)
    start_idx = washout_steps_hr
    end_idx = n  # to the end (1500s)

    def do_case(Twarm, linestyle, title_suffix):
        warm_steps = int(Twarm / dt)
        xr, start_aut_idx = esn.predict_with_warmup(x, warm_steps, start_idx, end_idx)
        # Build aligned time vectors
        t_pred = t[start_aut_idx:end_idx]
        # Plot
        plt.figure(figsize=(10, 4))
        plt.plot(t, x, lw=0.8, label="Ground truth x(t)", color="blue")
        plt.plot(t_pred, xr, lw=0.8, label="ESN output x_r(t)", color="red")
        # vertical line at Twarm
        plt.axvline(x=Twarm + (washout_steps_hr * dt), color="k", linestyle=linestyle, lw=1.2,
                    label=f"Warm-up {Twarm:.0f}s")
        plt.xlim([200, 1500])
        plt.xlabel("Time t")
        plt.ylabel("Membrane potential x(t)")
        plt.title(f"B3 – Regime {label} ({title_suffix})")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(f"Figure_B3_{label}_{int(Twarm)}s.png", dpi=300)

        # Uncomment to save:
        # plt.savefig(f"Figure_B3_{label}_{int(Twarm)}s.png", dpi=300)

    # Important: prediction warm-up starts AFTER 200s.
    # The vertical marker is placed at (200s + Twarm) to show when autonomous mode begins.
    do_case(T1, "-",  f"warm-up T1={T1:.0f}s")
    # re-use the same trained readout but reset internal state before second run
    esn.reset()
    do_case(T2, "--", f"warm-up T2={T2:.0f}s")

# ------------------------------------------
# Run B3 for each regime in your A2 data
# ------------------------------------------
for label in all_data:
    t = all_data[label]["time"]
    x = all_data[label]["x"]

    # train (or load your B2 readout here)
    esn = train_esn_for_regime(label, x, b2_hparams[label])

    T1, T2 = warmups_sec[label]
    plot_b3_overlays(label, t, x, esn, T1=T1, T2=T2)




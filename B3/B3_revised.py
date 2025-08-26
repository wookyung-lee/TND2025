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
    def __init__(self, Nres=600, p=0.02, alpha=0.25, rho=0.9,
                 lambda_reg=1e-3, in_scale=0.8, fb_scale=0.2, random_state=42):
        self.Nres = int(Nres)
        self.p = float(p)
        self.alpha = float(alpha)
        self.rho = float(rho)
        self.lambda_reg = float(lambda_reg)
        self.random_state = random_state
        self.in_scale = in_scale
        self.fb_scale = fb_scale
        # will be set in training
        self.has_bias = True
        self._feat_mean = None
        self._feat_std = None
        self.Wres = None
        self.Win = None
        self.Wout = None
        self._init_weights()

    def _init_weights(self):
        rng = np.random.default_rng(self.random_state)
        # input weights (scaled)
        self.Win = torch.tensor(
            rng.uniform(-0.5, 0.5, size=(self.Nres,)) * self.in_scale,
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
        # Reservoir states driven by normalized u_train
        X, _ = self._reservoir_states(u_train, washout=washout)  # X: (Nres, T')
        u = torch.tensor(u_train[washout:], dtype=torch.float32, device=X.device)  # (T',)

        # Align for one-step-ahead: features at t -> target at t+1
        X = X[:, :-1]  # (Nres, T'-1)
        u_feat = u[:-1].unsqueeze(0)  # (1, T'-1)  current input
        y = u[1:]  # (T'-1,)    next step

        # Add bias row
        bias = torch.ones(1, X.shape[1], device=X.device)  # (1, T'-1)

        # --- Features Φ = [X; bias; u(t)] ---
        Phi = torch.cat([X, bias, u_feat], dim=0)  # (Nres+2, T'-1)

        # --- Row-wise standardization (store stats for rollout) ---
        mu = Phi.mean(dim=1, keepdim=True)
        sd = Phi.std(dim=1, keepdim=True) + 1e-8
        Phi_n = (Phi - mu) / sd
        self._feat_mean = mu
        self._feat_std = sd

        # --- Ridge regression ---
        I = torch.eye(Phi_n.shape[0], device=Phi_n.device)
        self.Wout = torch.linalg.solve(Phi_n @ Phi_n.T + self.lambda_reg * I, Phi_n @ y)
        return float(torch.sqrt(torch.mean(((Phi_n.T @ self.Wout) - y) ** 2)))  # RMSE on train (optional)

    @torch.no_grad()
    def predict_with_warmup(self, u_true, start_idx, warmup_steps, end_idx):
        u = torch.tensor(u_true, dtype=torch.float32, device=device)
        x = torch.zeros(self.Nres, device=device)
        xr = []

        # -------- warm-up with TRUE input --------
        for n in range(start_idx, start_idx + warmup_steps):
            x = (1 - self.alpha) * x + self.alpha * torch.tanh(self.Wres @ x + self.Win * u[n])

        # For the first autonomous step, the "previous input" is the last true u
        u_prev = u[start_idx + warmup_steps - 1] if warmup_steps > 0 else u[start_idx]

        # -------- autonomous rollout --------
        for n in range(start_idx + warmup_steps, end_idx):
            # Build feature vector φ = [x; 1; u_prev] and standardize with training stats
            phi = torch.cat([x, torch.ones(1, device=device), u_prev.view(1)], dim=0).unsqueeze(1)  # (Nres+2,1)
            phi_n = (phi - self._feat_mean) / self._feat_std
            y_hat = (self.Wout @ phi_n).squeeze()  # scalar in normalized space

            # optional clamp in normalized space to keep stable
            y_hat = torch.clamp(y_hat, -3.0, 3.0)

            xr.append(float(y_hat))

            # feedback: use model output as next "input" (normalized domain)
            u_prev = y_hat

            # advance reservoir using scaled feedback
            x = (1 - self.alpha) * x + self.alpha * torch.tanh(self.Wres @ x + self.Win * (self.fb_scale * y_hat))

        return np.array(xr, dtype=float), (start_idx + warmup_steps)


# -------------------------
# B3 config & PKL loaders
# -------------------------
import os, sys, json, pickle, numpy as np

dt = 0.005
t_wash_hr = 200.0
i_wash_hr = int(t_wash_hr / dt)  # 40_000
T_end = 1500.0
# figures folder
outdir = "B3_figs"
os.makedirs(outdir, exist_ok=True)

# --- make numpy._core import-safe for cross-env pickles ---
import numpy.core as _np_core
sys.modules.setdefault("numpy._core", _np_core)

# --- helpers ---
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)



def _to_np(x):
    import numpy as np
    # turn numpy scalars/arrays/lists into float arrays
    if hasattr(x, "tolist"):
        x = x.tolist()
    return np.asarray(x, dtype=float)

def _extract_times_nrmse(entry):
    """
    Return (times_array, errs_array) in *seconds* or *steps* (we convert later).
    Supports many shapes:
      1) {'times'|'warmups'|'steps'|...: [...], 'nrmse'|'errors'|...: [...]}
      2) (times, errs), [times, errs], np.array shape (N,2)
      3) [{'time'|'warmup'|'step': t, 'nrmse'|'error': e}, ...]
      4) {time_or_step: error, ...}  (mapping)
      5) Nested dicts like {'curve': {'warmups': [...], 'NRMSE': [...]} }
    """
    # Case 2: tuple/list of two arrays
    if isinstance(entry, (list, tuple)) and len(entry) == 2:
        t, e = entry
        return _to_np(t), _to_np(e)

    # Case 2b: numpy array (N,2)
    try:
        import numpy as np
        if hasattr(entry, "shape"):
            arr = np.asarray(entry)
            if arr.ndim == 2 and arr.shape[1] == 2:
                return _to_np(arr[:, 0]), _to_np(arr[:, 1])
    except Exception:
        pass

    # If top-level is a dict, try multiple patterns
    if isinstance(entry, dict):
        # If it looks like a nested dict with a single key, dive one level
        if len(entry) == 1 and isinstance(next(iter(entry.values())), dict):
            try:
                return _extract_times_nrmse(next(iter(entry.values())))
            except Exception:
                pass

        # Pattern 1: direct arrays under common keys
        time_keys  = ("times", "warmups", "warmup_times", "t", "time", "T", "steps", "N_warmup", "n_warmup", "warmup_list")
        error_keys = ("nrmse", "NRMSE", "errors", "error", "err", "nrmse_p", "NRMSE_P", "val")
        t_arr = e_arr = None
        for tk in time_keys:
            if tk in entry:
                t_arr = _to_np(entry[tk]); break
        for ek in error_keys:
            if ek in entry:
                e_arr = _to_np(entry[ek]); break
        if t_arr is not None and e_arr is not None:
            return t_arr, e_arr

        # Pattern 3: list-of-dicts embedded under common container keys
        for ck in ("curve", "data", "results", "points", "series"):
            if ck in entry and isinstance(entry[ck], (list, tuple)):
                return _extract_times_nrmse(entry[ck])

        # Pattern 4: mapping time->error (all values numeric)
        if all(not isinstance(v, dict) for v in entry.values()):
            try:
                # keys could be str or numbers; cast to float where possible
                t_list, e_list = [], []
                for k, v in entry.items():
                    try:
                        t_list.append(float(k))
                        e_list.append(float(v))
                    except Exception:
                        # if keys are not numeric (e.g., '50s'), try to parse digits
                        import re
                        m = re.search(r"[\d.]+", str(k))
                        if m:
                            t_list.append(float(m.group()))
                            e_list.append(float(v))
                if len(t_list) > 0:
                    return _to_np(t_list), _to_np(e_list)
            except Exception:
                pass

        # Pattern 5: dict-of-arrays two-level (e.g., {'x': {'times': [...]}, 'y': {'errors': [...]}})
        for v in entry.values():
            if isinstance(v, dict):
                try:
                    return _extract_times_nrmse(v)
                except Exception:
                    continue

    # Case 3: list-of-dicts [{'warmup':..,'nrmse':..}, ...]
    if isinstance(entry, (list, tuple)) and len(entry) > 0 and isinstance(entry[0], dict):
        time_keys  = ("times", "warmup", "step", "time", "t")
        error_keys = ("nrmse", "NRMSE", "error", "errors", "err", "val")
        t_list, e_list = [], []
        for row in entry:
            t_val = e_val = None
            for tk in time_keys:
                if tk in row:
                    t_val = row[tk]; break
            for ek in error_keys:
                if ek in row:
                    e_val = row[ek]; break
            if t_val is not None and e_val is not None:
                t_list.append(t_val); e_list.append(e_val)
        if t_list:
            return _to_np(t_list), _to_np(e_list)

    # If we reach here, we couldn’t parse
    raise ValueError("No time/NRMSE arrays found in entry")

def _steps_to_seconds_if_needed(times):
    # If the largest value looks like steps (e.g., >3000), convert using dt
    return times * dt if float(np.max(times)) > 3000 else times



def pick_T1_T2(times_sec, errs, tol_ratio=1.05):
    """
    T1: time with worst error.
    T2: smallest time whose error is within tol_ratio * best_error.
    """
    idx_bad  = int(np.argmax(errs))
    idx_best = int(np.argmin(errs))
    best = float(errs[idx_best])
    tol = best * tol_ratio
    cands = np.where(errs <= tol)[0]
    idx_t2 = int(cands[0]) if cands.size else idx_best
    return float(times_sec[idx_bad]), float(times_sec[idx_t2])

# ------------------------------------------
# 1) Load optimized hyperparams (normalized)
#     Fallback to unnormalized, else defaults
# ------------------------------------------


# ------------------------------------------
# Robust loader for optimized_params_*.pkl
# Handles multiple schemas and key aliases.
# ------------------------------------------
import re

DEFAULT_OPT = {
    "a": {"Nres": 600,  "p": 0.02, "alpha": 0.30, "rho": 0.90},
    "b": {"Nres": 800,  "p": 0.02, "alpha": 0.25, "rho": 0.90},
    "c": {"Nres": 1000, "p": 0.02, "alpha": 0.20, "rho": 0.90},
    "e": {"Nres": 1000, "p": 0.02, "alpha": 0.20, "rho": 0.90},
}

def _coerce_float(x):
    try:
        return float(getattr(x, "item", lambda: x)())
    except Exception:
        return float(x)

def _coerce_int(x):
    try:
        return int(getattr(x, "item", lambda: x)())
    except Exception:
        return int(x)

def _map_param_names(d):
    """Map various aliases to {Nres,p,alpha,rho}. Returns None if cannot map."""
    if not isinstance(d, dict):
        return None

    # normalize keys (lowercase, strip)
    lk = {k.lower().strip(): k for k in d.keys()}

    def pick(*candidates, cast=float, required=True):
        for c in candidates:
            if c in lk:
                key = lk[c]
                try:
                    return cast(d[key])
                except Exception:
                    pass
        return None if not required else None

    Nres  = pick("nres", "n_res", "n", "nreservoir", "reservoir", "size", cast=_coerce_int)
    p     = pick("p", "density", "connectivity", "sparsity", cast=_coerce_float)
    alpha = pick("alpha", "leak", "leak_rate", cast=_coerce_float)
    rho   = pick("rho", "spectral_radius", "radius", cast=_coerce_float)

    # minimal required: Nres & rho; set sane defaults for missing p/alpha
    if Nres is None and rho is None and p is None and alpha is None:
        return None

    if p is None:     p = 0.02
    if alpha is None: alpha = 0.3
    if rho is None:   rho = 0.75
    if Nres is None:  Nres = 600

    return {"Nres": int(Nres), "p": float(p), "alpha": float(alpha), "rho": float(rho)}

def load_optimized_params():
    # Prefer JSON if present
    for p in ("optimized_params_normalized.json",
              "optimized_params.json",
              "../optimized_params_normalized.json",
              "../optimized_params.json"):
        if os.path.exists(p):
            with open(p) as jf:
                obj = json.load(jf)
            # ensure proper types
            out = {}
            for lbl in ("a","b","c","e"):
                v = obj.get(lbl)
                if isinstance(v, dict):
                    mapped = _map_param_names(v)
                    out[lbl] = mapped or DEFAULT_OPT[lbl]
                else:
                    out[lbl] = DEFAULT_OPT[lbl]
            print(f"[INFO] Loaded optimized params from JSON: {p}")
            return out

    # Try PKLs in common names (normalized preferred)
    candidates = (
        "optimized_params_normalized.pkl",
        "../optimized_params_normalized.pkl",
        "optimized_params.pkl",
        "../optimized_params.pkl",
        "optimized_params_unnormalized.pkl",
        "../optimized_params_unnormalized.pkl",
    )
    for p in candidates:
        if not os.path.exists(p):
            continue
        try:
            with open(p, "rb") as f:
                obj = pickle.load(f)
        except Exception as e:
            print(f"[WARN] Could not load {p}: {e}")
            continue

        # If the pickle has a top-level 'optimized_params', use it
        if isinstance(obj, dict) and "optimized_params" in obj:
            obj = obj["optimized_params"]

        out = {}
        if isinstance(obj, dict):
            # Case 1: {'a': {...}, 'b': {...}, ...}
            if all(k in obj for k in ("a","b","c","e")):
                for lbl in ("a","b","c","e"):
                    mapped = _map_param_names(obj.get(lbl, {}))
                    out[lbl] = mapped or DEFAULT_OPT[lbl]
                print(f"[INFO] Loaded optimized params from PKL: {p}")
                return out

            # Case 2: key


opt = load_optimized_params()

# ------------------------------------------
# 2) Build warmups_sec from warmup_times_*.pkl
#     (normalized preferred)
# ------------------------------------------
def load_warmup_choices():
    # load the whole dict of curves
    pkl_candidates = (
        "warmup_times_normalized.pkl",
        "../warmup_times_normalized.pkl",
        "warmup_times.pkl",
        "../warmup_times.pkl",
        "warmup_times_unnormalized.pkl",
        "../warmup_times_unnormalized.pkl",
    )
    curves = None
    for p in pkl_candidates:
        if os.path.exists(p):
            curves = load_pickle(p)
            print(f"Loaded warm-up curves from: {p}")
            break
    if curves is None:
        # fallback: keep your old fixed choices
        return {"a": (50.0, 1000.0), "b": (50.0, 1000.0),
                "c": (50.0, 1000.0), "e": (50.0, 1000.0)}

    warmups_sec = {}
    for lbl in ("a", "b", "c", "e"):
        if lbl not in curves:
            continue
        times, errs = _extract_times_nrmse(curves[lbl])
        times_sec = _steps_to_seconds_if_needed(times)
        # choose T1/T2
        T1, T2 = pick_T1_T2(times_sec, errs, tol_ratio=1.05)

        # clamp to available window (warm-up starts at 200s)
        max_warm = T_end - t_wash_hr - 1.0  # leave >=1s for autonomous
        T1 = float(min(max(T1, 0.0), max_warm))
        T2 = float(min(max(T2, 0.0), max_warm))

        warmups_sec[lbl] = (T1, T2)
        print(f"Regime {lbl}: T1={T1:.1f}s (worst), T2={T2:.1f}s (near-best)")
    return warmups_sec

warmups_sec = load_warmup_choices()


# ---- Training helper (consistent normalization & washout) ----
def train_readout_for_regime(x_full, params):
    esn = ESN(
        Nres=int(params["Nres"]),
        p=float(params["p"]),
        alpha=float(params["alpha"]),
        rho=float(params["rho"]),
        lambda_reg=1e-6,
        random_state=42,
    )

    # normalize after washout (>=200s)
    u = x_full[i_wash_hr:]
    mu, sd = float(np.mean(u)), float(np.std(u) + 1e-12)
    u_norm = (u - mu) / sd

    N_train = len(u_norm) // 2
    u_tr = u_norm[:N_train]

    # states
    X, _ = esn._reservoir_states(u_tr, washout=500)
    y = torch.tensor(u_tr[501:], dtype=torch.float32, device=X.device)
    X = X[:, :-1]

    # ---- add bias term ----
    ones = torch.ones(1, X.shape[1], device=X.device)
    Xb = torch.cat([X, ones], dim=0)   # shape (Nres+1, T)

    I = torch.eye(Xb.shape[0], device=X.device)
    esn.Wout = torch.linalg.solve(Xb @ Xb.T + esn.lambda_reg * I, Xb @ y)
    esn.train(u_norm[:N_train], washout=500)
    val_err = open_loop_nrmse(esn, u_norm[N_train - 1000:N_train + 5000])  # a small slice around boundary
    print(f"[{label}] open-loop NRMSE ≈ {val_err:.3f}")
    esn.has_bias = True  # flag
    return esn, mu, sd

# ---- Plot helper ----
def plot_overlay(label, t, x, esn, mu, sd, T_warm, linestyle, tag):
    start_idx = i_wash_hr
    end_idx = len(x)

    # normalize the WHOLE series with the SAME (mu, sd)
    u_full_norm = (x - mu) / sd

    # convert warm-up seconds to steps and clamp so we have some prediction window left
    warm_steps = int(T_warm / dt)
    max_warm = max(0, (end_idx - start_idx - 2))  # leave >= 2 steps
    warm_steps = max(0, min(warm_steps, max_warm))

    # teacher-forcing warm-up on normalized input, then autonomous prediction in normalized scale
    xr_norm, start_aut = esn.predict_with_warmup(
        u_full_norm, start_idx=start_idx, warmup_steps=warm_steps, end_idx=end_idx
    )
    t_pred = t[start_aut:end_idx]

    # de-normalize model output for plotting
    xr = xr_norm * sd + mu

    plt.figure(figsize=(10, 4))
    plt.plot(t, x, lw=0.9, label="Ground truth x(t)", color="blue")
    plt.plot(t_pred, xr, lw=0.9, label="ESN output x_r(t)", color="red")
    plt.axvline(x=t_wash_hr + warm_steps*dt, color="k", linestyle=linestyle, lw=1.1,
                label=f"Warm-up {warm_steps*dt:.0f}s")
    plt.xlim([200, 1500])
    plt.xlabel("Time t")
    plt.ylabel("Membrane potential x(t)")
    plt.title(f"B3 – Regime {label} ({tag})")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"B3_{label}_{int(warm_steps*dt)}s.png"), dpi=300)
    plt.close()

def rollout_nrmse(esn, x, mu, sd, start_idx, warm_s, end_idx):
    u_full_norm = (x - mu) / sd
    xr_norm, start_aut = esn.predict_with_warmup(u_full_norm, start_idx, warm_s, end_idx)
    xr = xr_norm * sd + mu
    y_true = x[start_aut:end_idx]
    err = np.sqrt(np.mean((xr - y_true)**2)) / (np.std(y_true) + 1e-12)
    return float(err)
def open_loop_nrmse(esn, u_val):
    # drive with true input, predict next step with readout features [x;1;u(t)]
    X, _ = esn._reservoir_states(u_val, washout=0)
    u = torch.tensor(u_val, dtype=torch.float32, device=X.device)
    X = X[:, :-1]
    u_feat = u[:-1].unsqueeze(0)
    bias = torch.ones(1, X.shape[1], device=X.device)
    Phi = torch.cat([X, bias, u_feat], dim=0)

    Phi_n = (Phi - esn._feat_mean) / esn._feat_std
    y_hat = (Phi_n.T @ esn.Wout).cpu().numpy()
    y = u[1:].cpu().numpy()
    return float(np.sqrt(np.mean((y_hat - y)**2)) / (np.std(y) + 1e-12))

for label in ["a", "b", "c", "e"]:
    t = all_data[label]["time"]
    x = all_data[label]["x"]

    params = opt.get(label, DEFAULT_OPT[label])
    esn, mu, sd = train_readout_for_regime(x, params)
    # before plotting
    T1, T2 = warmups_sec[label]
    warm_steps = int(T2 / dt)
    err = rollout_nrmse(esn, x, mu, sd, i_wash_hr, warm_steps, len(x))
    print(f"[{label}] rho={params['rho']} alpha={params['alpha']} in={esn.in_scale} fb={esn.fb_scale}  NRMSE={err:.3f}")

    # enforce reasonable minimums (seconds)
    if label in ("b", "e"):
        T1 = max(T1, 200.0)
        T2 = max(T2, 700.0)
    else:
        T1 = max(T1, 50.0)
        T2 = max(T2, 300.0)
    plot_overlay(label, t, x, esn, mu, sd, T1, "-",  f"T1={T1:.0f}s")
    plot_overlay(label, t, x, esn, mu, sd, T2, "--", f"T2={T2:.0f}s")



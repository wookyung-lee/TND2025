import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
from ESN import ESN
import os

# --- constants to match the project spec ---
dt = 0.005          # time step
T_total = 1500.0    # total duration
Tt = 200.0          # washed-out transient (time units)
seed = 42
np.random.seed(seed)

# Convert times to indices
N_total = int(T_total / dt)            # total samples
N_transient = int(Tt / dt)             # 200 / 0.005 = 40000

# Paths
OUTDIR = "./B3"
os.makedirs(OUTDIR, exist_ok=True)

def load_data():
    with open("../timeseries_data.pkl", "rb") as f:
        return pickle.load(f)

def load_warmups():
    with open("../warmup_times.pkl", "rb") as f:
        return pickle.load(f)

def get_optimized_params():
    # Keep consistent with your B2 optimized set (or load if you have a pkl)
    return {
        "a" : { 'Nres':760, 'p':0.2, 'alpha':0.7, 'rho':0.9 },
        "b" : { 'Nres':920, 'p':0.2, 'alpha':0.9, 'rho':0.75 },
        "c" : { 'Nres':450, 'p':0.2, 'alpha':0.7, 'rho':1.05 },
        "e" : { 'Nres':300, 'p':0.2, 'alpha':0.5, 'rho':1.05 }
    }

def prepare_series(x_full):
    """
    Returns:
      t (0..1500), x (full), x_after (after 200),
      u (inputs), y (targets),
      u_train, y_train (first half after transient),
      u_pred, y_pred (second half after transient),
      scaler (fit on post-transient data)
    """
    x = np.asarray(x_full).flatten()
    x = x[:N_total]  # ensure same length
    t = np.arange(N_total) * dt

    # post-transient segment
    x_after = x[N_transient:]
    # standardize on post-transient data (as in B2)
    scaler = StandardScaler()
    x_after_norm = scaler.fit_transform(x_after.reshape(-1, 1)).flatten()

    # build supervised pairs on post-transient segment
    inputs = x_after_norm[:-1]
    targets = x_after_norm[1:]

    # split 50/50 for train/predict (per spec)
    N = len(inputs)
    N_train = N // 2
    u_train, y_train = inputs[:N_train], targets[:N_train]
    u_pred,  y_pred  = inputs[N_train:], targets[N_train:]

    return t, x, x_after, inputs, targets, u_train, y_train, u_pred, y_pred, scaler
def safe_nrmse(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    L = min(len(y_true), len(y_pred))
    if L == 0:
        return np.nan
    y_true = y_true[:L]
    y_pred = y_pred[:L]
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    return rmse / (np.std(y_true) + 1e-12)


def train_and_predict(label, params, Nwarmup, x_full):
    """
    Train on first half of post-transient data; predict on second half.
    Returns:
      t (0..1500), x (truth), xr (ESN, aligned to t in (200,1500]),
      t_warm_abs (absolute time of warm-up boundary),
      nrmse_pred (NRMSE on prediction segment)
    """
    t, x, x_after, inputs, targets, u_train, y_train, u_pred, y_pred, scaler = prepare_series(x_full)

    # Build ESN with given hyperparams
    esn = ESN(Nres=params['Nres'], p=params['p'], alpha=params['alpha'], rho=params['rho'])

    # Train
    esn.train(u_train, y_train)

    # Predict on the post-train half
    y_pred_norm_hat, _ = esn.predict(u_pred, n_autonomous=len(y_pred))

    # Inverse transform predictions and targets back to original scale
    y_pred_hat = scaler.inverse_transform(np.asarray(y_pred_norm_hat).reshape(-1,1)).flatten()
    y_pred_true = scaler.inverse_transform(np.asarray(y_pred).reshape(-1,1)).flatten()

    # Compute NRMSE on prediction phase for reference
    nrmse_pred = safe_nrmse(y_pred_true, y_pred_hat)


    # Align ESN time series xr(t) to absolute time axis.
    # The prediction phase starts at time t_pred_start = 200 + (N_after/2)*dt,
    # but the plotted requirement is to show xr(t) over (200,1500]; here we show the ESN output
    # over its prediction window and overlay it on the absolute t axis.
    N_after = len(x_after)               # post-transient samples
    N_half  = N_after // 2               # start of prediction half (in samples after 200)
    t_pred_start = Tt + N_half * dt

    # Create an array xr aligned to t; fill NaNs except where we have predictions
    xr = np.full_like(x, np.nan, dtype=float)
    # y_pred indexes correspond to times from t_pred_start + dt onward (since target y is x shifted by 1)
    # We'll map y_pred_hat to those absolute indices.
    start_idx_abs = int(t_pred_start / dt)  # align exactly with the first target index

    end_idx_abs = start_idx_abs + len(y_pred_hat)
    xr[start_idx_abs:end_idx_abs] = y_pred_hat

    # Warm-up line in absolute time:
    # warm-up happens at the beginning of the prediction half: t = 200 + N_half*dt + Nwarmup*dt
    t_warm_abs = t_pred_start + Nwarmup * dt

    return t, x, xr, t_warm_abs, nrmse_pred

def plot_overlay(label, t, x, xr, t_warm_abs, style, title_suffix, outfile):
    plt.figure(figsize=(11,3))
    # ground truth in blue for [0,1500]
    plt.plot(t, x, linewidth=0.8, label='x(t) (Ground truth)')
    # ESN in red only where defined (after (200,1500])
    plt.plot(t, xr, linewidth=0.8, label='x_r(t) (ESN)', color='red')
    # vertical warm-up indicator
    if style == 'solid':
        plt.axvline(t_warm_abs, linestyle='-', linewidth=1.2, color='k', label='Warm-up boundary (T1)')
    else:
        plt.axvline(t_warm_abs, linestyle='--', linewidth=1.2, color='k', label='Warm-up boundary (T2)')

    plt.xlim(0, T_total)
    plt.xlabel('Time')
    plt.ylabel('Membrane potential x')
    plt.title(f'Problem B.3 – Regime {label.upper()} {title_suffix}')
    plt.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.01)
    plt.close()

def main():
    all_data = load_data()
    warmups = load_warmups()
    params_map = get_optimized_params()

    # Map the regimes to Problem A 2(.) as in the assignment:
    # a -> 2(a): periodic spikes (I=3.5, r=0.003)
    # b -> 2(b): chaotic spikes (I=3.34, r=0.003)
    # c -> 2(c): periodic bursts (3 spikes/burst) (I=1.67, r=0.003)
    # e -> 2(e): chaotic bursts (I=3.29, r=0.003)
    for label in ['a','b','c','e']:
        x_full = all_data[label]['x']

        params = params_map[label]
        T1 = int(warmups[label]['T1'])
        T2 = int(warmups[label]['T2'])

        # --- Plot with T1 (highest NRMSEP) ---
        t, x, xr, t_warm_abs, nrmse1 = train_and_predict(label, params, T1, x_full)
        plot_overlay(
            label, t, x, xr, t_warm_abs,
            style='solid',
            title_suffix=f'(T1={t_warm_abs:.2f}, Nwarmup={T1}, NRMSEP={nrmse1:.4f})',

            outfile=os.path.join(OUTDIR, f"Prob-B-3{label}_T1.png")
        )

        # --- Plot with T2 (later warm-up with drop in error) ---
        t, x, xr, t_warm_abs, nrmse2 = train_and_predict(label, params, T2, x_full)
        plot_overlay(
            label, t, x, xr, t_warm_abs,
            style='dashed',
            title_suffix=f'(T2={t_warm_abs:.2f}, Nwarmup={T2}, NRMSEP={nrmse2:.4f})',

            outfile=os.path.join(OUTDIR, f"Prob-B-3{label}_T2.png")
        )

        print(f"[Regime {label}] Saved: Prob-B-3{label}_T1.png (NRMSEP={nrmse1:.4f}), Prob-B-3{label}_T2.png (NRMSEP={nrmse2:.4f})")

if __name__ == "__main__":
    main()

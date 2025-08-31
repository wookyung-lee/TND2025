# TND Project — Problems A1, A2, B1, B2, B3, B4

This repository organizes all six problems from the Theory of Neural Dynamics project with consistent run instructions, dependencies, and per‑problem notes.

## Environment

- **Python**: 3.9+ recommended
- **Packages**: A2, ESN, joblib, matplotlib, numpy, torch, torchdiffeq, tqdm

Install (example):
```bash
pip install A2 ESN joblib matplotlib numpy torch torchdiffeq tqdm
```

## Repository layout (filenames as provided)

```
.
├── Prob-A-1a.py
├── Prob-A-2a.py
├── Prob-A-2.py
├── Prob-B-1.py
├── Prob-B-2.py
├── Prob-B-3.py
├── Prob-B-4.py
└── README.md
```

---

## Problem A1 — Bifurcation: I vs log(ISI)

Integrates the **Hindmarsh–Rose (HR)** neuron model with RK4 across a current sweep and plots a bifurcation diagram of **I vs log(ISI)**. GPU acceleration via PyTorch if available.

**Dependencies:** matplotlib, numpy, torch, tqdm

**How to run**
```bash
python Prob-A-1a.py Prob-A-1b.py
```
**Outputs:**
- Prob-A-1a.png
- Prob-A-1b.png

**Key parameters (typical in this script):** sweep `I∈[2.5,3.5]` with `ΔI=0.005`, `r=0.003`, `T=1500`, `dt=0.005`, transient `T_t≈200`, threshold `x_th=1.0` (crossing from below). Output figure `Prob-A-1a.png`.

---

## Problem A2 — Time Series in Five Regimes

Simulates HR via fixed‑step RK4 for **five canonical regimes** (periodic/chaotic spikes and bursts), and optionally plots `x(t)`. Saves all time series to a single pickle file for downstream use.

**Dependencies:** matplotlib, numpy

**How to run**
```bash
python Prob-A-2.py
```
**Outputs:**
- data: timeseries_data.pkl
- figure: os.path.join(script_dir, f"Figure_A2_{label}.png"

**Cases (a–e):** a: `(I=3.50,r=0.003)`, b: `(3.34,0.003)`, c: `(1.67,0.003)`, d: `(3.20,0.003)`, e: `(3.29,0.003)`. Output data `timeseries_data.pkl`.

---

## Problem B1 — Analysis

Hyperparameter sensitivity analysis of an Echo State Network (ESN) on the Hindmarsh–Rose (HR) neuron model).

**Dependencies:** A2, ESN, joblib, matplotlib, numpy, torch

**How to run**
```bash
python Prob-B-1.py
```
**Outputs:**
- figures: plots of NRMSE vs hyperparameter values for each regime (a, b, c, e) and each hyperparameter (Nres, p, alpha, rho).
- data: optimal_hyperparams.pkl — contains the dictionary of optimal hyperparameters (value + minimum NRMSE) for each regime
---

## Problem B2 — Analysis Warm-up time vs. NRMSE in Prediction

Given the hyperparameters from Problem B1, the NRMSE in prediction is calculated for different warm-up parameter values.
hyperparameter_optimization can be used as an alternative to B1 to get hyperparameters with Optuna/scikit-learn.

**Dependencies:** ESN, matplotlib, numpy, tqdm, pickle, scikit-learn

**How to run**
```bash
python Prob-B-2.py
```
**Outputs:**
- data: warmup_times.pkl
- figures: B2/Prob-B-2x.png

---

## Problem B3 — Echo State Network Warm-Up Comparison

This project implements **Problem B3** from the TND assignment:  
*Investigating the effect of warm-up duration on Echo State Network (ESN) predictions of Hindmarsh–Rose (HR) neuron dynamics.*

---

### 📌 Overview

We first simulate HR neuron voltage traces for different dynamical regimes, then train an ESN to reproduce these dynamics. The ESN is evaluated under two different **warm-up durations** (short vs long), to show how the length of teacher forcing before autonomous prediction influences the forecast quality.

---

### Steps Implemented

1. **Hindmarsh–Rose neuron simulation**  
   - Integrated using a 4th-order Runge–Kutta method.  
   - Simulated for 1500 s with timestep `dt = 0.005`.  
   - Regimes included:  
     - (a) Periodic spiking (I=3.5, r=0.003)  
     - (b) Chaotic spiking (I=3.34, r=0.003)  
     - (c) Periodic bursting (I=1.67, r=0.003)  
     - (e) Chaotic bursting (I=3.29, r=0.003)  

2. **Echo State Network (ESN)**  
   - Reservoir built as an Erdős–Rényi random graph, scaled to desired spectral radius.  
   - Input weights sampled uniformly from [-0.5, 0.5].  
   - Leaky tanh activation with leak rate `α`.  
   - Readout trained using **ridge regression** with washout of 500 steps.  

3. **Training procedure**  
   - First 200 s of the HR signal discarded (transient).  
   - Remaining signal normalized (zero mean, unit variance).  
   - Split into 50% training, 50% testing.  
   - ESN readout trained on the training part.

4. **Warm-up experiments**  
   - After 200 s, ESN is driven by the **true HR signal** for a chosen warm-up duration.  
   - Then switched to **closed-loop (autonomous) mode** until the end of the simulation.  
   - Two warm-up durations tested per regime:  
     - Short warm-up: **T1 = 50 s**  
     - Long warm-up: **T2 = 1000 s**

5. **Visualization**  
   - For each regime (a, b, c, e), two plots are produced:  
     - Blue = ground truth HR signal.  
     - Red = ESN prediction after warm-up.  
     - Vertical line marks the warm-up cutoff.  
   - Plots are saved in the `B3_figs/` folder.


---

## Problem B4 — Analysis

Spike analysis / ISI-based metrics. Detected model: **Hindmarsh–Rose (HR) neuron model**.

**Dependencies:** ESN, matplotlib, torch, torchdiffeq

**How to run**
```bash
python Prob-B-4.py
```
**Outputs:**
- figure: f'Prob-B-4_{key}.png'

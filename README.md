# TND Project — Problems A1, A2, B1, B2, B3, B4

This repository organizes all six problems from the Theory of Neural Dynamics project with consistent run instructions, dependencies, and per‑problem notes.

## Environment

- **Python**: 3.9+ recommended
- **Packages**: A2, ESN, ESN_new, joblib, matplotlib, numpy, torch, torchdiffeq, tqdm

Install (example):
```bash
pip install A2 ESN ESN_new joblib matplotlib numpy torch torchdiffeq tqdm
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
python Prob-A-1a.py
```
**Outputs:**
- figure: "Prob-A-1a.png", dpi=600, bbox_inches='tight', pad_inches=0.01

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

Learning / modeling. Detected model: **Hindmarsh–Rose (HR) neuron model**.

**Dependencies:** A2, ESN_new, joblib, matplotlib, numpy, torch

**How to run**
```bash
python Prob-B-1.py
```
**Outputs:**
- figure: os.path.join(label_dir, filename

---

## Problem B2 — Analysis

Learning / modeling. Detected model: **Model/analysis per script**.

**Dependencies:** ESN_new, matplotlib, numpy, tqdm

**How to run**
```bash
python Prob-B-2.py
```
**Outputs:**
- data: warmup_times.pkl
- figure: f"./B2/Prob-B-2{label}.png", dpi=600, bbox_inches='tight', pad_inches=0.01

---

## Problem B3 — Analysis

Phase / Hilbert transform analysis. Detected model: **Model/analysis per script**.

**Dependencies:** ESN_new, matplotlib, numpy

**How to run**
```bash
python Prob-B-3.py
```
**Outputs:**
- figure: outfile, dpi=600, bbox_inches='tight', pad_inches=0.01

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

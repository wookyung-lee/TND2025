# Problem B3 – Echo State Network Warm-Up Comparison

This project implements **Problem B3** from the TND assignment:  
*Investigating the effect of warm-up duration on Echo State Network (ESN) predictions of Hindmarsh–Rose (HR) neuron dynamics.*

---

## 📌 Overview

We first simulate HR neuron voltage traces for different dynamical regimes, then train an ESN to reproduce these dynamics. The ESN is evaluated under two different **warm-up durations** (short vs long), to show how the length of teacher forcing before autonomous prediction influences the forecast quality.

---

## 🔬 Steps Implemented

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

## 📂 Project Structure

```
├── B3_revised.py      # Main script for B3 overlays
├── ESN.py             # ESN class (if kept separate)
├── optimized_params.json / .pkl  # Hyperparameters from B2 (optional)
├── B3_figs/           # Generated figures (output)
└── README.md          # This file
```

---

## ▶️ How to Run

1. Ensure you have the dependencies installed:
   ```bash
   pip install numpy matplotlib torch networkx scikit-learn
   ```
   *(Torch can be CPU-only; no GPU required.)*

2. Run the B3 script:
   ```bash
   python B3_revised.py
   ```

3. Output:
   - Figures will be saved in the `B3_figs/` directory.  
   - Each figure corresponds to one regime and one warm-up duration.

---

## 📊 Expected Results

- With **short warm-up (50 s)**: ESN predictions quickly diverge from the true HR signal.  
- With **long warm-up (1000 s)**: ESN stays synchronized with the neuron dynamics for much longer.  
- This highlights the importance of adequate warm-up to align reservoir states before autonomous prediction.

---

## ✍️ Authors / Credits

- Assignment work for **Theory of Neural Dynamics (TND)**, 2025.  
- Hindmarsh–Rose simulation and ESN implementation by *[Your Name]*.

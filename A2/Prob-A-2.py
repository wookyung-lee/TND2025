
import numpy as np
import matplotlib.pyplot as plt
import os

# Hindmarsh-Rose model equations
def hr_model(state, I, r):
    x, y, z = state
    dx = y + 3*x**2 - x**3 - z + I
    dy = 1 - 5*x**2 - y
    dz = r * (4*(x + 8/5) - z)
    return np.array([dx, dy, dz])

# Runge–Kutta 4th order integrator
def rk4_step(func, state, dt, I, r):
    k1 = func(state, I, r)
    k2 = func(state + 0.5*dt*k1, I, r)
    k3 = func(state + 0.5*dt*k2, I, r)
    k4 = func(state + dt*k3, I, r)
    return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# Integrate HR model
def integrate_hr(I, r, T=1500, dt=0.005, init_state=(-1.0, 2.0, 0.5)):
    n_steps = int(T/dt)
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

    plt.figure(figsize=(10, 4))
    plt.plot(t, x, lw=0.5, color="blue")
    plt.title(f"Problem A.{label}")
    plt.xlabel("Time t (ms)")
    plt.ylabel("Membrane potential x(t)")
    plt.xlim([200, 1500])
    plt.tight_layout()

    # Save plots
    # script_dir = os.path.dirname(os.path.abspath(__file__))  # folder of the script
    # plt.savefig(os.path.join(script_dir, f"Figure_A2_{label}.png"), dpi=300)
    
    # plt.show()
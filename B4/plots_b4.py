import numpy as np
import matplotlib.pyplot as plt

# Load saved results
data = np.load("b4_results.npz")
all_I = data["I"]
all_logISI = data["logISI"]

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(all_I, all_logISI, s=1, c="red", alpha=0.5)
plt.xlabel("I[mV]")
plt.ylabel("log(ISI)")
plt.title("Problem B4: ESN-based Bifurcation Diagram")
plt.grid(True)
plt.savefig("Prob-B-4-replot.png", dpi=600, bbox_inches='tight', pad_inches=0.01)
plt.show()

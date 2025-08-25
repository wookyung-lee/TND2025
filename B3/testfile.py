


import pickle
with open("optimized_params.pkl", "rb") as f:
    opt = pickle.load(f)  # opt[label]['Nres'|'p'|'alpha'|'rho']
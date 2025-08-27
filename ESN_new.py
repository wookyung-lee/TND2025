from sklearn.linear_model import Ridge
import networkx as nx
from sklearn.metrics import mean_squared_error
import numpy as np

seed = 42
np.random.seed(seed)
train_fraction = 0.5
dt = 0.005
Tt = 40000

# ESN class
class ESN:
    def __init__(self, Nres, p, alpha, rho):
        self.Nres = Nres
        self.alpha = alpha
        self.rho = rho
        self.p = p
        self.Win = np.random.uniform(-0.5, 0.5, (Nres, 1))
        graph = nx.erdos_renyi_graph(Nres, p, directed=True)
        Wres = nx.to_numpy_array(graph)
        eigenvalues = np.linalg.eigvals(Wres)
        spectral_radius = np.max(np.abs(eigenvalues))
        self.Wres = (Wres / spectral_radius) * rho
        self.r = np.zeros((Nres, 1))
        self.Wout = None
        self.ridge = Ridge(alpha=1e-6, fit_intercept=False)
    
    def update_reservoir(self, u):
        self.r = (1.0 - self.alpha) * self.r + self.alpha * np.tanh(
            self.Wres @ self.r + self.Win * u
        )
        return self.r
    
    def calculate_nrmse(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        nrmse = rmse / np.std(y_true)
        return nrmse
    
    def train(self, u, y, warmup=1000):
        # Ensure warmup < len(u)
        warmup = min(warmup, len(u)-1)

        states = []
        for ut in u:
            r = self.update_reservoir(ut)
            states.append(r.flatten())
        states = np.array(states)

        states = states[warmup:]
        y = y[warmup:]

        self.ridge.fit(states, y)
        self.Wout = self.ridge.coef_

        # Compute NRMSE on training data
        y_pred = self.ridge.predict(states)
        nrmse = self.calculate_nrmse(y, y_pred)
        return nrmse
    
    def predict(self, u, y=None):
        self.r = np.zeros((self.Nres, 1))
        predictions = []
        for ut in u:
            r = self.update_reservoir(ut)
            y_pred = self.Wout @ r
            predictions.append(y_pred.flatten())
        predictions = np.array(predictions)
        nrmse = None
        if y is not None:
            nrmse = self.calculate_nrmse(y, predictions)
        return predictions, nrmse

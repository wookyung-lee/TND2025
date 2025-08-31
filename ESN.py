from sklearn.linear_model import Ridge
import networkx as nx
from sklearn.metrics import mean_squared_error
import numpy as np
from copy import deepcopy

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
        for node in graph.nodes():
            if np.random.random() < p:
                graph.add_edge(node, node)
        Wres = nx.to_numpy_array(graph)
        Wres = np.where(Wres > 0, np.random.uniform(-1, 1, Wres.shape), Wres)
        eigenvalues = np.linalg.eigvals(Wres)
        spectral_radius = np.max(np.abs(eigenvalues))
        self.Wres = (Wres / spectral_radius) * rho
        self.r = np.zeros((Nres, 1))
        self.Wout = None
        self.ridge = Ridge(alpha=1e-6, fit_intercept=False)
    
    def copy(self):
        copy = object.__new__(self.__class__)
        for attr, val in self.__dict__.items():
            if isinstance(val, np.ndarray):
                setattr(copy, attr, val.copy())
            else:
                setattr(copy, attr, deepcopy(val))
        return copy
    
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
    
    def train(self, u, y):
        self.r = np.zeros((self.Nres, 1))
        
        states = []
        for ut in u:
            r = self.update_reservoir(ut)
            states.append(r.flatten())
        states = np.array(states)
        
        self.ridge.fit(states, y)
        self.Wout = self.ridge.coef_
        
        # Compute NRMSE on training data
        y_pred = self.ridge.predict(states)
        nrmse = self.calculate_nrmse(y, y_pred)
        return nrmse
    
    def predict_autonomous(self, u_warmup, n_autonomous, y=None):
        self.r = np.zeros((self.Nres, 1))
        
        for ut in u_warmup:
            self.update_reservoir(ut)
        
        ut = self.Wout @ self.r
        predictions = []
        
        for _ in range(n_autonomous):
            r = self.update_reservoir(ut)
            y_pred = self.Wout @ r
            predictions.append(y_pred)
            ut = y_pred
        
        predictions = np.array(predictions)
        
        nrmse = None
        if y is not None:
            nrmse = self.calculate_nrmse(y, predictions)
        
        return predictions, nrmse
    
    def predict_regression(self, u_warmup, u_predict, y=None):
        self.r = np.zeros((self.Nres, 1))
        
        for ut in u_warmup:
            self.update_reservoir(ut)
        
        predictions = []
        for ut in u_predict:
            r = self.update_reservoir(ut)
            y_pred = self.Wout @ r
            predictions.append(y_pred)
        
        predictions = np.array(predictions)
        
        nrmse = None
        if y is not None:
            nrmse = self.calculate_nrmse(y, predictions)
        
        return predictions, nrmse

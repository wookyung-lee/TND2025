import numpy as np
import networkx as nx

class ESN:
    def __init__(self, Nres=300, p=0.75, alpha=0.5, rho=0.85, lambda_reg=1e-6, random_state=None):
        self.Nres = Nres
        self.p = p
        self.alpha = alpha
        self.rho = rho
        self.lambda_reg = lambda_reg
        self.random_state = random_state
        self.Win = None
        self.Wres = None
        self.Wout = None
        self._init_weights()

    def _init_weights(self):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Input weights: Nres x 1 for single input
        self.Win = np.random.uniform(-0.5, 0.5, size=(self.Nres,))
        
        # Reservoir weights
        G = nx.erdos_renyi_graph(self.Nres, self.p, directed=True, seed=None)

        # Add self-loops with probability self.p
        for node in G.nodes():
            if np.random.random() < self.p:
                G.add_edge(node, node)
        
        Wres = nx.to_numpy_array(G)
        Wres[Wres != 0] = np.random.uniform(-1, 1, size=np.count_nonzero(Wres))

        # Scale weights to achieve desired spectral radius
        eigvals = np.linalg.eigvals(Wres)
        max_eig = max(abs(eigvals))
        if max_eig > 0:
            Wres *= self.rho / max_eig

    def _reservoir_states(self, input_u):
        T = len(input_u)

        x_res = np.zeros((self.Nres, T))
        x = np.zeros(self.Nres)

        # Compute new state: (1-α) * x(t) + α * tanh(W_res * x(t) + W_in * u(t))
        for t in range(T):
            x = (1 - self.alpha) * x + self.alpha * np.tanh(self.Wres @ x + self.Win * input_u[t])
            x_res[:, t] = x

        return x_res
    
    def train(self, u_train, y_train):
        X_res = self._reservoir_states(u_train)  # GPU tensor

        # Closed-form Ridge regression (Tikhonov)
        self.Wout = np.linalg.solve(X_res @ X_res.T + self.lambda_reg * I, X_res @ y_train)

        # Predict on training set
        y_pred = X_res.T @ self.Wout
        nrmse = np.sqrt(np.mean((y_pred - y_train)**2)) / np.std(y_train)

        return nrmse



import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch

# scp C:\Users\prizl\Documents\GitHub\TND2025\B1\wooki\ESN.py qi24jovo@cip3a0.cip.cs.fau.de:~
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        np.random.seed(self.random_state)
        # Input weights: Nres x 1 for single input
        self.Win = torch.tensor(np.random.uniform(-0.5, 0.5, size=(self.Nres,)), dtype=torch.float32, device=device)
        
        # Reservoir weights
        G = nx.erdos_renyi_graph(self.Nres, self.p, directed=True, seed=None)

        # Add self-loops with probability self.p
        for node in G.nodes():
            if np.random.random() < self.p:
                G.add_edge(node, node)
        
        Wres = nx.to_numpy_array(G)
        Wres[Wres != 0] = np.random.uniform(-1, 1, size=np.count_nonzero(Wres))

        # Scale weights to achieve desired spectral radius
        Wres = torch.tensor(Wres, device='cuda')  # or device='cpu' if CPU
        eigvals = torch.linalg.eigvals(Wres)
        max_eig = max(abs(eigvals))
        if max_eig > 0:
            Wres *= self.rho / max_eig

        # self.Wres = torch.tensor(Wres, dtype=torch.float32, device=device)
        self.Wres = Wres.clone().detach().to(dtype=torch.float32, device=device)


    def _reservoir_states(self, input_u):
        T = len(input_u)
        input_u = torch.tensor(input_u, dtype=torch.float32, device=device)  # move input to GPU
        x_res = torch.zeros((self.Nres, T), device=device)
        x = torch.zeros(self.Nres, device=device)

        # Compute new state: (1-α) * x(t) + α * tanh(W_res * x(t) + W_in * u(t))
        for t in range(T):
            x = (1 - self.alpha) * x + self.alpha * torch.tanh(self.Wres @ x + self.Win * input_u[t])
            x_res[:, t] = x

        return x_res
    
    def train(self, u_train, y_train):
        X_res = self._reservoir_states(u_train)  # GPU tensor
        y_train = torch.tensor(y_train, dtype=torch.float32, device=device)

        # Closed-form Ridge regression (Tikhonov) on GPU
        I = torch.eye(self.Nres, device=device)
        A = X_res @ X_res.T + self.lambda_reg * I
        B = X_res @ y_train
        self.Wout = torch.linalg.solve(A, B)
        # self.Wout = np.linalg.solve(X_res @ X_res.T + self.lambda_reg * I, X_res @ y_train)

        # Predict on training set
        y_pred = X_res.T @ self.Wout
        nrmse = torch.sqrt(torch.mean((y_pred - y_train)**2)) / torch.std(y_train)
        return nrmse.item()



import numpy as np
import networkx as nx
import torch

# scp C:\Users\prizl\Documents\GitHub\TND2025\B1\ESN.py qi24jovo@cip3a0.cip.cs.fau.de:~
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

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
        self.x = None
        self._norm_params = None
        self._init_weights()
    
    def _init_weights(self):
        rng = np.random.RandomState(self.random_state)
        # Input weights: Nres x 1 for single input
        self.Win = torch.tensor(
            rng.uniform(-0.5, 0.5, size=(self.Nres,)).astype(np.float32)
            , dtype=dtype, device=device
        )
        
        # Reservoir weights
        G = nx.erdos_renyi_graph(self.Nres, self.p, directed=True, seed=self.random_state)
        
        # Add self-loops with probability self.p
        for node in G.nodes():
            if rng.rand() < self.p:
                G.add_edge(node, node)
        
        Wres = nx.to_numpy_array(G).astype(np.float32)
        Wres[Wres != 0] = rng.uniform(-1, 1, size=np.count_nonzero(Wres))
        
        # Scale weights to achieve desired spectral radius
        eigvals = np.linalg.eigvals(Wres)
        max_eig = np.max(np.abs(eigvals)) if eigvals.size > 0 else 0.0
        if max_eig > 0:
            Wres *= self.rho / max_eig
        
        self.Wres = torch.from_numpy(Wres).to(device=device, dtype=dtype)
        self.x = torch.zeros(self.Nres, dtype=dtype, device=device)
    
    def _run_reservoir(self, u, collect_states=True):
        if not torch.is_tensor(u):
            u = torch.from_numpy(u).to(device=device, dtype=dtype)
        
        T = u.shape[0]
        x = self.x.clone()
        states = []

        with torch.no_grad():
            # Compute new state: (1-α) * x(t) + α * tanh(W_res * x(t) + W_in * u(t))
            for t in range(T):
                x = (1.0 - self.alpha) * x + self.alpha * torch.tanh(self.Wres @ x + self.Win * u[t])
                if collect_states:
                    states.append(x.clone().unsqueeze(1))
            self.x = x
            
        if collect_states:
            if len(states) == 0:
                return torch.empty((self.Nres, 0), device=device)
            # return reservoir state matrix
            return torch.cat(states, dim=1)
        else:
            return None
    
    def train(self, u, y, transient_steps=40000, normalize=True):
        u = np.asarray(u).astype(np.float32)
        y = np.asarray(y).astype(np.float32)
        assert u.shape[0] == y.shape[0], "u and y have to have the same length"
        assert 0 <= transient_steps < u.shape[0], "transient_steps must not be longer than training input"
        
        # compute z-Transform
        if normalize:
            mu_u = u[transient_steps:].mean()
            sigma_u = u[transient_steps:].std() if u[transient_steps:].std() > 0 else 1.0
            mu_y = y[transient_steps:].mean()
            sigma_y = y[transient_steps:].std() if y[transient_steps:].std() > 0 else 1.0
            u = (u - mu_u) / sigma_u
            y = (y - mu_y) / sigma_y
            self._norm_params = [mu_u, sigma_u]
        else:
            self._norm_params = None
        
        self.x = torch.zeros(self.Nres, dtype=dtype, device=device)
        if transient_steps > 0: # update reservoir state for transient_steps inputs
            self._run_reservoir(u[:transient_steps], collect_states=False)
        
        # compute reservoir state matrix for input
        X_res = self._run_reservoir(u[transient_steps:])
        y_train = torch.from_numpy(y[transient_steps:]).to(dtype=dtype, device=device)
        
        # Closed-form Ridge regression (Tikhonov) on GPU
        with torch.no_grad():
            I = torch.eye(self.Nres, dtype=dtype, device=device)
            A = X_res @ X_res.T + self.lambda_reg * I
            B = X_res @ y_train
            self.Wout = torch.linalg.solve(A, B)
            
            # Predict on training set
            y_pred = X_res.T @ self.Wout
            nrmse = torch.sqrt(torch.mean((y_pred - y_train)**2)) / torch.std(y_train)
        
        return float(nrmse.cpu().numpy())
    
    def predict(self, u_warmup, n_steps):
        u_warmup = np.asarray(u_warmup).astype(np.float32)
        assert u_warmup.shape[0] > 0, "At least one new input necessary"
        
        # if the data was normalized in training, apply z-Transform here as well
        if self._norm_params is not None:
            mu_u, sigma_u = self._norm_params
            u_warmup = (u_warmup - mu_u) / sigma_u
        
        # update reservoir state for warmup input
        self.x = torch.zeros(self.Nres, dtype=dtype, device=device)
        u_warmup = torch.from_numpy(u_warmup).to(device=device, dtype=dtype)
        self._run_reservoir(u_warmup[:-1], collect_states=False)
        
        predictions = []
        with torch.no_grad():
            u = u_warmup[-1].clone()    
            for _ in range(n_steps):
                # Compute reservoir state based on previous output
                x = (1.0 - self.alpha) * self.x + self.alpha * torch.tanh(self.Wres @ self.x + self.Win * u)
                u = torch.dot(self.Wout, x)
                predictions.append(u.cpu().item())
                self.x = x
        
        return np.array(predictions).astype(np.float32)
    
    def train_and_test(self, u, y, train_fraction=0.5, transient_steps=40000, warmup_steps=0, normalize=True):
        u = np.asarray(u).astype(np.float32)
        y = np.asarray(y).astype(np.float32)
        
        # calculate steps for different phases
        N = u.shape[0]
        N_train = int(N * train_fraction)
        N_test = N - N_train
        assert N_train > transient_steps, "transient_steps must be smaller than training steps"
        assert N_test > warmup_steps, "warmup_steps must be smaller than test steps"
        
        u_train = u[:N_train]
        u_warmup = u[N_train : N_train + warmup_steps]
        y_train = y[:N_train]
        y_test = y[N_train + warmup_steps :]
        
        # compute NRMSE values for test and predict phase
        nrmse_train = self.train(u_train, y_train, transient_steps, normalize)
        y_pred = self.predict(u_warmup, N_test - warmup_steps)
        nrmse_test = np.sqrt(np.mean((y_pred - y_test)**2)) / np.std(y_test)
        
        return nrmse_train, nrmse_test


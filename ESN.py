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
        # W_in = rng.uniform(-0.5, 0.5, size=(self.Nres,)).astype(np.float32)
        # Bias added
        W_in = rng.uniform(-0.5, 0.5, size=(self.Nres, 2)).astype(np.float32)
        self.Win = torch.from_numpy(W_in).to(device=device, dtype=dtype)
        
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
        x = self.x
        N = self.Nres
        
        states = torch.empty((N, T), device=device, dtype=dtype) if collect_states else None
        
        with torch.no_grad():
            Wres = self.Wres
            Win = self.Win
            alpha = float(self.alpha)
            one_minus_alpha = 1.0 - alpha
            
            # Compute new state: (1-α) * x(t) + α * tanh(W_res * x(t) + W_in * u(t))
            for t in range(T):
                # pre_activation = Wres.matmul(x)
                # pre_activation += Win * u[t]
                # Bias added
                u_with_bias = torch.tensor([1.0, u[t]], dtype=dtype, device=device)  # bias + scalar input
                pre_activation = Wres.matmul(x) + Win.matmul(u_with_bias)
                pre_activation.tanh_()
                x = one_minus_alpha * x + alpha * pre_activation
                if collect_states:
                    states[:, t] = x
            
            self.x = x
            
        return states if collect_states else None
    
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
            self._norm_params = [mu_u, sigma_u, mu_y, sigma_y]
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
            A = X_res @ X_res.T
            A.diagonal().add_(self.lambda_reg)
            B = X_res @ y_train
            L = torch.linalg.cholesky(A)
            # solve LL^T Wout = B
            self.Wout = torch.cholesky_solve(B.unsqueeze(1), L).squeeze(1)
            
            # Predict on training set
            y_pred = X_res.T @ self.Wout
            nrmse = torch.sqrt(torch.mean((y_pred - y_train)**2)) / torch.std(y_train)
        
        return float(nrmse.cpu().numpy())
    
    def predict(self, u_warmup, n_steps):
        u_warmup = np.asarray(u_warmup).astype(np.float32)
        assert u_warmup.shape[0] > 0, "At least one new input necessary"
        
        # if the data was normalized in training, apply z-Transform here as well
        if self._norm_params is not None:
            mu_u, sigma_u, _, _ = self._norm_params
            u_warmup = (u_warmup - mu_u) / sigma_u
        
        # update reservoir state for warmup input
        #self.x = torch.zeros(self.Nres, dtype=dtype, device=device)
        u_warmup = torch.from_numpy(u_warmup).to(device=device, dtype=dtype)
        self._run_reservoir(u_warmup[:-1], collect_states=False)
        
        predictions = []
        with torch.no_grad():
            u = u_warmup[-1]
            for _ in range(n_steps):
                # Compute reservoir state based on previous output
                # x = (1.0 - self.alpha) * self.x + self.alpha * torch.tanh(self.Wres @ self.x + self.Win * u)
                # Bias added
                u_with_bias = torch.tensor([1.0, u], dtype=dtype, device=device)
                x = (1.0 - self.alpha) * self.x + self.alpha * torch.tanh(self.Wres @ self.x + self.Win @ u_with_bias)

                u = torch.dot(self.Wout, x)
                predictions.append(u.cpu().item())
                self.x = x
        
        predictions = np.array(predictions).astype(np.float32)
        
        if self._norm_params is not None:
            _, _, mu_y, sigma_y = self._norm_params
            predictions = predictions * sigma_y + mu_y
        
        return predictions
    
    def train_and_test(self, u, y, train_fraction=0.5, transient_steps=40000, warmup_steps=0, normalize=True):
        u = np.asarray(u).astype(np.float32)
        y = np.asarray(y).astype(np.float32)
        
        # calculate steps for different phases
        N = u.shape[0]
        N_train = int(N * train_fraction)
        N_test = N - N_train
        assert N_train > transient_steps, "transient_steps must be smaller than training steps"
        assert N_test > warmup_steps, "warmup_steps must be smaller than test steps"
        
        u_train = u[:N_train-1]
        u_warmup = u[N_train : N_train + warmup_steps]
        y_train = y[1:N_train]
        y_test = y[N_train + warmup_steps :]
        
        # compute NRMSE values for test and predict phase
        nrmse_train = self.train(u_train, y_train, transient_steps, normalize)
        y_pred = self.predict(u_warmup, N_test - warmup_steps)
        
        nrmse_test = np.sqrt(np.mean((y_pred - y_test)**2)) / np.std(y_test)
        
        return nrmse_train, nrmse_test




class ESNBatch:
    def __init__(self, batch_size=1, Nres=300, p=0.75, alpha=0.5, rho=0.85, lambda_reg=1e-6, random_state=None):
        self.B = int(batch_size)
        self.Nres = Nres
        self.p = self._expand(p)
        self.alpha = torch.tensor(self._expand(alpha), device=device, dtype=dtype).unsqueeze(1)
        self.rho = self._expand(rho)
        self.lambda_reg = self._expand(lambda_reg)
        self.random_state = random_state
        self.Win = None     # (B, N)
        self.Wres = None    # (B, N, N)
        self.Wout = None    # (B, N)
        self.x = None       # (B, N)
        self._norm_params = None
        self._init_weights()
    
    def _expand(self, x):
        if np.isscalar(x):
            return [x] * self.B
        elif len(x) == self.B:
            return list(x)
        raise ValueError("Hyperparam must be scalar or list with length B")
    
    def _init_weights(self):
        B, N = self.B, self.Nres
        Wins = np.zeros((B, N), dtype=np.float32)
        Wress = np.zeros((B, N, N), dtype=np.float32)
        rng = np.random.RandomState(self.random_state)
        
        for b in range(B):
            Wins[b] = rng.uniform(-0.5, 0.5, size=(N,)).astype(np.float32)
            
            G = nx.erdos_renyi_graph(N, self.p[b], directed=True, seed=self.random_state)
            for node in G.nodes():
                if rng.rand() < self.p[b]:
                    G.add_edge(node, node)
            Wmat = nx.to_numpy_array(G).astype(np.float32)
            Wmat[Wmat != 0] = rng.uniform(-1.0, 1.0, size=np.count_nonzero(Wmat))
            
            eigvals = np.linalg.eigvals(Wmat)
            max_eig = np.max(np.abs(eigvals)) if eigvals.size > 0 else 0.0
            if max_eig > 0:
                Wmat *= (self.rho[b] / max_eig)
            Wress[b] = Wmat
        
        self.Win = torch.from_numpy(Wins).to(device=device, dtype=dtype)
        self.Wres = torch.from_numpy(Wress).to(device=device, dtype=dtype)
        self.x = torch.zeros((B, N), dtype=dtype, device=device)
        self.Wout = torch.zeros((B, N), device=device, dtype=dtype)
    
    def _run_reservoir(self, u, collect_states=True):
        if not torch.is_tensor(u):
            u = torch.from_numpy(np.asarray(u, dtype=np.float32))
        u = u.to(device=device, dtype=dtype)
        
        if u.dim() == 1:
            u = u.unsqueeze(0).expand(self.B, -1) # (B, T)
        else:
            assert u.shape[0] == self.B, "Input has wrong batch dimension"
        
        B, N = self.B, self.Nres
        T = u.shape[1]
        x = self.x.clone() # (B, N)
        
        states = torch.empty((B, N, T), device=device, dtype=dtype) if collect_states else None
        
        with torch.no_grad():
            Wres = self.Wres    # (B, N, N)
            Win = self.Win      # (B, N)
            alpha = self.alpha  # (B, 1)
            one_minus_alpha = 1.0 - alpha
            
            # Compute new state: (1-α) * x(t) + α * tanh(W_res * x(t) + W_in * u(t))
            for t in range(T):
                x_t = x.unsqueeze(2)                    # (B, N, 1)
                Wx = torch.bmm(Wres, x_t).squeeze(2)   # (B, N)
                pre = Wx + Win * u[:, t].unsqueeze(1)   # (B, N)
                pre.tanh_()
                x = one_minus_alpha * x + alpha * pre   # (B, N)
                if collect_states:
                    states[:, :, t] = x
            
            self.x = x
            
        return states if collect_states else None
    
    def train(self, u, y, transient_steps=40000, normalize=True):
        if not torch.is_tensor(u):
            u = torch.from_numpy(np.asarray(u, dtype=np.float32))
        if not torch.is_tensor(y):
            y = torch.from_numpy(np.asarray(y, dtype=np.float32))
        
        # send to device
        u = u.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype)
        
        # correct shape if necessary
        if u.dim() == 1:
            u = u.unsqueeze(0).expand(self.B, -1)
        if y.dim() == 1:
            y = y.unsqueeze(0).expand(self.B, -1)
        
        B, T = u.shape
        assert B == self.B, "Wrong batch size"
        assert u.shape == y.shape, "u and y have to have the same length"
        assert 0 <= transient_steps < T, "transient_steps must not be longer than training input"
        
        # compute z-Transform
        if normalize:
            u_train = u[:, transient_steps:]  # (B, T_train)
            y_train = y[:, transient_steps:]
            mu_u = u_train.mean(dim=1, keepdim=True)   # (B,1)
            sigma_u = u_train.std(dim=1, keepdim=True)
            sigma_u[sigma_u == 0] = 1.0
            mu_y = y_train.mean(dim=1, keepdim=True)
            sigma_y = y_train.std(dim=1, keepdim=True)
            sigma_y[sigma_y == 0] = 1.0
            
            u = (u - mu_u) / sigma_u
            y = (y - mu_y) / sigma_y
            self._norm_params = [mu_u.cpu().numpy(), sigma_u.cpu().numpy(), mu_y.cpu().numpy(), sigma_y.cpu().numpy()]
        else:
            self._norm_params = None
        
        # reset reservoir state
        self.x = torch.zeros((B, self.Nres), dtype=dtype, device=device)
        
        # run transient steps without collecting states
        if transient_steps > 0:
            self._run_reservoir(u[:, :transient_steps], collect_states=False)
        
        # compute reservoir state matrix for input
        X_res = self._run_reservoir(u[:, transient_steps:])
        y_train = y[:, transient_steps:]
        
        with torch.no_grad():
            # Closed-form Ridge regression (Tikhonov) on GPU
            X_res_T = X_res.transpose(1, 2)             # (B, T, N)
            A = torch.bmm(X_res, X_res_T)               # (B, N, N)
            for b in range(B):
                A[b].diagonal().add_(self.lambda_reg[b])
            y_vec = y_train.unsqueeze(2)                # (B, T, 1)
            B = torch.bmm(X_res, y_vec)                 # (B, N, 1)
            Wout = torch.linalg.solve(A, B).squeeze(2)  # (B, N)
            self.Wout = Wout
            
            # Predict on training set
            y_pred = torch.bmm(X_res_T, Wout.unsqueeze(2)).squeeze(2)       # (B, T)
            mse = torch.mean((y_pred - y_train)**2, dim=1)
            denom = torch.std(y_train, dim=1)
            denom[denom == 0.0] = 1.0
            nrmse = torch.sqrt(mse) / denom
        
        return nrmse.cpu().numpy()
    
    def predict(self, u_warmup, n_steps):
        if not torch.is_tensor(u_warmup):
            u_warmup = torch.from_numpy(np.asarray(u_warmup, dtype=np.float32))
        u_warmup = u_warmup.to(device=device, dtype=dtype)
        
        if u_warmup.dim() == 1:
            u_warmup = u_warmup.unsqueeze(0).expand(self.B, -1)
        
        # normalize here if training was done with normalization
        if self._norm_params is not None:
            mu_u, sigma_u, _, _ = self._norm_params
            mu_u = torch.from_numpy(mu_u).to(device=device, dtype=dtype).squeeze(1)   # (B,)
            sigma_u = torch.from_numpy(sigma_u).to(device=device, dtype=dtype).squeeze(1)
            u_warmup = (u_warmup - mu_u.unsqueeze(1)) / sigma_u.unsqueeze(1)
        
        # update reservoir state for warmup input
        self.x = torch.zeros((self.B, self.Nres), dtype=dtype, device=device)
        if u_warmup.shape[1] > 1:
            self._run_reservoir_batch(u_warmup[:, :-1], collect_states=False)
        
        predictions = []
        with torch.no_grad():
            u = u_warmup[:, -1]
            Wres = self.Wres    # (B, N, N)
            Win = self.Win      # (B, N)
            Wout = self.Wout
            alpha = self.alpha  # (B, 1)
            one_minus_alpha = 1.0 - alpha
            
            for _ in range(n_steps):
                # Compute reservoir state based on previous output
                x_t = self.x.unsqueeze(2)              # (B, N, 1)
                Wx =torch.bmm(Wres, x_t).squeeze(2)    # (B, N)
                pre = Wx + Win * u.unsqueeze(1)        # (B, N)
                pre.tanh_()
                x = one_minus_alpha * x + alpha * pre
                u = torch.sum(Wout * x, dim=1)
                predictions.append(u.cpu().numpy())
                self.x = x
        
        predictions = np.stack(predictions, axis=1).astype(np.float32)
        
        if self._norm_params is not None:
            _, _, mu_y, sigma_y = self._norm_params
            predictions = predictions * sigma_y + mu_y
        
        return predictions
    
    def train_and_test(self, u, y, train_fraction=0.5, transient_steps=40000, warmup_steps=0, normalize=True):
        u = np.asarray(u).astype(np.float32)
        y = np.asarray(y).astype(np.float32)
        
        # calculate steps for different phases
        N = u.shape[0]
        N_train = int(N * train_fraction)
        N_test = N - N_train
        assert N_train > transient_steps, "transient_steps must be smaller than training steps"
        assert N_test > warmup_steps, "warmup_steps must be smaller than test steps"
        
        u_train = u[:N_train-1]
        u_warmup = u[N_train : N_train + warmup_steps]
        y_train = y[1:N_train]
        y_test = y[N_train + warmup_steps :]
        
        # compute NRMSE values for test and predict phase
        nrmse_train = self.train(u_train, y_train, transient_steps, normalize)
        y_pred = self.predict(u_warmup, N_test - warmup_steps)
        
        mse_test = torch.mean((y_pred - y_test)**2, dim=1)
        denom = torch.std(y_test, dim=1)
        denom[denom == 0.0] = 1.0
        nrmse_test = torch.sqrt(mse_test) / denom
        
        return nrmse_train, nrmse_test




class ESNNumpy:
    def __init__(self, Nres=300, p=0.75, alpha=0.5, rho=0.85, lambda_reg=1e-4, random_state=None):
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
        # self.Win = rng.uniform(-0.5, 0.5, size=(self.Nres,)).astype(np.float32)

        # Added bias: First column = bias, second column = input weight.
        self.Win = rng.uniform(-0.5, 0.5, size=(self.Nres,2)).astype(np.float32)

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
        
        self.Wres = Wres.astype(np.float32)
        self.x = np.zeros(self.Nres, dtype=np.float32)
    
    def _run_reservoir(self, u, collect_states=True):        
        T = u.shape[0]
        x = self.x
        N = self.Nres
        
        states = np.zeros((N, T), dtype=np.float32) if collect_states else None
        
        Wres = self.Wres
        Win = self.Win
        alpha = float(self.alpha)
        one_minus_alpha = 1.0 - alpha
        
        # Compute new state: (1-α) * x(t) + α * tanh(W_res * x(t) + W_in * u(t))
        for t in range(T):
            # Bias added
            u_with_bias = np.array([1.0, u[t]], dtype=np.float32)  # [bias, input]
            pre_activation = Wres @ x
            # pre_activation += Win * u[t]
            pre_activation += Win @ u_with_bias
            post_activation = np.tanh(pre_activation)
            x = one_minus_alpha * x + alpha * post_activation
            if collect_states:
                states[:, t] = x
        
        self.x = x
        return states if collect_states else None
    
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
            self._norm_params = [mu_u, sigma_u, mu_y, sigma_y]
        else:
            self._norm_params = None
        
        self.x = np.zeros(self.Nres, dtype=np.float32)
        if transient_steps > 0: # update reservoir state for transient_steps inputs
            self._run_reservoir(u[:transient_steps], collect_states=False)
        
        # compute reservoir state matrix for input
        X_res = self._run_reservoir(u[transient_steps:])
        y_train = y[transient_steps:]
        
        # Closed-form Ridge regression (Tikhonov)
        A = X_res @ X_res.T
        A += self.lambda_reg * np.eye(self.Nres)
        B = X_res @ y_train
        self.Wout = np.linalg.solve(A, B)
        
        # Predict on training set
        y_pred = X_res.T @ self.Wout
        nrmse = np.sqrt(np.mean((y_pred - y_train)**2)) / np.std(y_train)
        return nrmse
    
    def predict(self, u_warmup, n_steps):
        u_warmup = np.asarray(u_warmup).astype(np.float32)
        assert u_warmup.shape[0] > 0, "At least one new input necessary"
        
        # if the data was normalized in training, apply z-Transform here as well
        if self._norm_params is not None:
            mu_u, sigma_u, _, _ = self._norm_params
            u_warmup = (u_warmup - mu_u) / sigma_u
        
        # update reservoir state for warmup input
        self._run_reservoir(u_warmup[:-1], collect_states=False)
        
        predictions = []
        u = u_warmup[-1]
        x = self.x
        Wres = self.Wres
        Win = self.Win
        Wout = self.Wout
        alpha = float(self.alpha)
        one_minus_alpha = 1.0 - alpha
        
        for _ in range(n_steps):
            # Compute reservoir state based on previous output
            # Bias added
            u_with_bias = np.array([1.0, u], dtype=np.float32)
            pre_activation = Wres @ x + Win @ u_with_bias
            # pre_activation = Wres @ x + Win * u
            post_activation = np.tanh(pre_activation)
            x = one_minus_alpha * x + alpha * post_activation
            u = Wout @ x
            predictions.append(u)
        
        self.x = x
        predictions = np.array(predictions).astype(np.float32)
        
        if self._norm_params is not None:
            _, _, mu_y, sigma_y = self._norm_params
            predictions = predictions * sigma_y + mu_y
        
        return predictions
    
    def train_and_test(self, u, y, train_fraction=0.5, transient_steps=40000, warmup_steps=0, normalize=True):
        u = np.asarray(u).astype(np.float32)
        y = np.asarray(y).astype(np.float32)
        
        # calculate steps for different phases
        N = u.shape[0]
        N_train = int(N * train_fraction)
        N_test = N - N_train
        assert N_train > transient_steps, "transient_steps must be smaller than training steps"
        #assert N_test > warmup_steps, "warmup_steps must be smaller than test steps"
        
        u_train = u[:N_train-1]
        u_warmup = u[:warmup_steps] #u[N_train : N_train + warmup_steps]
        y_train = y[1:N_train]
        y_test = y[warmup_steps:] #y[N_train + warmup_steps :]
        
        # compute NRMSE values for test and predict phase
        nrmse_train = self.train(u_train, y_train, transient_steps, normalize)
        y_pred = self.predict(u_warmup, len(u) - warmup_steps)  #self.predict(u_warmup, N_test - warmup_steps)
        
        nrmse_test = np.sqrt(np.mean((y_pred - y_test)**2)) / np.std(y_test)
        
        return nrmse_train, nrmse_test
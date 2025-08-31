import numpy as np
from sklearn.linear_model import Ridge

class EchoStateNetwork:
    def __init__(self, N_res, p, alpha, rho):
        self.N_res = N_res
        self.p = p
        self.alpha = alpha
        self.rho = rho
        self.Win = np.random.uniform(-0.5, 0.5, (N_res, 1))
        self.Wres = self.initialize_reservoir(N_res, p, rho)
        self.Wout = None

    def initialize_reservoir(self, N_res, p, rho):
        W = np.random.uniform(-1, 1, (N_res, N_res))
        mask = np.random.rand(N_res, N_res) < p
        W[~mask] = 0
        spectral_radius = np.max(np.abs(np.linalg.eigvals(W)))
        if spectral_radius > 0:
            W *= rho / spectral_radius
        else:
            print("Warning: spectral radius too small")
        return W

    def train(self, u, y_target):
        states = np.zeros((len(u), self.N_res))
        state = np.zeros(self.N_res)
        for t in range(1, len(u)):
            state = (1 - self.alpha) * state + self.alpha * np.tanh(
                np.dot(self.Wres, state) + self.Win.flatten() * u[t]
            )
            states[t] = state
        reg = Ridge(alpha=1e-6)
        reg.fit(states, y_target)
        self.Wout = reg.coef_

    def predict(self, u):
        states = np.zeros((len(u), self.N_res))
        state = np.zeros(self.N_res)
        y_pred = np.zeros(len(u))
        for t in range(1, len(u)):
            state = (1 - self.alpha) * state + self.alpha * np.tanh(
                np.dot(self.Wres, state) + self.Win.flatten() * u[t]
            )
            states[t] = state
            y_pred[t] = np.dot(self.Wout, state)
        return y_pred
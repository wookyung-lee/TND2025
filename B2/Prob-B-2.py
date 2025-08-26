import numpy as np
import matplotlib.pyplot as plt
from ESN import ESNNumpy
import pickle
from tqdm import tqdm
import networkx as nx
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


seed = 42
np.random.seed(seed)
train_fraction = 0.5
dt = 0.005
Tt = 40000


class EchoStateNetwork:
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
    
    def train(self, u, y, warmup=40000):
        states = []
        
        for ut in u:
            r = self.update_reservoir(ut)
            states.append(r.flatten())
        
        states = np.array(states)
        
        states = states[warmup:]
        y = y[warmup:]
        
        self.ridge.fit(states, y)
        self.Wout = self.ridge.coef_
    
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


def load_data():
    with open("timeseries_data.pkl", "rb") as f:
        all_data = pickle.load(f)
        return all_data


def load_params():
    with open("optimized_params_rho.pkl", "rb") as f:
        optimized_params = pickle.load(f)
        return optimized_params


def main():
    all_data = load_data()
    #optimized_params = load_params()
    
    warmup_times = {}
    
    for label in tqdm(['a', 'b', 'c', 'e'], desc="Subtasks"):
        data = np.asarray(all_data[label]['x']).flatten()
        
        warmup_points = np.linspace(40001, 300000, 10, dtype=int)
        
        nrmse_pred_list = []
        for Nwarmup in tqdm(warmup_points, desc="Warmup points"):
            esn = EchoStateNetwork(Nres=875, p=0.8, alpha=0.5, rho=0.9)
            
            scaler = StandardScaler()
            data_normalized = scaler.fit_transform(data.reshape(-1, 1)).flatten()
            
            N_train = int(train_fraction * (len(data) - Nwarmup)) + Nwarmup
            
            inputs = data_normalized[:-1]
            targets = data_normalized[1:]
            if N_train >= len(inputs):
                # not enough data left for prediction; mark as nan and continue
                nrmse_pred_list.append(np.nan)
                continue
            
            u_train = inputs[:N_train]
            y_train = targets[:N_train]
            u_predict = inputs[N_train:]
            y_predict = targets[N_train:]
            
            if len(u_train) <= Nwarmup:
                nrmse_pred_list.append(np.nan)
                continue
            
            esn.train(u_train, y_train, warmup=Nwarmup)
            predictions_normalized, _ = esn.predict(u_predict)
            
            predictions = scaler.inverse_transform(predictions_normalized)
            
            true_values = scaler.inverse_transform(y_predict)
            nrmse = esn.calculate_nrmse(true_values, predictions)
            
            nrmse_pred_list.append(nrmse)
        
        nrmse_pred_list = np.asarray(nrmse_pred_list)
        
        T1_idx = int(np.argmax(nrmse_pred_list))
        T1 = warmup_points[T1_idx]
        if T1_idx < len(nrmse_pred_list) - 1:
            relative_T2_idx = int(np.argmin(nrmse_pred_list[T1_idx + 1:]))
            T2_idx = T1_idx + 1 + relative_T2_idx
            T2 = warmup_points[T2_idx]
        else:
            T2 = 200000
        
        warmup_times[label] = {"T1": T1, "T2": T2}
        
        # Plot NRMSE prediction vs warm-up time
        plt.figure()
        plt.plot(warmup_points * dt, nrmse_pred_list, marker='o')
        plt.xlabel('Warm-up Time (ms)')
        plt.ylabel('NRMSE in Prediction')
        plt.title(f'NRMSE vs Warm-up Time - Regime {label}')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"./B2/Prob-B-2{label}.png", dpi=600, bbox_inches='tight', pad_inches=0.01)
        #plt.show()
    
    with open("warmup_times.pkl", "wb") as f:
        pickle.dump(warmup_times, f, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    main()
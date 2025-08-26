import numpy as np
import matplotlib.pyplot as plt
from ESN import ESNNumpy
import pickle
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


seed = 42
np.random.seed(seed)
transient_pts = 40000  # Washed-out transient points to discard
dt = 0.005  # Time step used in integration
warmup_points_list = np.linspace(40001, 300000, 20, dtype=int) # Warm-up points to test (adjust upper bound <= data length)
train_fraction = 0.5
test_length = 10000
normalize = True


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
        for t in range(len(u)):
            state = (1 - self.alpha) * state + self.alpha * np.tanh(
                np.dot(self.Wres, state) + self.Win.flatten() * u[t]
            )
            states[t] = state
        reg = Ridge(alpha=1e-6, fit_intercept=True)
        reg.fit(states, y_target)
        self.Wout = reg.coef_.ravel()
        return states

    def predict(self, u, initial_state=None):
        states = np.zeros((len(u), self.N_res))
        if initial_state is not None:
            state = initial_state.copy()
        else:
            state = np.zeros(self.N_res)
        y_pred = np.zeros(len(u))
        for t in range(len(u)):
            state = (1 - self.alpha) * state + self.alpha * np.tanh(
                np.dot(self.Wres, state) + self.Win.flatten() * u[t]
            )
            states[t] = state
            y_pred[t] = np.dot(self.Wout, state)
        return y_pred


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
    optimized_params = load_params()
    
    warmup_times = {}
    
    for label in tqdm(['a', 'b', 'c', 'e'], desc="Subtasks"):
        data = np.asarray(all_data[label]['x']).flatten()
        #params = optimized_params[label]['best_params']
        
        # Ensure warmup points do not exceed training data size
        max_warmup = len(data) - test_length - 1
        warmup_points = warmup_points_list[warmup_points_list < max_warmup]
        
        nrmse_pred_list = []
        for Nwarmup in tqdm(warmup_points, desc="Warmup points"):
            if Nwarmup <= transient_pts:
                nrmse_pred_list.append(np.nan)
                continue
            
            #esn = ESNNumpy(Nres=params['Nres'], p=params['p'], alpha=params['alpha'], rho=params['rho'], random_state=seed)
            #_, nrmse_pred = esn.train_and_test(
            #    u=data[:-1], 
            #    y=data[1:], 
            #    train_fraction=train_fraction, 
            #    transient_steps=transient_pts, 
            #    warmup_steps=Nwarmup, 
            #    normalize=normalize
            #)
            
            esn = EchoStateNetwork(N_res=875, p=0.8, alpha=0.5, rho=0.9)
            
            u_train = data[transient_pts:Nwarmup]
            if u_train.size < 2:
                nrmse_pred_list.append(np.nan)
                continue
            
            u_test = data[Nwarmup:Nwarmup + test_length]
            if u_test.size < 2:
                nrmse_pred_list.append(np.nan)
                continue
            
            scaler = StandardScaler()
            u_train_reshaped = u_train.reshape(-1, 1)
            scaler.fit(u_train_reshaped)
            
            u_train_norm = scaler.transform(u_train_reshaped).flatten()
            u_train_input = u_train_norm[:-1]
            y_train_target = u_train_norm[1:]
            
            if len(u_train_input) == 0:
                nrmse_pred_list.append(np.nan)
                continue
            
            states = esn.train(u_train_input, y_train_target)
            final_train_state = states[-1]
            
            u_test_norm = scaler.transform(u_test.reshape(-1, 1)).flatten()
            u_test_input = u_test_norm[:-1]
            y_test_target = u_test_norm[1:]
            
            pred = esn.predict(u_test_input, initial_state=final_train_state)
            
            y_pred = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
            y_true = scaler.inverse_transform(y_test_target.reshape(-1, 1)).flatten()
            
            mse = np.mean((y_true - y_pred) ** 2)
            rmse = np.sqrt(mse)
            
            denom = np.std(y_true)
            if denom == 0:
                nrmse = np.nan
            else:
                nrmse = rmse / denom
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
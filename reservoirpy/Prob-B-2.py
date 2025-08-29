import numpy as np
import matplotlib.pyplot as plt
import pickle
import networkx as nx
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from reservoirpy.nodes import Reservoir, Ridge
from sklearn.metrics import mean_squared_error
from copy import deepcopy

seed = 42
np.random.seed(seed)
train_fraction = 0.5
dt = 0.005
Tt = 40000


def load_data():
    with open("../timeseries_data.pkl", "rb") as f:
        all_data = pickle.load(f)
        return all_data


def load_params():
    with open("optimized_params_rho.pkl", "rb") as f:
        optimized_params = pickle.load(f)
        return optimized_params


def create_esn(Nres, p, alpha, rho):
    Win = np.random.uniform(-0.5, 0.5, (Nres, 1))
    
    graph = nx.erdos_renyi_graph(Nres, p, directed=True)
    for node in graph.nodes():
        if np.random.random() < p:
            graph.add_edge(node, node)
    Wres = nx.to_numpy_array(graph)
    Wres = np.where(Wres > 0, np.random.uniform(-1, 1, Wres.shape), Wres)
    eigenvalues = np.linalg.eigvals(Wres)
    spectral_radius = np.max(np.abs(eigenvalues))
    Wres = (Wres / spectral_radius) * rho
    
    reservoir = Reservoir(Win=Win, W=Wres, bias=np.zeros(Nres), lr=alpha, input_dim=1, seed=seed)
    readout = Ridge(ridge=1e-6, fit_bias=False)
    esn = reservoir >> readout
    
    return esn


def predict_autonomous(esn, warmup, n_steps):
    esn.nodes[0].reset()
    warumup_preds = esn.run(warmup)
    
    predictions = []
    ut = warumup_preds[-1]
    
    for _ in range(n_steps):
        pred = esn(ut)
        predictions.append(pred)
        ut = pred
    
    return np.array(predictions)


def main():
    all_data = load_data()
    #optimized_params = load_params()
    
    optimized_params = {
        "a" : {'Nres':922, 'p':0.9, 'alpha':0.7, 'rho':1.1888889},
        "b" : {'Nres':1000, 'p':0.5, 'alpha':0.7, 'rho':1.1888889}, 
        "c" : {'Nres':844, 'p':0.4, 'alpha':0.7, 'rho':1.0333333},
        "e" : {'Nres':1000, 'p':0.5, 'alpha':0.5, 'rho':1.1888889} 
    }
    
    warmup_times = {}
    
    for label in tqdm(['a', 'b', 'c', 'e'], desc="Subtasks"):
        params = optimized_params[label]
        esn = create_esn(Nres=params['Nres'], p=params['p'], alpha=params['alpha'], rho=params['rho'])
        
        data = np.asarray(all_data[label]['x']).flatten()
        data = data[Tt:]
        N_train = int(train_fraction * len(data))
        
        scaler = StandardScaler()
        data_normalized = scaler.fit_transform(data.reshape(-1, 1)).flatten()
        
        inputs = data_normalized[:-1]
        targets = data_normalized[1:]
        
        u_train = inputs[:N_train].reshape(-1, 1)
        y_train = targets[:N_train].reshape(-1, 1)
        
        esn = esn.fit(u_train, y_train, warmup=0)
        
        nrmse_pred_list = []
        warmup_points = np.linspace(1, len(inputs) - 2, 10, dtype=int)
        for N_warmup in tqdm(warmup_points, desc="Warmup points"):
            esn_ = deepcopy(esn)
            
            u_warmup = inputs[:N_warmup].reshape(-1, 1)
            u_predict = inputs[N_warmup:].reshape(-1, 1)
            y_predict = targets[N_warmup:].reshape(-1, 1)
            
            predictions_normalized = predict_autonomous(esn, u_warmup, len(y_predict))
            
            predictions = scaler.inverse_transform(predictions_normalized)
            true_values = scaler.inverse_transform(y_predict)
            
            mse = mean_squared_error(true_values, predictions)
            nrmse = np.sqrt(mse) / np.std(true_values)
            nrmse_pred_list.append(nrmse)
        
        nrmse_pred_list = np.asarray(nrmse_pred_list)
        
        T1_idx = int(np.argmax(nrmse_pred_list))
        T1 = warmup_points[T1_idx]
        T2_idx = int(np.argmin(nrmse_pred_list))
        T2 = warmup_points[T2_idx]
        
        warmup_times[label] = {"T1": T1, "T2": T2}
        
        # Plot NRMSE prediction vs warm-up time
        plt.figure()
        plt.plot((warmup_points + Tt) * dt, nrmse_pred_list, marker='o')
        plt.xlabel('Warm-up Time (ms)')
        plt.ylabel('NRMSE in Prediction')
        plt.title(f'NRMSE vs Warm-up Time - Regime {label}')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"./B2_Plots/Prob-B-2{label}.png", dpi=600, bbox_inches='tight')
        #plt.show()
    
    with open("./reservoirpy/warmup_times.pkl", "wb") as f:
        pickle.dump(warmup_times, f, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    main()
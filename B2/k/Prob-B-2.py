import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from ESN import ESNNumpy
import pickle
from tqdm import tqdm



seed = 42
transient_pts = 40000  # Washed-out transient points to discard
dt = 0.005  # Time step used in integration
warmup_points_list = np.linspace(40001, 130000, 20, dtype=int) -1  # Warm-up points to test (adjust upper bound <= data length)
train_fraction = 0.5
normalize = False



def load_data():
    print("Importing data ...")
    data_folder = os.path.abspath('A2')
    sys.path.insert(0, data_folder)
    from A2 import all_data
    print("Data import complete")
    return all_data


def load_params():
    with open("optimized_params.pkl", "rb") as f:
        optimized_params = pickle.load(f)
        return optimized_params


def main():
    all_data = load_data()
    optimized_params = load_params()
    
    warmup_times = {}
    
    for label in tqdm(['a', 'b', 'c', 'e'], desc="Subtasks"):
        data = all_data[label]['x']
        params = optimized_params[label]['best_params']
        
        # Ensure warmup points do not exceed training data size
        warmup_points = warmup_points_list[warmup_points_list < len(data)]
        
        nrmse_pred_list = []
        for Nwarmup in tqdm(warmup_points, desc="Warmup points"):
            esn = ESNNumpy(Nres=params['Nres'], p=params['p'], alpha=params['alpha'], rho=params['rho'], random_state=seed)
            _, nrmse_pred = esn.train_and_test(
                u=data[:-1], 
                y=data[1:], 
                train_fraction=train_fraction, 
                transient_steps=transient_pts, 
                warmup_steps=Nwarmup, 
                normalize=normalize
            )
            nrmse_pred_list.append(nrmse_pred)
        
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
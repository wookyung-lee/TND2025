import sys
import os
import numpy as np
from tqdm import tqdm
from ESN import *
from skopt import gp_minimize
from skopt.space import Integer, Real

seed = 111

parameter_space = [
    Integer(300, 1000, name="Nres"),
    Real(0.0001, 1.0, name="p", prior="log-uniform"),
    Real(0.0001, 1.0, name="alpha", prior="log-uniform"),
    Real(0.0001, 1.5, name="rho", prior="log-uniform")
]

esn = ESN(random_state=seed)

def objective(params, u, y):
    esn.Nres, esn.p, esn.alpha, esn.rho = params
    esn._init_weights()
    nrmse = esn.train(u, y) #, transient_steps=40000, normalize=True)
    return nrmse

def main():
    print("Start importing data...")
    
    # Import dictionary
    data_folder = os.path.abspath('A2')
    sys.path.insert(0, data_folder)
    from A2 import all_data
    
    print("Imported Data")
    
    problems = ["a", "b", "c", "e"]
    results = {}
    
    for label in tqdm(problems):
        data = all_data[label]['x']
        
        # Split 50% for training
        N_train = len(data) // 2
        u_train = data[:N_train-1]
        y_train = data[1:N_train]
        
        result = gp_minimize(
            lambda params: objective(params, u_train, y_train), 
            parameter_space,
            n_calls=50,
            random_state=seed
        )
        
        results[label] = result.x
    print(results)

if __name__ == "__main__":
    main()

"""
Results: [Nres,                      p,              alpha,                   rho]
{
    'a': [1000,                    1.0,                1.0,                0.0001],
    'b': [ 765, 0.00012399193968154343,  0.844824224726829, 0.0056587772905221725], 
    'c': [1000,  0.0025652928118404484, 0.9097443637903118,                0.0001], 
    'e': [ 300,   0.005097932353805412, 0.9844819296403085,                0.0001]
"""
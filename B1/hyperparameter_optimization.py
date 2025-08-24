import sys
import os
import numpy as np
from tqdm import tqdm
from ESN import *
from skopt import gp_minimize
from skopt.space import Integer, Real
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.exceptions import TrialPruned



seed = 42
trials = 80
repeats = 3
jobs = -1  # -1 = adapt to CPU
normalize = True
optimizer = "optuna"
optimizers = ["optuna", "skopt"]


optuna_cache = {}
skopt_cache = {}


def load_data():
    print("Importing data ...")
    data_folder = os.path.abspath('A2')
    sys.path.insert(0, data_folder)
    from A2 import all_data
    print("Data import complete")
    return all_data


def evaluate_esn(Nres, p, alpha, rho, u_train, y_train, seed):
    esn = ESNNumpy(
        Nres=Nres,
        p=p,
        alpha=alpha,
        rho=rho,
        random_state=int(seed)
    )
    
    nrmse = esn.train(u_train, y_train, normalize=normalize)
    return float(nrmse)


def objective_factory(optimizer, u_train, y_train, repeats=repeats):
    def objective_optuna(trial: optuna.trial.Trial):
        Nres = trial.suggest_int("Nres", 300, 1000)
        p = trial.suggest_float("p", 1e-4, 1.0)
        alpha = trial.suggest_float("alpha", 1e-4, 1.0)
        rho = trial.suggest_float("rho", 1e-4, 1.5)
        
        cache_key = (Nres, round(float(p), 12), round(float(alpha), 12), round(float(rho), 12))
        
        if cache_key in optuna_cache:
            return optuna_cache[cache_key]
        
        scores = []
        for r in range(repeats):            
            # vary seed per repeat so reservoirs are different
            iter_seed = seed + 100 * r
            try:
                score = evaluate_esn(Nres, p, alpha, rho, u_train, y_train, iter_seed)
            except Exception as e:
                raise
            
            scores.append(score)
            
            trial.report(float(np.mean(scores)), step=r)
            if trial.should_prune():
                raise TrialPruned()
        
        mean_score = float(np.mean(scores))
        optuna_cache[cache_key] = mean_score
        return mean_score
    
    def objective_skopt(params):
        Nres, p, alpha, rho = params
        scores = []
        
        cache_key = (Nres, round(float(p), 12), round(float(alpha), 12), round(float(rho), 12))
        
        if cache_key in skopt_cache:
            return skopt_cache[cache_key]
        
        for r in range(repeats):
            iter_seed = seed + 100 * r
            try:
                score = evaluate_esn(Nres, p, alpha, rho, u_train, y_train, iter_seed)
            except Exception as e:
                raise
            
            scores.append(score)
        
        mean_score = float(np.mean(scores))
        skopt_cache[cache_key] = mean_score
        return mean_score
    
    if optimizer == "optuna":
        return objective_optuna
    elif optimizer == "skopt":
        return objective_skopt
    raise ValueError()


def optimize_study(label, data, trials=trials):
    data = np.asarray(data, dtype=np.float32)
    N_train = len(data) // 2
    
    u_train = data[:N_train-1]
    y_train = data[1:N_train]
    
    objective = objective_factory("optuna", u_train, y_train, repeats=repeats)
    
    sampler = TPESampler(seed=seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    
    study = optuna.create_study(
        sampler=sampler,
        pruner=pruner,
        direction="minimize",
        study_name=f"esn_{label}",
        load_if_exists=True
    )
    
    study.optimize(objective, n_trials=trials, n_jobs=-1, show_progress_bar=True)
    return study


def run_optuna_optimization():
    all_data = load_data()
    problems = ["a", "b", "c", "e"]
    results = {}
    
    for label in tqdm(problems):
        data = all_data[label]['x']
        study = optimize_study(label, data, trials=trials)
        
        best_params = study.best_params
        best_value = study.best_value
        print(f"\nProblem {label}: Best NRMSE = {best_value:.6f}, params = {best_params}")
        
        results[label] = {"best_params": best_params, "best_value": best_value}
    
    print("Optuna Results: ", results)


def run_skopt_optimization():
    all_data = load_data()
    problems = ["a", "b", "c", "e"]
    results = {}
    
    parameter_space = [
        Integer(300, 1000, name="Nres"),
        Real(1e-4, 1.0, name="p"),
        Real(1e-4, 1.0, name="alpha"),
        Real(1e-4, 1.5, name="rho")
    ]
    
    for label in tqdm(problems):
        data = all_data[label]['x']
        
        N_train = len(data) // 2
        u_train = data[:N_train-1]
        y_train = data[1:N_train]
        
        objective = objective_factory("skopt", u_train, y_train, repeats=repeats)
        
        result = gp_minimize(
            objective,
            parameter_space,
            n_calls=trials,
            random_state=seed
        )
        
        results[label] = result.x
    print(results)


def main():
    if optimizer == optimizers[0]:
        run_optuna_optimization()
    elif optimizer == optimizers[1]:
        run_skopt_optimization()
    raise ValueError()



if __name__ == "__main__":
    main()



"""
Attention: Values are outdated, correct values are calculated at this moment. Will submit later
"""

"""
skopt Results:
{
        [Nres,                      p,              alpha,                   rho]
    'a': [1000,                    1.0,                1.0,                0.0001],
    'b': [ 765, 0.00012399193968154343,  0.844824224726829, 0.0056587772905221725], 
    'c': [1000,  0.0025652928118404484, 0.9097443637903118,                0.0001], 
    'e': [ 300,   0.005097932353805412, 0.9844819296403085,                0.0001]
"""

"""
Optuna Results:
{
    'a': {'best_params': {'Nres': 671, 'p': 0.22666924790500015, 'alpha': 0.7702844943973604, 'rho': 0.282880094893985}, 'best_value': 0.0010156501666642725}, 
    'b': {'best_params': {'Nres': 713, 'p': 0.07292008527632693, 'alpha': 0.5769185573541913, 'rho': 0.2408994896666885}, 'best_value': 0.001097427390050143}, 
    'c': {'best_params': {'Nres': 918, 'p': 0.08256658276829093, 'alpha': 0.7441466451080895, 'rho': 0.06488940881320321}, 'best_value': 0.00037622313539031893}, 
    'e': {'best_params': {'Nres': 516, 'p': 0.24345941368690707, 'alpha': 0.21669078403442343, 'rho': 0.5432493223793498}, 'best_value': 0.0008018459775485098}
}
"""
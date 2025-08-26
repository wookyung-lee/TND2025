import sys
import os
import pickle
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
repeats = 1
jobs = 1  # -1 = adapt to CPU
normalize = False
optimizer = "optuna"
optimizers = ["optuna", "skopt"]

cache = {}

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


def objective_factory(opt_name, u_train, y_train, repeats=repeats, seed=seed, use_cache=True):
    def objective_optuna(trial: optuna.trial.Trial):
        Nres = trial.suggest_int("Nres", 300, 1000)
        p = trial.suggest_float("p", 1e-4, 1.0)
        alpha = trial.suggest_float("alpha", 1e-4, 1.0)
        rho = trial.suggest_float("rho", 1e-4, 1.5)
        
        cache_key = (int(Nres), round(float(p), 12), round(float(alpha), 12), round(float(rho), 12))
        if use_cache and cache_key in cache:
            return cache[cache_key]
        
        scores = []
        for r in range(repeats):
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
        if use_cache:
            cache[cache_key] = mean_score
        return mean_score
    
    def objective_skopt(params):
        Nres, p, alpha, rho = params
        cache_key = (int(Nres), round(float(p), 12), round(float(alpha), 12), round(float(rho), 12))
        if use_cache and cache_key in cache:
            return cache[cache_key]
        
        scores = []
        for r in range(repeats):
            iter_seed = seed + 100 * r
            try:
                score = evaluate_esn(int(Nres), p, alpha, rho, u_train, y_train, iter_seed)
            except Exception as e:
                raise
            scores.append(score)
        
        mean_score = float(np.mean(scores))
        if use_cache:
            cache[cache_key] = mean_score
        return mean_score
    
    if opt_name == "optuna":
        return objective_optuna
    elif opt_name == "skopt":
        return objective_skopt
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def optimize_study(label, data, trials=trials, jobs=jobs, seed=seed):
    data = np.asarray(data, dtype=np.float32)
    N_train = len(data) // 2
    
    u_train = data[:N_train-1]
    y_train = data[1:N_train]
    
    use_cache = False#(jobs == 1) 
    objective = objective_factory("optuna", u_train, y_train, repeats=repeats, seed=seed, use_cache=use_cache)
    
    sampler = TPESampler(seed=seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study = optuna.create_study(
        sampler=sampler,
        pruner=pruner,
        direction="minimize",
        study_name=f"esn_{label}",
        load_if_exists=True
    )
    
    study.optimize(objective, n_trials=trials, n_jobs=jobs, show_progress_bar=True)
    return study


def save_results(obj):
    with open("optimized_params.pkl", "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def run_optuna_optimization():
    all_data = load_data()
    problems = ["a", "b", "c", "e"]
    results = {}
    
    for label in tqdm(problems):
        cache = {}
        data = all_data[label]['x']
        study = optimize_study(label, data, trials=trials, jobs=jobs, seed=seed)
        
        best_params = study.best_params
        best_value = study.best_value
        print(f"\nProblem {label}: Best NRMSE = {best_value:.6f}, params = {best_params}")
        
        results[label] = {"best_params": best_params, "best_value": best_value}
    
    print("Optuna Results: ", results)
    save_results(results)


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
        cache = {}
        data = all_data[label]['x']
        
        N_train = len(data) // 2
        u_train = data[:N_train-1]
        y_train = data[1:N_train]
        
        use_cache = False#(jobs == 1)
        objective = objective_factory("skopt", u_train, y_train, repeats=repeats, seed_local=seed, use_cache=use_cache)
        
        result = gp_minimize(
            objective,
            parameter_space,
            n_calls=trials,
            random_state=seed
        )
        
        results[label] = {"best_params": result.x, "best_value": result.fun}
    
    print("skopt results:", results)
    save_results(results)


def main():
    if optimizer == optimizers[0]:
        run_optuna_optimization()
    elif optimizer == optimizers[1]:
        run_skopt_optimization()
    else:
        raise ValueError()


if __name__ == "__main__":
    main()



"""
Optuna Results:  
{
    'a': {'best_params': {'Nres': 632, 'p': 0.037253043667155, 'alpha': 0.459272851235089, 'rho': 0.29023141605060604}, 'best_value': 0.0003393310180399567}, 
    'b': {'best_params': {'Nres': 814, 'p': 0.4498185943886106, 'alpha': 0.9276076641420588, 'rho': 0.314287351371228}, 'best_value': 0.0002586861955933273}, 
    'c': {'best_params': {'Nres': 902, 'p': 0.5855082690763134, 'alpha': 0.33364729981778907, 'rho': 0.24766302512462152}, 'best_value': 0.00023816426983103156}, 
    'e': {'best_params': {'Nres': 545, 'p': 0.9349879047529279, 'alpha': 0.7468897985893244, 'rho': 0.23162981573551591}, 'best_value': 0.0002043792192125693}
}

{
    'a': {'best_params': {'Nres': 728, 'p': 0.13957991126597663, 'alpha': 0.2922154340703646, 'rho': 0.5496061287562082}, 'best_value': 0.00028779252897948027}, 
    'b': {'best_params': {'Nres': 414, 'p': 0.062089841265991194, 'alpha': 0.7936016109756305, 'rho': 0.6063407607363651}, 'best_value': 0.0002715206937864423}, 
    'c': {'best_params': {'Nres': 374, 'p': 0.284638845893426, 'alpha': 0.9965251852821025, 'rho': 0.5149532108977859}, 'best_value': 0.00042221671901643276}, 
    'e': {'best_params': {'Nres': 738, 'p': 0.052115983846589296, 'alpha': 0.6603374425420461, 'rho': 0.3619010867556344}, 'best_value': 0.0004923829110339284}}
"""
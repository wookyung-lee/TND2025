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
jobs = 1
normalize = True
optimizer = "optuna"
optimizers = ["optuna", "skopt"]

cache = {}

def load_data():
    with open("timeseries_data.pkl", "rb") as f:
        all_data = pickle.load(f)
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
        rho = trial.suggest_float("rho", 0.8, 1.2)
        
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
        Real(0.8, 1.2, name="rho")
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
    'a': {'best_params': {'Nres': 333, 'p': 0.9919455079534404, 'alpha': 0.5043661273238631, 'rho': 1.063965261750552}, 'best_value': 0.0005712347337976098}, 
    'b': {'best_params': {'Nres': 659, 'p': 0.28192894209914016, 'alpha': 0.9019817937722701, 'rho': 0.9796075961238936}, 'best_value': 0.0004226422752253711}, 
    'c': {'best_params': {'Nres': 542, 'p': 0.9446500233494823, 'alpha': 0.605041141272483, 'rho': 0.8457941008952162}, 'best_value': 0.0005894032656215131}, 
    'e': {'best_params': {'Nres': 726, 'p': 0.2785517184767674, 'alpha': 0.857564760675654, 'rho': 0.8300096735131435}, 'best_value': 0.0003667937417048961}}
"""
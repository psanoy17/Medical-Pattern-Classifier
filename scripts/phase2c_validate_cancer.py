"""
Phase 2C: Validate top 6 configs (from heart tuning) on cancer dataset
Runs k-fold CV with multiple runs per fold using the configs found on heart data.
"""

import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cancer'))

from acor_cancer import FNN, objective_function
from lm_local_search import MultipleColonyACOR


def evaluate_config(config, X, y, n_splits, runs_per_fold, lm_max_cap, seed):
    input_dim = 9
    hidden_dim = 6
    output_dim = 1
    num_weights = FNN.get_num_weights(input_dim, hidden_dim, output_dim)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for run in range(runs_per_fold):
            model = FNN(input_dim, hidden_dim, output_dim)

            def obj_func(weights):
                return objective_function(weights, model, X_train, y_train)

            acor_lm = MultipleColonyACOR(
                obj_func=obj_func,
                dim=num_weights,
                n_colonies=config['n_colonies'],
                n_ants=2,
                n_samples=230,
                q=0.01,
                xi=0.95,
                max_iter=100,
                patience=15,
                local_patience=config['local_patience'],
                sharing_frequency=config['sharing_frequency'],
                sharing_ratio=config['sharing_ratio'],
                initial_mu=config['initial_mu'],
                lm_max_iterations=min(config['lm_max_iterations'], lm_max_cap),
                seed=seed + fold_idx + run
            )

            best_weights, best_loss, iterations, _ = acor_lm.optimize_with_history(
                lb=-3.0, ub=3.0, model=model, X_train=X_train, y_train=y_train
            )

            model.set_weights(best_weights)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            scores.append(acc)

    return float(np.mean(scores)), float(np.std(scores)), len(scores)


def main():
    parser = argparse.ArgumentParser(description='Validate top configs on cancer dataset')
    parser.add_argument('--n-splits', type=int, default=4, help='CV folds')
    parser.add_argument('--runs-per-fold', type=int, default=5, help='Runs per fold')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--lm-max-cap', type=int, default=10, help='Cap on LM iterations per local search')
    parser.add_argument('--output', type=str, default='scripts/phase2c_cancer_validate.csv', help='Output CSV')
    args = parser.parse_args()

    # Load cancer dataset
    cancer_path = Path(__file__).resolve().parents[1] / 'cancer' / 'cancer1.dat'
    data = pd.read_csv(cancer_path, sep=' ', header=None)
    X = data.iloc[:, :-2].values
    y_onehot = data.iloc[:, -2:].values
    y = np.argmax(y_onehot, axis=1)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Top 6 configs from heart validation (5 n_colonies=4 + 1 n_colonies=5)
    configs = [
        {'n_colonies':4, 'local_patience':10, 'sharing_frequency':15, 'sharing_ratio':0.15, 'initial_mu':1e-1,  'lm_max_iterations':10},
        {'n_colonies':4, 'local_patience':7,  'sharing_frequency':10, 'sharing_ratio':0.20, 'initial_mu':1e-3,  'lm_max_iterations':10},
        {'n_colonies':4, 'local_patience':7,  'sharing_frequency':15, 'sharing_ratio':0.10, 'initial_mu':1e-3,  'lm_max_iterations':10},
        {'n_colonies':4, 'local_patience':7,  'sharing_frequency':15, 'sharing_ratio':0.05, 'initial_mu':1e-3,  'lm_max_iterations':10},
        {'n_colonies':4, 'local_patience':7,  'sharing_frequency':15, 'sharing_ratio':0.20, 'initial_mu':1e-3,  'lm_max_iterations':10},
        {'n_colonies':5, 'local_patience':3,  'sharing_frequency':5,  'sharing_ratio':0.15, 'initial_mu':1e-3,  'lm_max_iterations':10},
    ]

    results = []
    start = time.time()
    print(f"Validating {len(configs)} configs on cancer with {args.n_splits}-fold CV and {args.runs_per_fold} runs/fold")

    for idx, cfg in enumerate(configs, 1):
        cfg_start = time.time()
        mean_acc, std_acc, n_evals = evaluate_config(
            cfg,
            X,
            y,
            args.n_splits,
            args.runs_per_fold,
            args.lm_max_cap,
            args.seed,
        )
        elapsed = time.time() - cfg_start
        print(f"[{idx}/{len(configs)}] acc={mean_acc:.4f}+/-{std_acc:.4f} ({n_evals} evals, {elapsed:.1f}s)")
        cfg_out = cfg.copy()
        cfg_out.update({'mean_accuracy': mean_acc, 'std_accuracy': std_acc, 'n_evals': n_evals, 'dataset': 'cancer'})
        results.append(cfg_out)

    total = time.time() - start
    print(f"Done in {total/60:.1f} minutes")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()

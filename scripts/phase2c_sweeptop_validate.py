"""
Phase 2C: Validate new sweep-identified top configs (4x5 budget)
Cross-dataset validation of the best single configs from 500-config sweeps.
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


def evaluate_config_heart(config, X, y, n_splits, runs_per_fold, lm_max_cap, seed):
    """Evaluate config on heart dataset."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'heart'))
    from acor_heart import FNN, objective_function
    from lm_local_search import MultipleColonyACOR
    
    input_dim = 35
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


def evaluate_config_diabetes(config, X, y, n_splits, runs_per_fold, lm_max_cap, seed):
    """Evaluate config on diabetes dataset."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'diabetes'))
    from acor_diabetes import FNN, objective_function
    from lm_local_search import MultipleColonyACOR
    
    input_dim = 8
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


def evaluate_config_cancer(config, X, y, n_splits, runs_per_fold, lm_max_cap, seed):
    """Evaluate config on cancer dataset."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cancer'))
    from acor_cancer import FNN, objective_function
    from lm_local_search import MultipleColonyACOR
    
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
    parser = argparse.ArgumentParser(description='Validate sweep-identified top configs')
    parser.add_argument('--n-splits', type=int, default=4, help='CV folds')
    parser.add_argument('--runs-per-fold', type=int, default=5, help='Runs per fold')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--lm-max-cap', type=int, default=10, help='Cap on LM iterations')
    args = parser.parse_args()

    base_path = Path(__file__).resolve().parents[1]
    
    # Top configs from 500-config sweeps
    configs_to_validate = [
        {
            'dataset': 'heart',
            'path': base_path / 'heart' / 'heart1.dat',
            'input_dim': 35,
            'eval_func': evaluate_config_heart,
            'configs': [
                {'n_colonies': 4, 'local_patience': 10, 'sharing_frequency': 15, 'sharing_ratio': 0.15, 'initial_mu': 0.1, 'lm_max_iterations': 10, 'name': 'mu0.1_test'},
            ]
        },
        {
            'dataset': 'diabetes',
            'path': base_path / 'diabetes' / 'diabetes1.dat',
            'input_dim': 8,
            'eval_func': evaluate_config_diabetes,
            'configs': [
                {'n_colonies': 4, 'local_patience': 10, 'sharing_frequency': 15, 'sharing_ratio': 0.15, 'initial_mu': 0.1, 'lm_max_iterations': 10, 'name': 'mu0.1_test'},
            ]
        },
        {
            'dataset': 'cancer',
            'path': base_path / 'cancer' / 'cancer1.dat',
            'input_dim': 9,
            'eval_func': evaluate_config_cancer,
            'configs': [
                {'n_colonies': 4, 'local_patience': 10, 'sharing_frequency': 15, 'sharing_ratio': 0.15, 'initial_mu': 0.1, 'lm_max_iterations': 10, 'name': 'mu0.1_test'},
            ]
        },
    ]

    all_results = []
    start = time.time()

    for dataset_info in configs_to_validate:
        dataset = dataset_info['dataset']
        data_path = dataset_info['path']
        eval_func = dataset_info['eval_func']
        
        print(f"\n{'='*60}")
        print(f"Validating {dataset.upper()} configs (4x5 budget)")
        print(f"{'='*60}")
        
        # Load dataset
        data = pd.read_csv(data_path, sep=' ', header=None)
        X = data.iloc[:, :-2].values
        y_onehot = data.iloc[:, -2:].values
        y = np.argmax(y_onehot, axis=1)
        
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        for cfg in dataset_info['configs']:
            cfg_start = time.time()
            config_copy = {k: v for k, v in cfg.items() if k != 'name'}
            mean_acc, std_acc, n_evals = eval_func(
                config_copy, X, y, args.n_splits, args.runs_per_fold, args.lm_max_cap, args.seed
            )
            elapsed = time.time() - cfg_start
            
            print(f"  [{cfg['name']}] acc={mean_acc:.4f}+/-{std_acc:.4f} ({n_evals} evals, {elapsed:.1f}s)")
            
            result = {
                'dataset': dataset,
                **config_copy,
                'mean_accuracy': mean_acc,
                'std_accuracy': std_acc,
                'n_evals': n_evals
            }
            all_results.append(result)

    total = time.time() - start
    print(f"\n{'='*60}")
    print(f"Total time: {total/60:.1f} minutes")
    print(f"{'='*60}")

    # Save results
    out_path = Path('scripts/phase2c_sweeptop_validate_4x5.csv')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_results).to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()

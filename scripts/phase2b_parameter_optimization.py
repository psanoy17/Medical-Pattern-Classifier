r"""
Phase 2B: Hybrid Parameter Sensitivity (ACOR + Minimal LM)

Tests parameter sensitivity without expensive Jacobian computation.
Uses high local_patience (20) to almost never trigger LM.

Configurations tested: 36 (4 × 3 × 3)
- n_colonies: [2, 3, 4, 5]
- local_patience: [3, 5, 20]  <- HIGH to skip LM
- initial_mu: [1e-4, 1e-2, 1e-1]

Evaluation: 2-fold CV, 2 runs per fold = 4 evals per config
Total: 36 configs × 4 evals = 144 runs
Time: ~8-10 minutes
"""

import os
import sys
import csv
import itertools
import time

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from heart.acor_heart import FNN, objective_function
from lm_local_search import MultipleColonyACOR


def load_heart_data():
    import pandas as pd
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'heart', 'heart1.dat'), sep=' ', header=None)
    X = data.iloc[:, :-2].values
    y_onehot = data.iloc[:, -2:].values
    y = np.argmax(y_onehot, axis=1)
    return X, y


def evaluate_config(config, X, y):
    """Evaluate single configuration"""
    try:
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        accuracies = []
        losses = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            for run in range(2):
                model = FNN(35, 6, 1)

                def obj_fn(weights):
                    if weights.ndim == 1:
                        return objective_function(weights, model, X_train, y_train)
                    else:
                        return np.array([objective_function(w, model, X_train, y_train) for w in weights])

                acor_lm = MultipleColonyACOR(
                    obj_func=obj_fn,
                    dim=FNN.get_num_weights(35, 6, 1),
                    n_colonies=config['n_colonies'],
                    n_ants=2,
                    n_samples=230,
                    q=0.01,
                    xi=0.95,
                    max_iter=100,
                    patience=15,
                    local_patience=config['local_patience'],
                    sharing_frequency=10,
                    sharing_ratio=0.1,
                    initial_mu=config['initial_mu'],
                    lm_max_iterations=5,  # Very low, rarely invoked with high local_patience
                    seed=42 + run
                )

                best_weights, best_loss, _, _ = acor_lm.optimize_with_history(
                    lb=-3.0, ub=3.0, model=model, X_train=X_train, y_train=y_train
                )

                model.set_weights(best_weights)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                accuracies.append(acc)
                losses.append(best_loss)

        return {
            'mean_accuracy': float(np.mean(accuracies)),
            'std_accuracy': float(np.std(accuracies)),
            'mean_loss': float(np.mean(losses)),
            'std_loss': float(np.std(losses)),
            'success': True
        }

    except Exception as e:
        return {
            'mean_accuracy': np.nan,
            'std_accuracy': np.nan,
            'mean_loss': np.nan,
            'std_loss': np.nan,
            'success': False,
            'error': str(e)[:50]
        }


def main():
    print("="*70)
    print("PHASE 2B: HYBRID PARAMETER SENSITIVITY ANALYSIS")
    print("="*70)

    X, y = load_heart_data()
    print(f"\nDataset: {X.shape}")

    grid = {
        'n_colonies': [2, 3, 4, 5],
        'local_patience': [3, 5, 20],  # HIGH to skip LM
        'initial_mu': [1e-4, 1e-2, 1e-1],
    }

    keys = list(grid.keys())
    configs = [dict(zip(keys, combo)) for combo in itertools.product(*[grid[k] for k in keys])]

    print(f"\nGrid: {len(configs)} configurations")
    print(f"  n_colonies: {grid['n_colonies']}")
    print(f"  local_patience: {grid['local_patience']} (HIGH = LM rarely triggers)")
    print(f"  initial_mu: {grid['initial_mu']}")
    print(f"\nEvaluation: 2-fold CV, 2 runs/fold = 4 evals per config")
    print(f"Total runs: {len(configs) * 4}")
    print()

    results = []
    start_time = time.time()

    for idx, config in enumerate(configs, 1):
        cfg_start = time.time()
        result = evaluate_config(config, X, y)
        cfg_time = time.time() - cfg_start

        col = config['n_colonies']
        lp = config['local_patience']
        mu = config['initial_mu']

        if result['success']:
            acc_str = f"{result['mean_accuracy']:.4f}±{result['std_accuracy']:.4f}"
        else:
            acc_str = f"FAIL: {result.get('error', 'Unknown')}"

        print(f"[{idx:2d}/36] {cfg_time:5.1f}s | n={col} lp={lp:2d} mu={mu:.0e} | {acc_str}")

        results.append((config, result))

    total_time = time.time() - start_time

    # Write results
    out_path = "scripts/phase2b_sensitivity_results.csv"
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(keys + ['mean_accuracy', 'std_accuracy', 'mean_loss', 'std_loss', 'success'])

        sorted_results = sorted(results, key=lambda x: x[1]['mean_accuracy'] if x[1]['success'] else -1, reverse=True)
        for config, result in sorted_results:
            row = [config[k] for k in keys] + [
                result['mean_accuracy'],
                result['std_accuracy'],
                result['mean_loss'],
                result['std_loss'],
                result['success']
            ]
            writer.writerow(row)

    print(f"\n{'='*70}")
    print(f"Total: {total_time:.1f}s ({total_time/len(configs):.1f}s per config)")
    print(f"Results: {out_path}")

    # Top 10
    success = [r for c, r in results if r['success']]
    if success:
        print(f"\nTop 10 Configurations:")
        top_10 = sorted(results, key=lambda x: x[1]['mean_accuracy'] if x[1]['success'] else -1, reverse=True)[:10]
        for i, (config, result) in enumerate(top_10, 1):
            print(f"  {i:2d}. n={config['n_colonies']} lp={config['local_patience']:2d} mu={config['initial_mu']:.0e} "
                  f"| acc={result['mean_accuracy']:.4f}±{result['std_accuracy']:.4f}")


if __name__ == '__main__':
    main()

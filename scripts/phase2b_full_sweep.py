r"""
Phase 2B: Full Parameter Sweep (ACOR-only, no LM)

Tests all 36 configurations without LM local search to:
1. Identify best parameter combinations
2. Avoid expensive Jacobian computation
3. Get clean parameter sensitivity data

Parameters tested:
- n_colonies: [2, 3, 4, 5]
- local_patience: [5, 10, 15]
- initial_mu: [1e-4, 1e-2, 1e-1]

Evaluation: 2-fold CV, 2 runs/fold = 4 evals per config
Total: 36 configs × 4 evals = 144 runs
Estimated time: 8-10 minutes (ACOR only, no LM overhead)

Strategy: Disable LM by setting local_patience very high (100)
so LM is never triggered, pure colony-based optimization.
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
    """Evaluate single configuration (ACOR only, LM disabled)"""
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

                # Use hybrid with LM disabled (very high local_patience = never triggers)
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
                    local_patience=100,  # VERY HIGH: LM never triggered
                    sharing_frequency=10,
                    sharing_ratio=0.1,
                    initial_mu=config['initial_mu'],
                    lm_max_iterations=1,  # Won't be called
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
    print("PHASE 2B: FULL PARAMETER SWEEP (ACOR ONLY)")
    print("="*70)

    X, y = load_heart_data()
    print(f"\nDataset: {X.shape}")

    # Grid definition
    grid = {
        'n_colonies': [2, 3, 4, 5],
        'local_patience': [5, 10, 15],
        'initial_mu': [1e-4, 1e-2, 1e-1],
    }

    keys = list(grid.keys())
    configs = [dict(zip(keys, combo)) for combo in itertools.product(*[grid[k] for k in keys])]

    print(f"\nConfiguration Space:")
    print(f"  n_colonies: {grid['n_colonies']}")
    print(f"  local_patience: {grid['local_patience']}")
    print(f"  initial_mu: {grid['initial_mu']}")
    print(f"  Total configs: {len(configs)}")
    print(f"\nNote: LM is DISABLED (local_patience set to 100)")
    print(f"This is pure colony-based optimization without local search overhead")
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
            acc_str = f"FAIL"

        print(f"[{idx:2d}/36] {cfg_time:5.1f}s | n={col} lp={lp:2d} mu={mu:.0e} | {acc_str}")

        results.append((config, result))

    total_time = time.time() - start_time

    # Write results to CSV
    out_path = "scripts/phase2b_full_sweep_results.csv"
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
    print(f"SWEEP COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {total_time:.1f}s ({total_time/len(configs):.1f}s per config)")
    print(f"Results: {out_path}")

    # Top 10
    success_results = [(c, r) for c, r in results if r['success']]
    if success_results:
        print(f"\nTOP 10 CONFIGURATIONS (by accuracy):")
        print(f"{'Rank':<5} {'n_col':<6} {'l_pat':<6} {'mu':<8} {'Accuracy':<15} {'Loss':<10}")
        print("-" * 60)
        
        top_10 = sorted(success_results, key=lambda x: x[1]['mean_accuracy'], reverse=True)[:10]
        for i, (config, result) in enumerate(top_10, 1):
            print(f"{i:<5} {config['n_colonies']:<6} {config['local_patience']:<6} "
                  f"{config['initial_mu']:.0e}    {result['mean_accuracy']:.4f}±{result['std_accuracy']:.4f}  "
                  f"{result['mean_loss']:.4f}")

    # Statistics
    all_accs = [r['mean_accuracy'] for c, r in success_results]
    if all_accs:
        print(f"\nAccuracy Statistics:")
        print(f"  Mean:   {np.mean(all_accs):.4f}")
        print(f"  Median: {np.median(all_accs):.4f}")
        print(f"  Std:    {np.std(all_accs):.4f}")
        print(f"  Min:    {np.min(all_accs):.4f}")
        print(f"  Max:    {np.max(all_accs):.4f}")


if __name__ == '__main__':
    main()

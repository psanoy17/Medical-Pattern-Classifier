"""
Phase 2C: 500-Config Random Sweep on Diabetes Dataset
Full parameter exploration to match heart dataset sweep methodology.
"""

import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'diabetes'))

from acor_diabetes import FNN, objective_function
from lm_local_search import MultipleColonyACOR


def evaluate_single_run(config, X_train, y_train, X_test, y_test, seed):
    """Single train-test run for a config."""
    input_dim = 8
    hidden_dim = 6
    output_dim = 1
    num_weights = FNN.get_num_weights(input_dim, hidden_dim, output_dim)

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
        lm_max_iterations=min(config['lm_max_iterations'], 5),
        seed=seed
    )

    best_weights, best_loss, iterations, _ = acor_lm.optimize_with_history(
        lb=-3.0, ub=3.0, model=model, X_train=X_train, y_train=y_train
    )

    model.set_weights(best_weights)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc


def main():
    parser = argparse.ArgumentParser(description='500-config random sweep on diabetes')
    parser.add_argument('--n-configs', type=int, default=500, help='Number of configs to sample')
    parser.add_argument('--n-runs', type=int, default=2, help='Runs per config')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='scripts/phase2c_diabetes_sweep_500.csv', help='Output CSV')
    args = parser.parse_args()

    # Load diabetes dataset
    diabetes_path = Path(__file__).resolve().parents[1] / 'diabetes' / 'diabetes1.dat'
    data = pd.read_csv(diabetes_path, sep=' ', header=None)
    X = data.iloc[:, :-2].values
    y_onehot = data.iloc[:, -2:].values
    y = np.argmax(y_onehot, axis=1)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 80-20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    print(f"Loaded diabetes: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Training: {len(X_train)}, Test: {len(X_test)}")
    print(f"Running {args.n_configs} random configs with {args.n_runs} runs each")

    # Parameter space
    n_colonies_list = [2, 3, 4, 5]
    local_patience_list = [3, 5, 7, 10]
    sharing_frequency_list = [5, 10, 15, 20]
    sharing_ratio_list = [0.05, 0.10, 0.15, 0.20]
    initial_mu_list = [0.001, 0.01, 0.1]
    lm_max_iterations_list = [10]

    rng = np.random.RandomState(args.seed)
    results = []
    start = time.time()

    for cfg_idx in range(args.n_configs):
        # Random sample from parameter space
        config = {
            'n_colonies': rng.choice(n_colonies_list),
            'local_patience': rng.choice(local_patience_list),
            'sharing_frequency': rng.choice(sharing_frequency_list),
            'sharing_ratio': rng.choice(sharing_ratio_list),
            'initial_mu': rng.choice(initial_mu_list),
            'lm_max_iterations': rng.choice(lm_max_iterations_list),
        }

        cfg_accs = []
        for run in range(args.n_runs):
            acc = evaluate_single_run(config, X_train, y_train, X_test, y_test, args.seed + cfg_idx + run)
            cfg_accs.append(acc)

        mean_acc = np.mean(cfg_accs)
        result = config.copy()
        result['mean_accuracy'] = mean_acc
        results.append(result)

        if (cfg_idx + 1) % 50 == 0:
            elapsed = time.time() - start
            print(f"[{cfg_idx+1}/{args.n_configs}] acc={mean_acc:.4f}, time={elapsed:.1f}s")

    total = time.time() - start
    print(f"\nDone in {total/60:.1f} minutes")
    print(f"Best accuracy: {max(r['mean_accuracy'] for r in results):.4f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()

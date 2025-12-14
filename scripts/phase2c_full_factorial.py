"""
Phase 2C: Full Factorial Parameter Sweep
Complete evaluation of all hybrid-specific parameters with original suggested ranges

Parameters:
- n_colonies: {2, 3, 4, 5}
- local_patience: {3, 5, 7, 10}
- sharing_frequency: {5, 10, 15, 20}
- sharing_ratio: {0.05, 0.1, 0.15, 0.2}
- initial_mu: {1e-4, 1e-3, 1e-2, 1e-1}
- lm_max_iterations: {10, 25, 50, 100}

Total configurations: 4 × 4 × 4 × 4 × 4 × 4 = 4,096
Strategy: Random sampling approach for efficient exploration
"""

import os
import sys
import time
import argparse
import csv
import random
from itertools import product
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'heart'))

# Import from heart dataset
from acor_heart import SOCHA_ACOR, FNN, objective_function
from lm_local_search import MultipleColonyACOR


def evaluate_config(config, X_train, X_test, y_train, y_test, n_runs, config_idx, total_configs):
    """Evaluate a single configuration with train/test split"""
    
    print(f"\n[{config_idx}/{total_configs}] Evaluating:")
    param_str = (f"n_col={config['n_colonies']}, lp={config['local_patience']}, "
                 f"sf={config['sharing_frequency']}, sr={config['sharing_ratio']:.2f}, "
                 f"mu={config['initial_mu']:.0e}, lm_iter={config['lm_max_iterations']}")
    print(f"  {param_str}")
    
    config_start = time.time()
    
    try:
        input_dim = 35
        hidden_dim = 6
        output_dim = 1
        num_weights = FNN.get_num_weights(input_dim, hidden_dim, output_dim)
        
        run_scores = []
        
        for run in range(n_runs):
            model = FNN(input_dim, hidden_dim, output_dim)
            
            def obj_func(weights):
                return objective_function(weights, model, X_train, y_train)
            
            # Run hybrid ACOR-LM with current config
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
                lm_max_iterations=config['lm_max_iterations'],
                seed=42 + run
            )
            
            # Use optimize_with_history to get loss history
            best_weights, best_loss, iterations, loss_history = acor_lm.optimize_with_history(
                lb=-3.0, ub=3.0, model=model, X_train=X_train, y_train=y_train
            )
            
            # Evaluate on test set
            model.set_weights(best_weights)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            run_scores.append(acc)
        
        mean_accuracy = np.mean(run_scores)
        std_accuracy = np.std(run_scores)
        
        result = {
            'n_colonies': config['n_colonies'],
            'local_patience': config['local_patience'],
            'sharing_frequency': config['sharing_frequency'],
            'sharing_ratio': config['sharing_ratio'],
            'initial_mu': config['initial_mu'],
            'lm_max_iterations': config['lm_max_iterations'],
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'n_runs': len(run_scores),
            'status': 'SUCCESS'
        }
        
        config_time = time.time() - config_start
        print(f"  [OK] Mean acc={mean_accuracy:.4f}+/-{std_accuracy:.4f} ({config_time:.1f}s)")
        
        return result
        
    except Exception as e:
        config_time = time.time() - config_start
        print(f"  [ERROR] {str(e)[:80]} ({config_time:.1f}s)")
        
        return {
            'n_colonies': config['n_colonies'],
            'local_patience': config['local_patience'],
            'sharing_frequency': config['sharing_frequency'],
            'sharing_ratio': config['sharing_ratio'],
            'initial_mu': config['initial_mu'],
            'lm_max_iterations': config['lm_max_iterations'],
            'mean_accuracy': np.nan,
            'std_accuracy': np.nan,
            'n_runs': 0,
            'status': 'ERROR',
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='Phase 2C: Full Factorial Parameter Sweep')
    parser.add_argument('--n-runs', type=int, default=2, help='Runs per config')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max-configs', type=int, default=100, help='Max configs to evaluate')
    parser.add_argument('--output', type=str, default='scripts/phase2c_results.csv', help='Output CSV')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("PHASE 2C: FULL FACTORIAL PARAMETER SWEEP (Heart Dataset)")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Runs per config: {args.n_runs}")
    print(f"  Max configs: {args.max_configs}")
    print(f"  Random seed: {args.seed}")
    
    # Load heart disease data
    print("\nLoading heart disease dataset...")
    heart_path = os.path.join(os.path.dirname(__file__), '..', 'heart', 'heart1.dat')
    data = pd.read_csv(heart_path, sep=' ', header=None)
    X = data.iloc[:, :-2].values
    y_onehot = data.iloc[:, -2:].values
    y = np.argmax(y_onehot, axis=1)
    print(f"  Shape: {X.shape}, Classes: {len(np.unique(y))}")
    
    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Define parameter ranges (COMPLETE original list)
    param_ranges = {
        'n_colonies': [2, 3, 4, 5],
        'local_patience': [3, 5, 7, 10],
        'sharing_frequency': [5, 10, 15, 20],
        'sharing_ratio': [0.05, 0.1, 0.15, 0.2],
        'initial_mu': [1e-4, 1e-3, 1e-2, 1e-1],
        'lm_max_iterations': [10, 25, 50, 100]
    }
    
    keys = list(param_ranges.keys())
    all_combos = list(product(*[param_ranges[k] for k in keys]))
    
    print(f"\nParameter Space:")
    for k, v in param_ranges.items():
        print(f"  {k}: {v}")
    print(f"\nTotal configurations: {len(all_combos)}")
    
    # Random sample
    random.seed(args.seed)
    sampled_combos = random.sample(all_combos, min(args.max_configs, len(all_combos)))
    configs = [dict(zip(keys, combo)) for combo in sampled_combos]
    
    print(f"Randomly sampling {len(configs)} configurations")
    print(f"Total evals: {len(configs)} configs × {args.n_runs} runs = {len(configs) * args.n_runs} evals")
    print("=" * 80)
    
    # Train/test split for efficiency
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=args.seed, stratify=y)
    
    start_time = time.time()
    results = []
    
    for config_idx, config in enumerate(configs, 1):
        result = evaluate_config(config, X_train, X_test, y_train, y_test, args.n_runs, config_idx, len(configs))
        results.append(result)
    
    total_time = time.time() - start_time
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    
    print("\n" + "=" * 80)
    print("SWEEP COMPLETE")
    print("=" * 80)
    
    successful = [r for r in results if r['status'] == 'SUCCESS']
    failed = [r for r in results if r['status'] == 'ERROR']
    
    print(f"\nResults Summary:")
    print(f"  Total configs: {len(configs)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
    
    if successful:
        accuracies = [r['mean_accuracy'] for r in successful]
        print(f"\nAccuracy Statistics:")
        print(f"  Mean: {np.mean(accuracies):.4f}")
        print(f"  Std:  {np.std(accuracies):.4f}")
        print(f"  Min:  {np.min(accuracies):.4f}")
        print(f"  Max:  {np.max(accuracies):.4f}")
        
        # Best config
        best_idx = np.argmax(accuracies)
        best = successful[best_idx]
        print(f"\nBEST Configuration (Acc={best['mean_accuracy']:.4f}):")
        for k in keys:
            print(f"   {k}: {best[k]}")
        
        # Worst config
        worst_idx = np.argmin(accuracies)
        worst = successful[worst_idx]
        print(f"\nWORST Configuration (Acc={worst['mean_accuracy']:.4f}):")
        for k in keys:
            print(f"   {k}: {worst[k]}")
        
        print(f"\nResults saved: {args.output}")
        print(f"Improvement range: {np.max(accuracies) - np.min(accuracies):.4f}")


if __name__ == '__main__':
    main()

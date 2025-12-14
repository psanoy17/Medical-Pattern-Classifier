"""
Quick K-Fold Comparison: Baseline ACOR vs Hybrid ACOR-LM (simplified)
- 2-fold CV (faster)
- 10 runs per fold (instead of 50)
- Hybrid uses only light LM (local_patience=10 means LM rarely triggers)
"""

import numpy as np
import pandas as pd
import os
import sys
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from heart.acor_heart import SOCHA_ACOR, FNN, objective_function
from lm_local_search import MultipleColonyACOR

np.random.seed(42)


def load_heart_data():
    data = pd.read_csv(
        os.path.join(os.path.dirname(__file__), 'heart1.dat'),
        sep=' ',
        header=None
    )
    X = data.iloc[:, :-2].values
    y_onehot = data.iloc[:, -2:].values
    y = np.argmax(y_onehot, axis=1)
    return X, y


def evaluate_baseline_acor(X_train, X_test, y_train, y_test, n_runs=10):
    """Evaluate baseline ACOR"""
    results = {'accuracy': [], 'loss': []}
    
    for run in range(n_runs):
        model = FNN(35, 6, 1)
        
        def obj_func(weights):
            return objective_function(weights, model, X_train, y_train)
        
        acor = SOCHA_ACOR(
            obj_func=obj_func,
            dim=FNN.get_num_weights(35, 6, 1),
            n_ants=2,
            n_samples=230,
            q=0.01,
            xi=0.95,
            max_iter=100,
            patience=15,
            seed=42 + run
        )
        
        best_weights, best_loss = acor.optimize(lb=-3.0, ub=3.0)
        
        model.set_weights(best_weights)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        results['accuracy'].append(acc)
        results['loss'].append(best_loss)
        print(f"  Baseline run {run+1}/{n_runs}: acc={acc:.4f}", end="\n")
    
    return results


def evaluate_hybrid_acor_lm(X_train, X_test, y_train, y_test, n_runs=10):
    """Evaluate hybrid ACOR-LM with high local_patience to minimize LM invocations"""
    results = {'accuracy': [], 'loss': []}
    
    for run in range(n_runs):
        model = FNN(35, 6, 1)
        
        def obj_func(weights):
            return objective_function(weights, model, X_train, y_train)
        
        acor_lm = MultipleColonyACOR(
            obj_func=obj_func,
            dim=FNN.get_num_weights(35, 6, 1),
            n_colonies=3,
            n_ants=2,
            n_samples=230,
            q=0.01,
            xi=0.95,
            max_iter=100,
            patience=15,
            local_patience=15,  # HIGH: LM rarely triggers
            sharing_frequency=10,
            sharing_ratio=0.1,
            initial_mu=0.001,
            lm_max_iterations=10,  # LOW: fewer LM iterations when triggered
            seed=42 + run
        )
        
        best_weights, best_loss, _, _ = acor_lm.optimize_with_history(
            lb=-3.0, ub=3.0, model=model, X_train=X_train, y_train=y_train
        )
        
        model.set_weights(best_weights)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        results['accuracy'].append(acc)
        results['loss'].append(best_loss)
        print(f"  Hybrid run {run+1}/{n_runs}: acc={acc:.4f}", end="\n")
    
    return results


def main():
    print("Loading data...")
    X, y = load_heart_data()
    print(f"Data shape: {X.shape}, Classes: {np.unique(y)}")
    
    print("\n" + "="*70)
    print("QUICK K-FOLD COMPARISON: BASELINE ACOR vs HYBRID ACOR-LM")
    print("="*70)
    print("Settings: 2-fold CV, 10 runs per fold, light LM (local_patience=15)")
    print()
    
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    all_baseline = {'accuracy': [], 'loss': []}
    all_hybrid = {'accuracy': [], 'loss': []}
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        print(f"\nFold {fold_idx}")
        print("Baseline ACOR:")
        baseline_results = evaluate_baseline_acor(X_train, X_test, y_train, y_test, n_runs=10)
        all_baseline['accuracy'].extend(baseline_results['accuracy'])
        all_baseline['loss'].extend(baseline_results['loss'])
        
        print("Hybrid ACOR-LM:")
        hybrid_results = evaluate_hybrid_acor_lm(X_train, X_test, y_train, y_test, n_runs=10)
        all_hybrid['accuracy'].extend(hybrid_results['accuracy'])
        all_hybrid['loss'].extend(hybrid_results['loss'])
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY RESULTS")
    print("="*70)
    
    baseline_acc = np.array(all_baseline['accuracy'])
    hybrid_acc = np.array(all_hybrid['accuracy'])
    
    print(f"\nBaseline ACOR (n={len(baseline_acc)}):")
    print(f"  Accuracy: {baseline_acc.mean():.4f} ± {baseline_acc.std():.4f}")
    print(f"  Loss: {np.mean(all_baseline['loss']):.6f} ± {np.std(all_baseline['loss']):.6f}")
    
    print(f"\nHybrid ACOR-LM (n={len(hybrid_acc)}):")
    print(f"  Accuracy: {hybrid_acc.mean():.4f} ± {hybrid_acc.std():.4f}")
    print(f"  Loss: {np.mean(all_hybrid['loss']):.6f} ± {np.std(all_hybrid['loss']):.6f}")
    
    print(f"\nImprovement:")
    acc_improvement = hybrid_acc.mean() - baseline_acc.mean()
    print(f"  Accuracy: {acc_improvement:+.4f} ({acc_improvement/baseline_acc.mean()*100:+.1f}%)")
    
    # Save results
    results_df = pd.DataFrame({
        'Metric': ['Accuracy Mean', 'Accuracy Std', 'Loss Mean', 'Loss Std'],
        'Baseline': [
            baseline_acc.mean(),
            baseline_acc.std(),
            np.mean(all_baseline['loss']),
            np.std(all_baseline['loss'])
        ],
        'Hybrid': [
            hybrid_acc.mean(),
            hybrid_acc.std(),
            np.mean(all_hybrid['loss']),
            np.std(all_hybrid['loss'])
        ]
    })
    
    results_path = os.path.join(os.path.dirname(__file__), 'quick_comparison_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()

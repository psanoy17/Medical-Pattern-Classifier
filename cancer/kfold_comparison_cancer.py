"""
K-Fold Cross-Validation Comparison for Cancer Classification

This script compares:
- Baseline ACOR (SOCHA-ACOR)
- Hybrid ACOR-LM (Multiple Colony ACOR with Levenberg-Marquardt local search)

Evaluation:
- 4-fold stratified cross-validation
- 50 independent runs per fold for each algorithm
- FNN Architecture: 9 inputs, 6 hidden (ReLU), 1 output (Sigmoid)
- Binary Cross-Entropy Loss
"""

import numpy as np
import pandas as pd
import os
import sys
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import baseline ACOR components
from acor_cancer import SOCHA_ACOR, FNN, objective_function

# Import hybrid ACOR-LM components
from lm_local_search import MultipleColonyACOR, LevenbergMarquardt

# Set random seed for reproducibility
np.random.seed(42)


# ==============================================================================
# DATA LOADING
# ==============================================================================
def load_cancer_data():
    """Load and return cancer dataset"""
    data = pd.read_csv(
        os.path.join(os.path.dirname(__file__), 'cancer1.dat'),
        sep=' ',
        header=None
    )
    
    X = data.iloc[:, :-2].values  # First 9 columns are features
    y_onehot = data.iloc[:, -2:].values  # Last 2 columns are one-hot encoded target
    y = np.argmax(y_onehot, axis=1)  # Convert to single label
    
    return X, y


# ==============================================================================
# EVALUATION FUNCTIONS
# ==============================================================================
def evaluate_baseline_acor(X_train, X_test, y_train, y_test, n_runs=50, verbose=False):
    """
    Evaluate Baseline ACOR (SOCHA-ACOR) using multiple independent runs
    
    Args:
        X_train: Training features (already scaled)
        X_test: Test features (already scaled)
        y_train: Training labels
        y_test: Test labels
        n_runs: Number of independent runs
        verbose: Print progress for each run
        
    Returns:
        Dictionary with evaluation results
    """
    results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'confusion_matrices': [],
        'best_losses': [],
        'iterations': []
    }
    
    input_dim = 9
    hidden_dim = 6
    output_dim = 1
    num_weights = FNN.get_num_weights(input_dim, hidden_dim, output_dim)
    
    for run in range(n_runs):
        if verbose:
            print(f"  Baseline Run {run + 1}/{n_runs}", end=" ")
        
        # Initialize model
        model = FNN(input_dim, hidden_dim, output_dim)
        
        # Create objective function
        def obj_func(weights):
            if weights.ndim == 1:
                return objective_function(weights, model, X_train, y_train)
            else:
                return np.array([objective_function(w, model, X_train, y_train) for w in weights])
        
        # Initialize and run ACOR
        acor = SOCHA_ACOR(
            obj_func=obj_func,
            dim=num_weights,
            n_ants=2,
            n_samples=148,
            q=0.95,
            xi=0.98,
            max_iter=100,
            patience=15,
            seed=42 + run
        )
        
        best_weights, best_loss, iterations = acor.optimize(lb=-3, ub=3)
        
        # Evaluate on test set
        model.set_weights(best_weights)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        results['accuracy'].append(acc)
        results['precision'].append(prec)
        results['recall'].append(rec)
        results['f1_score'].append(f1)
        results['confusion_matrices'].append(cm)
        results['best_losses'].append(best_loss)
        results['iterations'].append(iterations)
        
        if verbose:
            print(f"Acc: {acc:.3f}, Loss: {best_loss:.4f}")
    
    return results


def evaluate_hybrid_acor_lm(X_train, X_test, y_train, y_test, n_runs=50, verbose=False):
    """
    Evaluate Hybrid ACOR-LM using multiple independent runs
    
    Args:
        X_train: Training features (already scaled)
        X_test: Test features (already scaled)
        y_train: Training labels
        y_test: Test labels
        n_runs: Number of independent runs
        verbose: Print progress for each run
        
    Returns:
        Dictionary with evaluation results
    """
    results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'confusion_matrices': [],
        'best_losses': [],
        'iterations': []
    }
    
    input_dim = 9
    hidden_dim = 6
    output_dim = 1
    num_weights = FNN.get_num_weights(input_dim, hidden_dim, output_dim)
    
    for run in range(n_runs):
        if verbose:
            print(f"  Hybrid Run {run + 1}/{n_runs}", end=" ")
        
        # Initialize model (use FNN from baseline - same architecture)
        model = FNN(input_dim, hidden_dim, output_dim)
        
        # Create objective function
        def obj_func(weights):
            return objective_function(weights, model, X_train, y_train)
        
        # Initialize and run ACOR-LM
        acor_lm = MultipleColonyACOR(
            obj_func=obj_func,
            dim=num_weights,
            n_colonies=3,
            n_ants=2,
            n_samples=148,
            q=0.95,
            xi=0.98,
            max_iter=100,
            patience=15,
            sharing_frequency=10,
            sharing_ratio=0.1,
            seed=42 + run
        )
        
        best_weights, best_loss, iterations = acor_lm.optimize(
            lb=-3.0, ub=3.0, model=model, X_train=X_train, y_train=y_train
        )
        
        # Evaluate on test set
        model.set_weights(best_weights)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        results['accuracy'].append(acc)
        results['precision'].append(prec)
        results['recall'].append(rec)
        results['f1_score'].append(f1)
        results['confusion_matrices'].append(cm)
        results['best_losses'].append(best_loss)
        results['iterations'].append(iterations)
        
        if verbose:
            print(f"Acc: {acc:.3f}, Loss: {best_loss:.4f}")
    
    return results


# ==============================================================================
# K-FOLD CROSS-VALIDATION
# ==============================================================================
def kfold_cross_validation(X, y, n_splits=4, n_runs=50, verbose=True):
    """
    Perform k-fold cross-validation comparing Baseline ACOR vs Hybrid ACOR-LM
    
    Args:
        X: Full feature matrix
        y: Full target vector
        n_splits: Number of folds (default: 4)
        n_runs: Number of independent runs per fold per algorithm
        verbose: Print progress
        
    Returns:
        Dictionary with cross-validation results for both algorithms
    """
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Initialize result containers
    cv_results = {
        'baseline': {
            'fold_results': {metric: [] for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'iterations', 'losses']},
            'all_results': {metric: [] for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'iterations', 'best_losses']},
            'individual_runs': []  # Store individual run data with fold and run info
        },
        'hybrid': {
            'fold_results': {metric: [] for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'iterations', 'losses']},
            'all_results': {metric: [] for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'iterations', 'best_losses']},
            'individual_runs': []  # Store individual run data with fold and run info
        },
        'n_folds': n_splits,
        'n_runs_per_fold': n_runs
    }
    
    if verbose:
        print(f"\nPerforming {n_splits}-fold cross-validation...")
        print(f"Runs per fold per algorithm: {n_runs}")
        print("=" * 70)
    
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        if verbose:
            print(f"\n{'='*70}")
            print(f"FOLD {fold_idx + 1}/{n_splits}")
            print(f"{'='*70}")
            print(f"Training samples: {len(train_idx)}, Test samples: {len(test_idx)}")
        
        # Split and scale data for this fold
        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]
        
        scaler = StandardScaler()
        X_train_fold = scaler.fit_transform(X_train_fold)
        X_test_fold = scaler.transform(X_test_fold)
        
        # Evaluate Baseline ACOR
        if verbose:
            print(f"\n[Baseline ACOR] Running {n_runs} experiments...")
        baseline_results = evaluate_baseline_acor(
            X_train_fold, X_test_fold, y_train_fold, y_test_fold, 
            n_runs=n_runs, verbose=False
        )
        
        # Evaluate Hybrid ACOR-LM
        if verbose:
            print(f"[Hybrid ACOR-LM] Running {n_runs} experiments...")
        hybrid_results = evaluate_hybrid_acor_lm(
            X_train_fold, X_test_fold, y_train_fold, y_test_fold,
            n_runs=n_runs, verbose=False
        )
        
        # Store fold averages
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            cv_results['baseline']['fold_results'][metric].append(np.mean(baseline_results[metric]))
            cv_results['hybrid']['fold_results'][metric].append(np.mean(hybrid_results[metric]))
        
        cv_results['baseline']['fold_results']['iterations'].append(np.mean(baseline_results['iterations']))
        cv_results['baseline']['fold_results']['losses'].append(np.mean(baseline_results['best_losses']))
        cv_results['hybrid']['fold_results']['iterations'].append(np.mean(hybrid_results['iterations']))
        cv_results['hybrid']['fold_results']['losses'].append(np.mean(hybrid_results['best_losses']))
        
        # Store all individual results
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'iterations', 'best_losses']:
            cv_results['baseline']['all_results'][metric].extend(baseline_results[metric])
            cv_results['hybrid']['all_results'][metric].extend(hybrid_results[metric])
        
        # Store individual run data with fold and run information
        for run_idx in range(n_runs):
            cv_results['baseline']['individual_runs'].append({
                'fold': fold_idx + 1,
                'run': run_idx + 1,
                'algorithm': 'Baseline ACOR',
                'accuracy': baseline_results['accuracy'][run_idx],
                'precision': baseline_results['precision'][run_idx],
                'recall': baseline_results['recall'][run_idx],
                'f1_score': baseline_results['f1_score'][run_idx],
                'iterations': baseline_results['iterations'][run_idx],
                'best_loss': baseline_results['best_losses'][run_idx]
            })
            cv_results['hybrid']['individual_runs'].append({
                'fold': fold_idx + 1,
                'run': run_idx + 1,
                'algorithm': 'Hybrid ACOR-LM',
                'accuracy': hybrid_results['accuracy'][run_idx],
                'precision': hybrid_results['precision'][run_idx],
                'recall': hybrid_results['recall'][run_idx],
                'f1_score': hybrid_results['f1_score'][run_idx],
                'iterations': hybrid_results['iterations'][run_idx],
                'best_loss': hybrid_results['best_losses'][run_idx]
            })
        
        # Print fold summary
        if verbose:
            print(f"\nFold {fold_idx + 1} Summary:")
            print(f"  Baseline:  Acc={np.mean(baseline_results['accuracy']):.4f}, "
                  f"F1={np.mean(baseline_results['f1_score']):.4f}, "
                  f"Iter={np.mean(baseline_results['iterations']):.1f}")
            print(f"  Hybrid:    Acc={np.mean(hybrid_results['accuracy']):.4f}, "
                  f"F1={np.mean(hybrid_results['f1_score']):.4f}, "
                  f"Iter={np.mean(hybrid_results['iterations']):.1f}")
    
    return cv_results


# ==============================================================================
# RESULTS REPORTING
# ==============================================================================
def print_results(cv_results):
    """Print formatted cross-validation results"""
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION RESULTS - PER-FOLD AVERAGES")
    print("=" * 70)
    
    n_folds = cv_results['n_folds']
    
    for fold_idx in range(n_folds):
        print(f"\nFold {fold_idx + 1}:")
        for algo in ['baseline', 'hybrid']:
            algo_name = "Baseline ACOR" if algo == 'baseline' else "Hybrid ACOR-LM"
            fr = cv_results[algo]['fold_results']
            print(f"  {algo_name:15}: Acc={fr['accuracy'][fold_idx]:.4f}, "
                  f"Prec={fr['precision'][fold_idx]:.4f}, "
                  f"Rec={fr['recall'][fold_idx]:.4f}, "
                  f"F1={fr['f1_score'][fold_idx]:.4f}, "
                  f"Iter={fr['iterations'][fold_idx]:.1f}")
    
    print("\n" + "=" * 70)
    print("OVERALL RESULTS (Mean ± Std across all folds and runs)")
    print("=" * 70)
    
    for algo in ['baseline', 'hybrid']:
        algo_name = "Baseline ACOR" if algo == 'baseline' else "Hybrid ACOR-LM"
        ar = cv_results[algo]['all_results']
        
        print(f"\n{algo_name}:")
        print("-" * 60)
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            mean_val = np.mean(ar[metric])
            std_val = np.std(ar[metric])
            print(f"  {metric.capitalize():12}: {mean_val:.4f} ± {std_val:.4f}")
        print(f"  {'Iterations':12}: {np.mean(ar['iterations']):.1f} ± {np.std(ar['iterations']):.1f}")
        print(f"  {'Loss':12}: {np.mean(ar['best_losses']):.6f} ± {np.std(ar['best_losses']):.6f}")

def save_results(cv_results, output_dir):
    """Save results to text file"""
    # Save text report
    txt_path = os.path.join(output_dir, 'cancer_kfold_comparison_results.txt')
    with open(txt_path, 'w') as f:
        f.write("Cancer Classification - Baseline ACOR vs Hybrid ACOR-LM\n")
        f.write("K-Fold Cross-Validation Comparison\n")
        f.write("=" * 70 + "\n")
        f.write(f"Architecture: 9-6-1 (Total weights: 67)\n")
        f.write(f"Number of folds: {cv_results['n_folds']}\n")
        f.write(f"Runs per fold per algorithm: {cv_results['n_runs_per_fold']}\n\n")
        
        f.write("PER-FOLD RESULTS\n")
        f.write("=" * 70 + "\n")
        
        for fold_idx in range(cv_results['n_folds']):
            f.write(f"\nFold {fold_idx + 1}:\n")
            for algo in ['baseline', 'hybrid']:
                algo_name = "Baseline ACOR" if algo == 'baseline' else "Hybrid ACOR-LM"
                fr = cv_results[algo]['fold_results']
                f.write(f"  {algo_name}:\n")
                f.write(f"    Accuracy:   {fr['accuracy'][fold_idx]:.4f}\n")
                f.write(f"    Precision:  {fr['precision'][fold_idx]:.4f}\n")
                f.write(f"    Recall:     {fr['recall'][fold_idx]:.4f}\n")
                f.write(f"    F1-Score:   {fr['f1_score'][fold_idx]:.4f}\n")
                f.write(f"    Iterations: {fr['iterations'][fold_idx]:.1f}\n")
        
        f.write("\n\nOVERALL RESULTS (Mean ± Std)\n")
        f.write("=" * 70 + "\n")
        
        for algo in ['baseline', 'hybrid']:
            algo_name = "Baseline ACOR" if algo == 'baseline' else "Hybrid ACOR-LM"
            ar = cv_results[algo]['all_results']
            f.write(f"\n{algo_name}:\n")
            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                mean_val = np.mean(ar[metric])
                std_val = np.std(ar[metric])
                f.write(f"  {metric.capitalize()}: {mean_val:.4f} ± {std_val:.4f}\n")
            f.write(f"  Iterations: {np.mean(ar['iterations']):.1f} ± {np.std(ar['iterations']):.1f}\n")
            f.write(f"  Loss: {np.mean(ar['best_losses']):.6f} ± {np.std(ar['best_losses']):.6f}\n")
    
    print(f"\nResults saved to: {txt_path}")

def save_results_to_csv(cv_results, output_dir):
    """Save all individual run results to CSV file"""
    # Combine all individual runs from both algorithms
    all_runs = cv_results['baseline']['individual_runs'] + cv_results['hybrid']['individual_runs']
    
    # Create DataFrame
    df = pd.DataFrame(all_runs)
    
    # Reorder columns for better readability
    column_order = ['fold', 'run', 'algorithm', 'accuracy', 'precision', 'recall', 
                    'f1_score', 'iterations', 'best_loss']
    df = df[column_order]
    
    # Sort by fold, then algorithm, then run
    df = df.sort_values(['fold', 'algorithm', 'run']).reset_index(drop=True)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'cancer_kfold_all_runs.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"All individual runs saved to CSV: {csv_path}")
    print(f"Total runs: {len(df)} ({cv_results['n_folds']} folds × {cv_results['n_runs_per_fold']} runs × 2 algorithms)")

def create_comparison_plot(cv_results, output_dir):
    """Create and save comparison bar plot"""
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    x = np.arange(len(metrics))
    width = 0.35
    
    baseline_ar = cv_results['baseline']['all_results']
    hybrid_ar = cv_results['hybrid']['all_results']
    
    baseline_means = [np.mean(baseline_ar[m]) for m in metrics]
    baseline_stds = [np.std(baseline_ar[m]) for m in metrics]
    hybrid_means = [np.mean(hybrid_ar[m]) for m in metrics]
    hybrid_stds = [np.std(hybrid_ar[m]) for m in metrics]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, baseline_means, width, yerr=baseline_stds,
                   label='Baseline ACOR', capsize=5, alpha=0.7, color='skyblue')
    bars2 = ax.bar(x + width/2, hybrid_means, width, yerr=hybrid_stds,
                   label='Hybrid ACOR-LM', capsize=5, alpha=0.7, color='orange')
    
    ax.set_ylabel('Score')
    ax.set_title(f'Cancer Classification - Baseline ACOR vs Hybrid ACOR-LM\n'
                 f'({cv_results["n_folds"]}-fold CV, {cv_results["n_runs_per_fold"]} runs/fold)')
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'cancer_kfold_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved to: {plot_path}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print("Cancer Classification - K-Fold Cross-Validation Comparison")
    print("Baseline ACOR vs Hybrid ACOR-LM")
    print("=" * 70)
    
    # Load data
    X, y = load_cancer_data()
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target distribution: Class 0: {np.sum(y==0)}, Class 1: {np.sum(y==1)}")
    print(f"Architecture: 9 inputs, 6 hidden (ReLU), 1 output (Sigmoid)")
    print(f"Total weights: {FNN.get_num_weights(9, 6, 1)}")
    
    # Run k-fold cross-validation
    cv_results = kfold_cross_validation(X, y, n_splits=4, n_runs=50, verbose=True)
    
    # Print results
    print_results(cv_results)
    
    # Save results
    output_dir = os.path.dirname(__file__)
    save_results(cv_results, output_dir)
    
    # Save all individual runs to CSV
    save_results_to_csv(cv_results, output_dir)
    
    # Create comparison plot
    create_comparison_plot(cv_results, output_dir)
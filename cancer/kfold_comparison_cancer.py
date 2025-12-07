"""
K-Fold Cross-Validation Comparison for Cancer Classification
(Baseline ACOR vs Hybrid ACOR-LM)

Features:
- 4-fold Stratified CV
- 50 independent runs per fold
- Paired T-Test for Statistical Significance
- Detailed CSV logging (TP, TN, FP, FN included and placed first)
- Custom Console Output Format
"""

import numpy as np
import pandas as pd
import os
import sys
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import ACOR components
# Assumes acor_cancer.py and lm_local_search.py exist in the path
from acor_cancer import SOCHA_ACOR, FNN, objective_function
from lm_local_search import MultipleColonyACOR, LevenbergMarquardt

# Set random seed for reproducibility
np.random.seed(42)


# ==============================================================================
# DATA LOADING
# ==============================================================================
def load_cancer_data():
    """Load and return cancer dataset"""
    # Try to find the file in the current directory or the script's directory
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cancer1.dat')
    if not os.path.exists(file_path):
        file_path = 'cancer1.dat'  # Fallback to current working directory
        
    data = pd.read_csv(
        file_path,
        sep=' ',
        header=None
    )
    
    X = data.iloc[:, :-2].values   # First 9 columns are features
    y_onehot = data.iloc[:, -2:].values  # Last 2 columns are one-hot encoded target
    y = np.argmax(y_onehot, axis=1)  # Convert to single label
    
    return X, y


# ==============================================================================
# EVALUATION FUNCTIONS (Updated Output Format)
# ==============================================================================
def evaluate_baseline_acor(X_train, X_test, y_train, y_test, n_runs=50, verbose=False):
    """
    Evaluate Baseline ACOR (SOCHA-ACOR) using multiple independent runs
    """
    results = {
        'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 
        'confusion_matrices': [], 'best_losses': [], 'iterations': [],
        'TP': [], 'TN': [], 'FP': [], 'FN': []
    }
    
    input_dim = 9
    hidden_dim = 6
    output_dim = 1
    num_weights = FNN.get_num_weights(input_dim, hidden_dim, output_dim)
    
    for run in range(n_runs):
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
        
        # Calculate CM components (TN, FP, FN, TP) safely
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        if cm.size == 4:
             TN, FP, FN, TP = cm.ravel()
        else:
             TP = np.sum((y_pred == 1) & (y_test == 1))
             TN = np.sum((y_pred == 0) & (y_test == 0))
             FP = np.sum((y_pred == 1) & (y_test == 0))
             FN = np.sum((y_pred == 0) & (y_test == 1))
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        results['accuracy'].append(acc)
        results['precision'].append(prec)
        results['recall'].append(rec)
        results['f1_score'].append(f1)
        results['confusion_matrices'].append(cm)
        results['best_losses'].append(best_loss)
        results['iterations'].append(iterations)
        
        results['TP'].append(TP)
        results['TN'].append(TN)
        results['FP'].append(FP)
        results['FN'].append(FN)
        
        if verbose:
            # Updated Print Format
            print(f"[Run {run + 1}/{n_runs}] [Iter {iterations}] Stopping: No improvement for 15 iterations")
    
    return results


def evaluate_hybrid_acor_lm(X_train, X_test, y_train, y_test, n_runs=50, verbose=False):
    """
    Evaluate Hybrid ACOR-LM using multiple independent runs
    """
    results = {
        'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 
        'confusion_matrices': [], 'best_losses': [], 'iterations': [],
        'TP': [], 'TN': [], 'FP': [], 'FN': []
    }
    
    input_dim = 9
    hidden_dim = 6
    output_dim = 1
    num_weights = FNN.get_num_weights(input_dim, hidden_dim, output_dim)
    
    for run in range(n_runs):
        # Initialize model
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
        
        # Calculate CM components (TN, FP, FN, TP) safely
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        if cm.size == 4:
             TN, FP, FN, TP = cm.ravel()
        else:
             TP = np.sum((y_pred == 1) & (y_test == 1))
             TN = np.sum((y_pred == 0) & (y_test == 0))
             FP = np.sum((y_pred == 1) & (y_test == 0))
             FN = np.sum((y_pred == 0) & (y_test == 1))
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        results['accuracy'].append(acc)
        results['precision'].append(prec)
        results['recall'].append(rec)
        results['f1_score'].append(f1)
        results['confusion_matrices'].append(cm)
        results['best_losses'].append(best_loss)
        results['iterations'].append(iterations)
        
        results['TP'].append(TP)
        results['TN'].append(TN)
        results['FP'].append(FP)
        results['FN'].append(FN)
        
        if verbose:
            # Updated Print Format
            print(f"[Run {run + 1}/{n_runs}] [Iter {iterations}] Stopping: No improvement for 15 iterations")
    
    return results


# ==============================================================================
# K-FOLD CROSS-VALIDATION
# ==============================================================================
def kfold_cross_validation(X, y, n_splits=4, n_runs=50, verbose=True):
    """
    Perform k-fold cross-validation comparing Baseline ACOR vs Hybrid ACOR-LM
    """
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Initialize result containers
    cv_results = {
        'baseline': {
            'fold_results': {metric: [] for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'iterations', 'losses']},
            'all_results': {metric: [] for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'iterations', 'best_losses']},
            'individual_runs': [] 
        },
        'hybrid': {
            'fold_results': {metric: [] for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'iterations', 'losses']},
            'all_results': {metric: [] for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'iterations', 'best_losses']},
            'individual_runs': [] 
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
            n_runs=n_runs, verbose=True  # ENABLED VERBOSE HERE
        )
        
        # Evaluate Hybrid ACOR-LM
        if verbose:
            print(f"\n[Hybrid ACOR-LM] Running {n_runs} experiments...")
        hybrid_results = evaluate_hybrid_acor_lm(
            X_train_fold, X_test_fold, y_train_fold, y_test_fold,
            n_runs=n_runs, verbose=True  # ENABLED VERBOSE HERE
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
        
        # Store individual run data
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
                'best_loss': baseline_results['best_losses'][run_idx],
                'TP': baseline_results['TP'][run_idx],
                'TN': baseline_results['TN'][run_idx],
                'FP': baseline_results['FP'][run_idx],
                'FN': baseline_results['FN'][run_idx]
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
                'best_loss': hybrid_results['best_losses'][run_idx],
                'TP': hybrid_results['TP'][run_idx],
                'TN': hybrid_results['TN'][run_idx],
                'FP': hybrid_results['FP'][run_idx],
                'FN': hybrid_results['FN'][run_idx]
            })
        
        if verbose:
            print(f"  Baseline Avg Acc: {np.mean(baseline_results['accuracy']):.4f}")
            print(f"  Hybrid Avg Acc:   {np.mean(hybrid_results['accuracy']):.4f}")
    
    return cv_results


# ==============================================================================
# STATISTICAL COMPARISON (PAIRED T-TEST)
# ==============================================================================
def perform_paired_t_test(cv_results, metric='accuracy'):
    """Performs a paired t-test between Baseline and Hybrid results."""
    
    # Get arrays of results across all folds and runs
    baseline_scores = np.array(cv_results['baseline']['all_results'][metric])
    hybrid_scores = np.array(cv_results['hybrid']['all_results'][metric])
    
    if len(baseline_scores) != len(hybrid_scores) or len(baseline_scores) == 0:
        print(f"Error: Cannot perform paired t-test. Data length mismatch or zero data points.")
        return {'t_stat': np.nan, 'p_value': np.nan, 'is_significant': False, 'winner': 'Error', 'metric': metric}

    # Perform Paired T-Test
    t_stat, p_value = ttest_rel(hybrid_scores, baseline_scores)
    is_significant = p_value < 0.05
    
    mean_hybrid = np.mean(hybrid_scores)
    mean_baseline = np.mean(baseline_scores)
    
    winner = 'No statistically significant difference'
    if is_significant:
        winner = 'Hybrid ACOR-LM' if mean_hybrid > mean_baseline else 'Baseline ACOR'

    print("\n" + "=" * 70)
    print(f"PAIRED T-TEST STATISTICAL COMPARISON ({metric.upper()})")
    print("=" * 70)
    print(f"Data Points: {len(baseline_scores)}")
    print(f"Mean Baseline: {mean_baseline:.4f}")
    print(f"Mean Hybrid:   {mean_hybrid:.4f}")
    print(f"T-statistic:   {t_stat:.4f}")
    print(f"P-value:       {p_value:.6f}")
    
    interpretation = ""
    if is_significant:
        interpretation = (f"Result is statistically significant (p < 0.05). "
                          f"The mean performance of {winner} is better.")
    else:
        interpretation = (f"Result is NOT statistically significant (p >= 0.05). "
                          f"Difference is likely due to chance.")
    print(f"Conclusion: {interpretation}")
    
    return {'t_stat': t_stat, 'p_value': p_value, 'is_significant': is_significant, 'winner': winner, 'metric': metric}


# ==============================================================================
# RESULTS REPORTING & SAVING
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

def save_results(cv_results, output_dir, t_test_results):
    """Save summary text report including t-test results"""
    txt_path = os.path.join(output_dir, 'cancer_kfold_comparison_results.txt')
    try:
        with open(txt_path, 'w') as f:
            f.write("Cancer Classification - Baseline ACOR vs Hybrid ACOR-LM\n")
            f.write("K-Fold Cross-Validation Comparison\n")
            f.write("=" * 70 + "\n\n")
            
            # Overall Results
            f.write("OVERALL RESULTS (Mean ± Std)\n")
            f.write("=" * 70 + "\n")
            for algo in ['baseline', 'hybrid']:
                algo_name = "Baseline ACOR" if algo == 'baseline' else "Hybrid ACOR-LM"
                ar = cv_results[algo]['all_results']
                f.write(f"\n{algo_name}:\n")
                for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                    f.write(f"  {metric.capitalize()}: {np.mean(ar[metric]):.4f} ± {np.std(ar[metric]):.4f}\n")
            
            # Statistical Comparison
            f.write("\n\nSTATISTICAL COMPARISON (PAIRED T-TEST)\n")
            f.write("=" * 70 + "\n")
            f.write(f"Metric Compared: {t_test_results['metric'].capitalize()}\n")
            f.write(f"T-Statistic:     {t_test_results['t_stat']:.4f}\n")
            f.write(f"P-Value:         {t_test_results['p_value']:.6f}\n")
            f.write(f"Significant:     {'Yes' if t_test_results['is_significant'] else 'No'}\n")
            f.write(f"Winner:          {t_test_results['winner']}\n")
            
        print(f"\nSummary text report saved to: {txt_path}")
    except Exception as e:
        print(f"Error saving results to text file: {e}")

def save_results_to_csv(cv_results, output_dir):
    """
    Save all individual run results to CSV (Stacked format).
    Includes TP, TN, FP, FN placed BEFORE metrics.
    """
    all_runs = cv_results['baseline']['individual_runs'] + cv_results['hybrid']['individual_runs']
    df = pd.DataFrame(all_runs)
    
    # Column order: Identifiers -> CM -> Metrics -> Other
    column_order = [
        'fold', 'run', 'algorithm', 
        'TP', 'TN', 'FP', 'FN',
        'accuracy', 'precision', 'recall', 'f1_score', 
        'iterations', 'best_loss'
    ]
    df = df[column_order].sort_values(['fold', 'algorithm', 'run']).reset_index(drop=True)
    
    csv_path = os.path.join(output_dir, 'cancer_kfold_all_runs.csv')
    df.to_csv(csv_path, index=False)
    print(f"All individual runs saved to CSV: {csv_path}")

def save_paired_comparison_csv(cv_results, output_dir):
    """
    Saves a CSV file with side-by-side comparison per (fold, run).
    Includes TP, TN, FP, FN placed BEFORE other metrics.
    """
    baseline_runs = pd.DataFrame(cv_results['baseline']['individual_runs']).sort_values(['fold', 'run']).reset_index(drop=True)
    hybrid_runs = pd.DataFrame(cv_results['hybrid']['individual_runs']).sort_values(['fold', 'run']).reset_index(drop=True)

    # Define metrics to include in specific order (CM first, then metrics)
    metrics_to_include = {
        'TP': 'TP', 'TN': 'TN', 'FP': 'FP', 'FN': 'FN',
        'Accuracy': 'accuracy', 'Precision': 'precision', 'Recall': 'recall', 
        'F1_score': 'f1_score', 'Best_loss': 'best_loss', 'Iterations': 'iterations'
    }
    
    comparison_data = {'Fold': baseline_runs['fold'], 'Run': baseline_runs['run']}
    final_columns = ['Fold', 'Run']
    
    for metric_name_title, metric_key in metrics_to_include.items():
        baseline_col = f'Baseline_{metric_name_title}'
        hybrid_col = f'Hybrid_{metric_name_title}'
        diff_col = f'Difference_{metric_name_title}'
        
        comparison_data[baseline_col] = baseline_runs[metric_key]
        comparison_data[hybrid_col] = hybrid_runs[metric_key]
        comparison_data[diff_col] = hybrid_runs[metric_key] - baseline_runs[metric_key]
        
        final_columns.extend([baseline_col, hybrid_col, diff_col])

    df_paired = pd.DataFrame(comparison_data)
    df_paired = df_paired[final_columns]
    
    csv_path = os.path.join(output_dir, 'cancer_kfold_paired_comparison.csv')
    df_paired.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"Paired comparison CSV saved to: {csv_path}")

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
    bars1 = ax.bar(x - width/2, baseline_means, width, yerr=baseline_stds, label='Baseline ACOR', capsize=5, alpha=0.7, color='skyblue')
    bars2 = ax.bar(x + width/2, hybrid_means, width, yerr=hybrid_stds, label='Hybrid ACOR-LM', capsize=5, alpha=0.7, color='orange')
    
    ax.set_ylabel('Score')
    ax.set_title(f'Cancer Classification - Baseline ACOR vs Hybrid ACOR-LM\n'
                 f'({cv_results["n_folds"]}-fold CV, {cv_results["n_runs_per_fold"]} runs/fold)')
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
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
    try:
        X, y = load_cancer_data()
        print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
        
    print(f"Architecture: 9 inputs, 6 hidden (ReLU), 1 output (Sigmoid)")
    
    # Run k-fold cross-validation
    cv_results = kfold_cross_validation(X, y, n_splits=4, n_runs=50, verbose=True)
    
    # Print results
    print_results(cv_results)
    
    # Perform paired t-test on Accuracy
    t_test_results = perform_paired_t_test(cv_results, metric='accuracy') 
    
    # Save results
    output_dir = os.path.dirname(os.path.abspath(__file__))
    save_results(cv_results, output_dir, t_test_results)
    
    # Save all individual runs to CSV (Stacked format with TP/TN/FP/FN before Accuracy)
    save_results_to_csv(cv_results, output_dir)
    
    # Save the PAIRWISE comparison CSV (Side-by-side format with TP/TN/FP/FN included and placed first)
    save_paired_comparison_csv(cv_results, output_dir)
    
    # Create comparison plot
    create_comparison_plot(cv_results, output_dir)
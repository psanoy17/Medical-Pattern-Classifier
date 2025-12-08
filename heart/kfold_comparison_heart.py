"""
K-Fold Cross-Validation Comparison for Heart Disease Classification

This script compares:
- Baseline ACOR (SOCHA-ACOR)
- Hybrid ACOR-LM (Multiple Colony ACOR with Levenberg-Marquardt local search)

Evaluation:
- 4-fold stratified cross-validation
- 50 independent runs per fold for each algorithm
- FNN Architecture: 35 inputs, 6 hidden (ReLU), 1 output (Sigmoid)
- Binary Cross-Entropy Loss
- Convergence Speed: Iterations to reach threshold (101 penalty for failures)
- Loss Threshold: Computed post-hoc per fold from baseline's average final loss
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
from acor_heart import SOCHA_ACOR, FNN, objective_function

# Import hybrid ACOR-LM components
from lm_local_search import MultipleColonyACOR, LevenbergMarquardt

# Set random seed for reproducibility
np.random.seed(42)


# ==============================================================================
# DATA LOADING
# ==============================================================================
def load_heart_data():
    """Load and return heart disease dataset"""
    data = pd.read_csv(
        os.path.join(os.path.dirname(__file__), 'heart1.dat'),
        sep=' ',
        header=None
    )
    
    X = data.iloc[:, :-2].values  # First 35 columns are features
    y_onehot = data.iloc[:, -2:].values  # Last 2 columns are one-hot encoded target
    y = np.argmax(y_onehot, axis=1)  # Convert to single label
    
    return X, y


# ==============================================================================
# HELPER FUNCTION
# ==============================================================================
def find_iteration_to_threshold(loss_history, threshold):
    """
    Find the iteration when loss first dropped below threshold
    
    Args:
        loss_history: List of (iteration, best_loss) tuples
        threshold: Target loss threshold
        
    Returns:
        Iteration number when threshold was reached, or 101 as penalty if never reached
    """
    for iteration, loss in loss_history:
        if loss < threshold:
            return iteration
    return 101  # Penalty value for runs that never reached threshold


# ==============================================================================
# EVALUATION FUNCTIONS
# ==============================================================================
def evaluate_baseline_acor_with_history(X_train, X_test, y_train, y_test, n_runs=50, verbose=False):
    """
    Evaluate Baseline ACOR (SOCHA-ACOR) and return results WITH loss histories
    for post-hoc threshold analysis.
    
    Args:
        X_train: Training features (already scaled)
        X_test: Test features (already scaled)
        y_train: Training labels
        y_test: Test labels
        n_runs: Number of independent runs
        verbose: Print progress for each run
        
    Returns:
        Dictionary with evaluation results including loss_histories
    """
    results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'confusion_matrices': [],
        'best_losses': [],
        'iterations': [],
        'loss_histories': []  # Store for post-hoc analysis
    }
    
    input_dim = 35
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
            n_samples=230,
            q=0.6,
            xi=0.9,
            max_iter=100,
            patience=15,
            seed=42 + run
        )
        
        best_weights, best_loss, iterations, loss_history = acor.optimize(lb=-3, ub=3)
        
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
        results['loss_histories'].append(loss_history)
        
        if verbose:
            print(f"Acc: {acc:.3f}, Loss: {best_loss:.4f}")
    
    return results


def compute_convergence_speed_posthoc(results, loss_threshold):
    """
    Compute convergence speed (iterations to threshold) post-hoc from stored loss histories.
    
    Args:
        results: Dictionary with 'loss_histories' key
        loss_threshold: Threshold to check against
        
    Returns:
        Tuple of (convergence_speed_list, threshold_reached_list)
    """
    convergence_speed = []
    threshold_reached = []
    
    for loss_history in results['loss_histories']:
        iter_to_thresh = find_iteration_to_threshold(loss_history, loss_threshold)
        convergence_speed.append(iter_to_thresh)
        threshold_reached.append(iter_to_thresh < 101)
    
    return convergence_speed, threshold_reached


def evaluate_hybrid_acor_lm_with_history(X_train, X_test, y_train, y_test, n_runs=50, verbose=False):
    """
    Evaluate Hybrid ACOR-LM and return results WITH loss histories
    for post-hoc threshold analysis.
    
    Args:
        X_train: Training features (already scaled)
        X_test: Test features (already scaled)
        y_train: Training labels
        y_test: Test labels
        n_runs: Number of independent runs
        verbose: Print progress for each run
        
    Returns:
        Dictionary with evaluation results including loss_histories
    """
    results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'confusion_matrices': [],
        'best_losses': [],
        'iterations': [],
        'loss_histories': []  # Store for post-hoc analysis
    }
    
    input_dim = 35
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
        
        # Initialize and run ACOR-LM (without threshold - we'll compute post-hoc)
        acor_lm = MultipleColonyACOR(
            obj_func=obj_func,
            dim=num_weights,
            n_colonies=3,
            n_ants=2,
            n_samples=230,
            q=0.6,
            xi=0.9,
            max_iter=100,
            patience=15,
            local_patience=5,
            sharing_frequency=10,
            sharing_ratio=0.1,
            seed=42 + run
        )
        
        # Run optimization and capture loss history
        best_weights, best_loss, iterations, loss_history = acor_lm.optimize_with_history(
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
        results['loss_histories'].append(loss_history)
        
        if verbose:
            print(f"Acc: {acc:.3f}, Loss: {best_loss:.4f}")
    
    return results


# ==============================================================================
# K-FOLD CROSS-VALIDATION WITH POST-HOC THRESHOLD
# ==============================================================================
def kfold_cross_validation(X, y, n_splits=4, n_runs=50, verbose=True):
    """
    Perform k-fold cross-validation comparing Baseline ACOR vs Hybrid ACOR-LM
    with post-hoc threshold computation per fold.
    
    The loss threshold for each fold is computed as the average final loss
    of the baseline ACOR runs for that fold.
    
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
            'fold_results': {metric: [] for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'convergence_speed', 'losses']},
            'all_results': {metric: [] for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'convergence_speed', 'best_losses', 'threshold_reached']},
            'individual_runs': []
        },
        'hybrid': {
            'fold_results': {metric: [] for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'convergence_speed', 'losses']},
            'all_results': {metric: [] for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'convergence_speed', 'best_losses', 'threshold_reached']},
            'individual_runs': []
        },
        'n_folds': n_splits,
        'n_runs_per_fold': n_runs,
        'fold_thresholds': []  # Store threshold used for each fold
    }
    
    if verbose:
        print(f"\nPerforming {n_splits}-fold cross-validation...")
        print(f"Runs per fold per algorithm: {n_runs}")
        print(f"Loss Threshold: Computed post-hoc per fold from baseline's average final loss")
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
        
        # ==================================================================
        # PHASE 1: Run Baseline ACOR (with loss history tracking)
        # ==================================================================
        if verbose:
            print(f"\n[Baseline ACOR] Running {n_runs} experiments...")
        baseline_results = evaluate_baseline_acor_with_history(
            X_train_fold, X_test_fold, y_train_fold, y_test_fold, 
            n_runs=n_runs, verbose=False
        )
        
        # ==================================================================
        # PHASE 2: Compute fold-specific threshold from baseline results
        # ==================================================================
        fold_threshold = np.mean(baseline_results['best_losses'])
        fold_threshold_std = np.std(baseline_results['best_losses'])
        cv_results['fold_thresholds'].append(fold_threshold)
        
        if verbose:
            print(f"\n  >>> Fold {fold_idx + 1} Loss Threshold (from baseline avg): {fold_threshold:.6f} ± {fold_threshold_std:.6f}")
        
        # ==================================================================
        # PHASE 3: Compute baseline convergence speed post-hoc
        # ==================================================================
        baseline_conv_speed, baseline_thresh_reached = compute_convergence_speed_posthoc(
            baseline_results, fold_threshold
        )
        baseline_results['convergence_speed'] = baseline_conv_speed
        baseline_results['threshold_reached'] = baseline_thresh_reached
        
        # ==================================================================
        # PHASE 4: Run Hybrid ACOR-LM (with loss history tracking)
        # ==================================================================
        if verbose:
            print(f"\n[Hybrid ACOR-LM] Running {n_runs} experiments...")
        hybrid_results = evaluate_hybrid_acor_lm_with_history(
            X_train_fold, X_test_fold, y_train_fold, y_test_fold,
            n_runs=n_runs, verbose=False
        )
        
        # ==================================================================
        # PHASE 5: Compute hybrid convergence speed post-hoc using SAME threshold
        # ==================================================================
        hybrid_conv_speed, hybrid_thresh_reached = compute_convergence_speed_posthoc(
            hybrid_results, fold_threshold
        )
        hybrid_results['convergence_speed'] = hybrid_conv_speed
        hybrid_results['threshold_reached'] = hybrid_thresh_reached
        
        # ==================================================================
        # Store fold averages
        # ==================================================================
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            cv_results['baseline']['fold_results'][metric].append(np.mean(baseline_results[metric]))
            cv_results['hybrid']['fold_results'][metric].append(np.mean(hybrid_results[metric]))
        
        cv_results['baseline']['fold_results']['convergence_speed'].append(np.mean(baseline_conv_speed))
        cv_results['baseline']['fold_results']['losses'].append(np.mean(baseline_results['best_losses']))
        cv_results['hybrid']['fold_results']['convergence_speed'].append(np.mean(hybrid_conv_speed))
        cv_results['hybrid']['fold_results']['losses'].append(np.mean(hybrid_results['best_losses']))
        
        # Store all individual results
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            cv_results['baseline']['all_results'][metric].extend(baseline_results[metric])
            cv_results['hybrid']['all_results'][metric].extend(hybrid_results[metric])
        
        cv_results['baseline']['all_results']['convergence_speed'].extend(baseline_conv_speed)
        cv_results['baseline']['all_results']['best_losses'].extend(baseline_results['best_losses'])
        cv_results['baseline']['all_results']['threshold_reached'].extend(baseline_thresh_reached)
        
        cv_results['hybrid']['all_results']['convergence_speed'].extend(hybrid_conv_speed)
        cv_results['hybrid']['all_results']['best_losses'].extend(hybrid_results['best_losses'])
        cv_results['hybrid']['all_results']['threshold_reached'].extend(hybrid_thresh_reached)
        
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
                'convergence_speed': baseline_conv_speed[run_idx],
                'best_loss': baseline_results['best_losses'][run_idx],
                'threshold_reached': baseline_thresh_reached[run_idx],
                'fold_threshold': fold_threshold
            })
            cv_results['hybrid']['individual_runs'].append({
                'fold': fold_idx + 1,
                'run': run_idx + 1,
                'algorithm': 'Hybrid ACOR-LM',
                'accuracy': hybrid_results['accuracy'][run_idx],
                'precision': hybrid_results['precision'][run_idx],
                'recall': hybrid_results['recall'][run_idx],
                'f1_score': hybrid_results['f1_score'][run_idx],
                'convergence_speed': hybrid_conv_speed[run_idx],
                'best_loss': hybrid_results['best_losses'][run_idx],
                'threshold_reached': hybrid_thresh_reached[run_idx],
                'fold_threshold': fold_threshold
            })
        
        # Compute success rates for this fold
        baseline_success_rate = sum(baseline_thresh_reached) / n_runs * 100
        hybrid_success_rate = sum(hybrid_thresh_reached) / n_runs * 100
        
        # Print fold summary
        if verbose:
            print(f"\nFold {fold_idx + 1} Summary (Threshold: {fold_threshold:.6f}):")
            print(f"  Baseline:  Acc={np.mean(baseline_results['accuracy']):.4f}, "
                  f"F1={np.mean(baseline_results['f1_score']):.4f}, "
                  f"ConvSpeed={np.mean(baseline_conv_speed):.1f}, "
                  f"SuccessRate={baseline_success_rate:.1f}%")
            print(f"  Hybrid:    Acc={np.mean(hybrid_results['accuracy']):.4f}, "
                  f"F1={np.mean(hybrid_results['f1_score']):.4f}, "
                  f"ConvSpeed={np.mean(hybrid_conv_speed):.1f}, "
                  f"SuccessRate={hybrid_success_rate:.1f}%")
    
    return cv_results


# ==============================================================================
# RESULTS REPORTING
# ==============================================================================
def print_results(cv_results):
    """Print formatted cross-validation results"""
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION RESULTS - PER-FOLD AVERAGES")
    print("=" * 70)
    print("Loss Threshold: Computed post-hoc per fold from baseline's average final loss")
    print("Penalty for runs not reaching threshold: 101")
    
    n_folds = cv_results['n_folds']
    
    # Print per-fold thresholds
    print("\n--- Per-Fold Loss Thresholds ---")
    for fold_idx in range(n_folds):
        print(f"  Fold {fold_idx + 1}: {cv_results['fold_thresholds'][fold_idx]:.6f}")
    print(f"  Average: {np.mean(cv_results['fold_thresholds']):.6f} ± {np.std(cv_results['fold_thresholds']):.6f}")
    
    print("\n--- Per-Fold Results ---")
    for fold_idx in range(n_folds):
        threshold = cv_results['fold_thresholds'][fold_idx]
        print(f"\nFold {fold_idx + 1} (Threshold: {threshold:.6f}):")
        for algo in ['baseline', 'hybrid']:
            algo_name = "Baseline ACOR" if algo == 'baseline' else "Hybrid ACOR-LM"
            fr = cv_results[algo]['fold_results']
            print(f"  {algo_name:15}: Acc={fr['accuracy'][fold_idx]:.4f}, "
                  f"Prec={fr['precision'][fold_idx]:.4f}, "
                  f"Rec={fr['recall'][fold_idx]:.4f}, "
                  f"F1={fr['f1_score'][fold_idx]:.4f}, "
                  f"ConvSpeed={fr['convergence_speed'][fold_idx]:.1f}")
    
    print("\n" + "=" * 70)
    print("OVERALL RESULTS (Mean ± Std across all folds and runs)")
    print("=" * 70)
    
    for algo in ['baseline', 'hybrid']:
        algo_name = "Baseline ACOR" if algo == 'baseline' else "Hybrid ACOR-LM"
        ar = cv_results[algo]['all_results']
        
        # Calculate success rate
        success_count = sum(ar['threshold_reached'])
        total_runs = len(ar['threshold_reached'])
        success_rate = success_count / total_runs * 100
        
        print(f"\n{algo_name}:")
        print("-" * 60)
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            mean_val = np.mean(ar[metric])
            std_val = np.std(ar[metric])
            print(f"  {metric.capitalize():18}: {mean_val:.4f} ± {std_val:.4f}")
        
        # Convergence speed (with penalty)
        mean_conv = np.mean(ar['convergence_speed'])
        std_conv = np.std(ar['convergence_speed'])
        print(f"  {'Convergence Speed':18}: {mean_conv:.1f} ± {std_conv:.1f} (with penalty=101)")
        print(f"  {'Success Rate':18}: {success_rate:.1f}% ({success_count}/{total_runs} reached threshold)")
        print(f"  {'Loss':18}: {np.mean(ar['best_losses']):.6f} ± {np.std(ar['best_losses']):.6f}")


def save_results(cv_results, output_dir):
    """Save results to text file"""
    txt_path = os.path.join(output_dir, 'heart_kfold_comparison_results.txt')
    with open(txt_path, 'w') as f:
        f.write("Heart Disease Classification - Baseline ACOR vs Hybrid ACOR-LM\n")
        f.write("K-Fold Cross-Validation Comparison\n")
        f.write("=" * 70 + "\n")
        f.write(f"Architecture: 35-6-1 (Total weights: 223)\n")
        f.write(f"Number of folds: {cv_results['n_folds']}\n")
        f.write(f"Runs per fold per algorithm: {cv_results['n_runs_per_fold']}\n")
        f.write(f"Loss Threshold: Computed post-hoc per fold from baseline's average final loss\n")
        f.write(f"Penalty for runs not reaching threshold: 101\n\n")
        
        # Per-fold thresholds
        f.write("PER-FOLD LOSS THRESHOLDS\n")
        f.write("=" * 70 + "\n")
        for fold_idx in range(cv_results['n_folds']):
            f.write(f"  Fold {fold_idx + 1}: {cv_results['fold_thresholds'][fold_idx]:.6f}\n")
        f.write(f"  Average: {np.mean(cv_results['fold_thresholds']):.6f} ± {np.std(cv_results['fold_thresholds']):.6f}\n")
        
        f.write("\nPER-FOLD RESULTS\n")
        f.write("=" * 70 + "\n")
        
        for fold_idx in range(cv_results['n_folds']):
            threshold = cv_results['fold_thresholds'][fold_idx]
            f.write(f"\nFold {fold_idx + 1} (Threshold: {threshold:.6f}):\n")
            for algo in ['baseline', 'hybrid']:
                algo_name = "Baseline ACOR" if algo == 'baseline' else "Hybrid ACOR-LM"
                fr = cv_results[algo]['fold_results']
                f.write(f"  {algo_name}:\n")
                f.write(f"    Accuracy:          {fr['accuracy'][fold_idx]:.4f}\n")
                f.write(f"    Precision:         {fr['precision'][fold_idx]:.4f}\n")
                f.write(f"    Recall:            {fr['recall'][fold_idx]:.4f}\n")
                f.write(f"    F1-Score:          {fr['f1_score'][fold_idx]:.4f}\n")
                f.write(f"    Convergence Speed: {fr['convergence_speed'][fold_idx]:.1f}\n")
                f.write(f"    Loss:              {fr['losses'][fold_idx]:.6f}\n")
        
        f.write("\n\nOVERALL RESULTS (Mean ± Std)\n")
        f.write("=" * 70 + "\n")
        
        for algo in ['baseline', 'hybrid']:
            algo_name = "Baseline ACOR" if algo == 'baseline' else "Hybrid ACOR-LM"
            ar = cv_results[algo]['all_results']
            
            # Calculate success rate
            success_count = sum(ar['threshold_reached'])
            total_runs = len(ar['threshold_reached'])
            success_rate = success_count / total_runs * 100
            
            f.write(f"\n{algo_name}:\n")
            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                mean_val = np.mean(ar[metric])
                std_val = np.std(ar[metric])
                f.write(f"  {metric.capitalize()}: {mean_val:.4f} ± {std_val:.4f}\n")
            f.write(f"  Convergence Speed: {np.mean(ar['convergence_speed']):.1f} ± {np.std(ar['convergence_speed']):.1f}\n")
            f.write(f"  Success Rate: {success_rate:.1f}% ({success_count}/{total_runs})\n")
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
                    'f1_score', 'convergence_speed', 'best_loss', 'threshold_reached', 'fold_threshold']
    df = df[column_order]
    
    # Sort by fold, then algorithm, then run
    df = df.sort_values(['fold', 'algorithm', 'run']).reset_index(drop=True)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'heart_kfold_all_runs.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"All individual runs saved to CSV: {csv_path}")
    print(f"Total runs: {len(df)} ({cv_results['n_folds']} folds × {cv_results['n_runs_per_fold']} runs × 2 algorithms)")
    print(f"Note: convergence_speed = iterations to reach fold-specific threshold (101 = penalty)")
    print(f"      fold_threshold = loss threshold used for that fold (from baseline avg)")


def create_comparison_plot(cv_results, output_dir):
    """Create and save comparison bar plot"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Performance metrics
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    x = np.arange(len(metrics))
    width = 0.35
    
    baseline_ar = cv_results['baseline']['all_results']
    hybrid_ar = cv_results['hybrid']['all_results']
    
    baseline_means = [np.mean(baseline_ar[m]) for m in metrics]
    baseline_stds = [np.std(baseline_ar[m]) for m in metrics]
    hybrid_means = [np.mean(hybrid_ar[m]) for m in metrics]
    hybrid_stds = [np.std(hybrid_ar[m]) for m in metrics]
    
    bars1 = axes[0].bar(x - width/2, baseline_means, width, yerr=baseline_stds,
                        label='Baseline ACOR', capsize=5, alpha=0.7, color='skyblue')
    bars2 = axes[0].bar(x + width/2, hybrid_means, width, yerr=hybrid_stds,
                        label='Hybrid ACOR-LM', capsize=5, alpha=0.7, color='orange')
    
    axes[0].set_ylabel('Score')
    axes[0].set_title('Performance Metrics')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([m.capitalize() for m in metrics])
    axes[0].legend()
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Convergence Speed comparison
    x2 = np.arange(1)
    
    baseline_conv_mean = np.mean(baseline_ar['convergence_speed'])
    baseline_conv_std = np.std(baseline_ar['convergence_speed'])
    hybrid_conv_mean = np.mean(hybrid_ar['convergence_speed'])
    hybrid_conv_std = np.std(hybrid_ar['convergence_speed'])
    
    bars3 = axes[1].bar(x2 - width/2, [baseline_conv_mean], width, yerr=[baseline_conv_std],
                        label='Baseline ACOR', capsize=5, alpha=0.7, color='skyblue')
    bars4 = axes[1].bar(x2 + width/2, [hybrid_conv_mean], width, yerr=[hybrid_conv_std],
                        label='Hybrid ACOR-LM', capsize=5, alpha=0.7, color='orange')
    
    axes[1].set_ylabel('Iterations to Threshold')
    avg_threshold = np.mean(cv_results['fold_thresholds'])
    axes[1].set_title(f'Convergence Speed\n(Avg Threshold: {avg_threshold:.6f}, Penalty: 101)')
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(['Convergence Speed'])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Add value labels for convergence speed
    for bar, mean, std in [(bars3[0], baseline_conv_mean, baseline_conv_std), 
                            (bars4[0], hybrid_conv_mean, hybrid_conv_std)]:
        axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 1,
                    f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Add success rate annotations
    baseline_success = sum(baseline_ar['threshold_reached']) / len(baseline_ar['threshold_reached']) * 100
    hybrid_success = sum(hybrid_ar['threshold_reached']) / len(hybrid_ar['threshold_reached']) * 100
    axes[1].text(0.5, 0.95, f'Success Rate: Baseline={baseline_success:.1f}%, Hybrid={hybrid_success:.1f}%',
                transform=axes[1].transAxes, ha='center', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Heart Disease Classification - Baseline ACOR vs Hybrid ACOR-LM\n'
                 f'({cv_results["n_folds"]}-fold CV, {cv_results["n_runs_per_fold"]} runs/fold, Post-hoc Threshold per Fold)',
                 fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'heart_kfold_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved to: {plot_path}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print("Heart Disease Classification - K-Fold Cross-Validation Comparison")
    print("Baseline ACOR vs Hybrid ACOR-LM")
    print("=" * 70)
    
    # Load data
    X, y = load_heart_data()
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target distribution: Class 0: {np.sum(y==0)}, Class 1: {np.sum(y==1)}")
    print(f"Architecture: 35 inputs, 6 hidden (ReLU), 1 output (Sigmoid)")
    print(f"Total weights: {FNN.get_num_weights(35, 6, 1)}")
    print(f"Loss Threshold: Computed post-hoc per fold from baseline's average final loss")
    print(f"Penalty for runs not reaching threshold: 101")
    
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
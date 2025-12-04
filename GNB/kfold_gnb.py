"""
K-Fold Cross-Validation for Gaussian Naive Bayes (GNB) Model

This script evaluates the custom MyGaussianNB model using:
- 4-fold stratified cross-validation
- 50 "runs" per fold (GNB is deterministic, so results are repeated for consistency)
- Matches the evaluation methodology used for ACOR models

This ensures a fair apples-to-apples comparison with:
- Baseline ACOR (SOCHA-ACOR)
- Hybrid ACOR-LM

Note: GNB is deterministic, so running it 50 times produces identical results.
      We repeat the results 50 times to maintain array consistency for statistical tests.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the custom GNB model from train_gnb.py
from train_gnb import MyGaussianNB, load_proben1_dat

# Set random seed for reproducibility (for fold generation)
np.random.seed(42)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset configurations matching train_gnb.py
DATASETS = {
    "cancer": {
        "file": "cancer1.dat",
        "n_inputs": 9,
        "n_outputs": 2,
        "label_names": ["Benign", "Malignant"]
    },
    "heart": {
        "file": "heart1.dat",
        "n_inputs": 35,
        "n_outputs": 2,
        "label_names": ["Negative", "Positive"]
    },
    "diabetes": {
        "file": "diabetes1.dat",
        "n_inputs": 8,
        "n_outputs": 2,
        "label_names": ["Non-Diabetic", "Diabetic"]
    }
}


# ==============================================================================
# DATA LOADING
# ==============================================================================
def load_dataset(dataset_name):
    """
    Load dataset using the same method as train_gnb.py
    
    Args:
        dataset_name: One of 'cancer', 'heart', or 'diabetes'
        
    Returns:
        X: Feature matrix
        y: Target vector
        config: Dataset configuration dictionary
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(DATASETS.keys())}")
    
    config = DATASETS[dataset_name]
    
    # Try to find the dataset file
    candidates = [
        os.path.join(BASE_DIR, config["file"]),
        os.path.join(BASE_DIR, "datasets", config["file"]),
    ]
    
    dataset_path = None
    for p in candidates:
        if os.path.exists(p):
            dataset_path = p
            break
    
    if dataset_path is None:
        raise FileNotFoundError(f"Dataset file not found. Tried:\n" + '\n'.join(candidates))
    
    X, y = load_proben1_dat(dataset_path, config["n_inputs"], config["n_outputs"])
    
    return X, y, config


# ==============================================================================
# EVALUATION FUNCTION
# ==============================================================================
def evaluate_gnb_single_fold(X_train, X_test, y_train, y_test, n_runs=50):
    """
    Evaluate GNB on a single fold.
    
    Since GNB is deterministic, we train once and repeat the results n_runs times
    to maintain consistency with the ACO evaluation methodology.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        n_runs: Number of times to repeat results (for consistency with ACO)
        
    Returns:
        Dictionary with evaluation results (each metric is a list of n_runs identical values)
    """
    # Train GNB model once (deterministic)
    model = MyGaussianNB()
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate metrics (single result)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    # Repeat results n_runs times for consistency with ACO methodology
    # This ensures arrays have the same length for statistical tests
    results = {
        'accuracy': [acc] * n_runs,
        'precision': [prec] * n_runs,
        'recall': [rec] * n_runs,
        'f1_score': [f1] * n_runs,
        'confusion_matrices': [cm] * n_runs,
    }
    
    return results


# ==============================================================================
# K-FOLD CROSS-VALIDATION
# ==============================================================================
def kfold_cross_validation_gnb(dataset_name, n_splits=4, n_runs=50, verbose=True):
    """
    Perform k-fold cross-validation for GNB model.
    
    Args:
        dataset_name: One of 'cancer', 'heart', or 'diabetes'
        n_splits: Number of folds (default: 4)
        n_runs: Number of "runs" per fold (results are repeated for consistency)
        verbose: Print progress
        
    Returns:
        Dictionary with cross-validation results
    """
    # Load data
    X, y, config = load_dataset(dataset_name)
    
    if verbose:
        print(f"\nDataset: {dataset_name.upper()}")
        print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
        print(f"Class distribution: {dict(zip(config['label_names'], np.bincount(y)))}")
    
    # Initialize k-fold splitter
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Initialize result containers
    cv_results = {
        'dataset': dataset_name,
        'n_folds': n_splits,
        'n_runs_per_fold': n_runs,
        'label_names': config['label_names'],
        'fold_results': {
            metric: [] for metric in ['accuracy', 'precision', 'recall', 'f1_score']
        },
        'all_results': {
            metric: [] for metric in ['accuracy', 'precision', 'recall', 'f1_score']
        },
        'individual_runs': [],
        'confusion_matrices_per_fold': []
    }
    
    if verbose:
        print(f"\nPerforming {n_splits}-fold cross-validation...")
        print(f"Runs per fold: {n_runs} (repeated for consistency with ACO)")
        print("=" * 70)
    
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        if verbose:
            print(f"\n{'='*70}")
            print(f"FOLD {fold_idx + 1}/{n_splits}")
            print(f"{'='*70}")
            print(f"Training samples: {len(train_idx)}, Test samples: {len(test_idx)}")
        
        # Split data for this fold (no scaling needed for GNB - data already normalized)
        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]
        
        # Print class distribution per fold
        if verbose:
            train_dist = np.bincount(y_train_fold, minlength=len(config['label_names']))
            test_dist = np.bincount(y_test_fold, minlength=len(config['label_names']))
            print(f"Train class distribution: {dict(zip(config['label_names'], train_dist))}")
            print(f"Test class distribution: {dict(zip(config['label_names'], test_dist))}")
        
        # Evaluate GNB
        if verbose:
            print(f"\n[GNB] Training and evaluating (deterministic, results repeated {n_runs}x)...")
        
        fold_results = evaluate_gnb_single_fold(
            X_train_fold, X_test_fold, y_train_fold, y_test_fold,
            n_runs=n_runs
        )
        
        # Store fold average (since GNB is deterministic, all n_runs are identical)
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            cv_results['fold_results'][metric].append(fold_results[metric][0])  # All identical
        
        # Store all individual results
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            cv_results['all_results'][metric].extend(fold_results[metric])
        
        # Store confusion matrix for this fold
        cv_results['confusion_matrices_per_fold'].append(fold_results['confusion_matrices'][0])
        
        # Store individual run data
        for run_idx in range(n_runs):
            cv_results['individual_runs'].append({
                'fold': fold_idx + 1,
                'run': run_idx + 1,
                'accuracy': fold_results['accuracy'][run_idx],
                'precision': fold_results['precision'][run_idx],
                'recall': fold_results['recall'][run_idx],
                'f1_score': fold_results['f1_score'][run_idx],
            })
        
        # Print fold summary
        if verbose:
            print(f"\nFold {fold_idx + 1} Results:")
            print(f"  Accuracy:  {fold_results['accuracy'][0]:.4f}")
            print(f"  Precision: {fold_results['precision'][0]:.4f}")
            print(f"  Recall:    {fold_results['recall'][0]:.4f}")
            print(f"  F1-Score:  {fold_results['f1_score'][0]:.4f}")
            print(f"\n  Confusion Matrix:")
            cm = fold_results['confusion_matrices'][0]
            for i, label in enumerate(config['label_names']):
                print(f"    {label}: {cm[i]}")
    
    return cv_results


# ==============================================================================
# RESULTS REPORTING
# ==============================================================================
def print_results(cv_results):
    """Print formatted cross-validation results"""
    print("\n" + "=" * 70)
    print(f"CROSS-VALIDATION RESULTS - {cv_results['dataset'].upper()}")
    print("=" * 70)
    
    n_folds = cv_results['n_folds']
    
    print("\nPER-FOLD RESULTS:")
    print("-" * 60)
    
    for fold_idx in range(n_folds):
        fr = cv_results['fold_results']
        print(f"  Fold {fold_idx + 1}: Acc={fr['accuracy'][fold_idx]:.4f}, "
              f"Prec={fr['precision'][fold_idx]:.4f}, "
              f"Rec={fr['recall'][fold_idx]:.4f}, "
              f"F1={fr['f1_score'][fold_idx]:.4f}")
    
    print("\n" + "=" * 70)
    print("OVERALL RESULTS (Mean ± Std across all folds)")
    print("=" * 70)
    
    ar = cv_results['all_results']
    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        mean_val = np.mean(ar[metric])
        std_val = np.std(ar[metric])
        print(f"  {metric.capitalize():12}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Note about deterministic nature
    print("\nNote: GNB is deterministic, so std within each fold is 0.")
    print(f"      The std shown reflects variation across {n_folds} folds only.")
    
    # Print average confusion matrix
    print("\nAverage Confusion Matrix:")
    avg_cm = np.mean(cv_results['confusion_matrices_per_fold'], axis=0)
    for i, label in enumerate(cv_results['label_names']):
        print(f"  {label}: {avg_cm[i]}")


def save_results_to_txt(cv_results, output_dir):
    """Save results to text file"""
    txt_path = os.path.join(output_dir, f"{cv_results['dataset']}_gnb_kfold_results.txt")
    
    with open(txt_path, 'w') as f:
        f.write(f"{cv_results['dataset'].upper()} Classification - Gaussian Naive Bayes\n")
        f.write("K-Fold Cross-Validation Results\n")
        f.write("=" * 70 + "\n")
        f.write(f"Number of folds: {cv_results['n_folds']}\n")
        f.write(f"Runs per fold: {cv_results['n_runs_per_fold']} (repeated for consistency)\n")
        f.write(f"Note: GNB is deterministic; all runs within a fold are identical.\n\n")
        
        f.write("PER-FOLD RESULTS\n")
        f.write("=" * 70 + "\n")
        
        for fold_idx in range(cv_results['n_folds']):
            fr = cv_results['fold_results']
            f.write(f"\nFold {fold_idx + 1}:\n")
            f.write(f"  Accuracy:   {fr['accuracy'][fold_idx]:.4f}\n")
            f.write(f"  Precision:  {fr['precision'][fold_idx]:.4f}\n")
            f.write(f"  Recall:     {fr['recall'][fold_idx]:.4f}\n")
            f.write(f"  F1-Score:   {fr['f1_score'][fold_idx]:.4f}\n")
            
            # Add confusion matrix for this fold
            f.write(f"  Confusion Matrix:\n")
            cm = cv_results['confusion_matrices_per_fold'][fold_idx]
            for i, label in enumerate(cv_results['label_names']):
                f.write(f"    {label}: {cm[i]}\n")
        
        f.write("\n\nOVERALL RESULTS (Mean ± Std across all folds)\n")
        f.write("=" * 70 + "\n")
        
        ar = cv_results['all_results']
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            mean_val = np.mean(ar[metric])
            std_val = np.std(ar[metric])
            f.write(f"  {metric.capitalize()}: {mean_val:.4f} ± {std_val:.4f}\n")
        
        # Add average confusion matrix
        f.write("\nAverage Confusion Matrix:\n")
        avg_cm = np.mean(cv_results['confusion_matrices_per_fold'], axis=0)
        for i, label in enumerate(cv_results['label_names']):
            f.write(f"  {label}: {avg_cm[i]}\n")
    
    print(f"Results saved to: {txt_path}")


def save_results_to_csv(cv_results, output_dir):
    """Save all individual run results to CSV file"""
    df = pd.DataFrame(cv_results['individual_runs'])
    
    # Reorder columns for better readability
    column_order = ['fold', 'run', 'accuracy', 'precision', 'recall', 'f1_score']
    df = df[column_order]
    
    # Sort by fold, then run
    df = df.sort_values(['fold', 'run']).reset_index(drop=True)
    
    csv_path = os.path.join(output_dir, f"{cv_results['dataset']}_gnb_kfold_all_runs.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"All individual runs saved to CSV: {csv_path}")
    print(f"Total entries: {len(df)} ({cv_results['n_folds']} folds × {cv_results['n_runs_per_fold']} runs)")


def create_performance_plot(cv_results, output_dir):
    """Create and save performance bar plot"""
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    ar = cv_results['all_results']
    means = [np.mean(ar[m]) for m in metrics]
    stds = [np.std(ar[m]) for m in metrics]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(metrics, means, yerr=stds, capsize=5,
                  color=['skyblue', 'orange', 'green', 'red'], alpha=0.7)
    
    ax.set_ylabel('Score')
    ax.set_title(f'{cv_results["dataset"].upper()} Classification - Gaussian Naive Bayes\n'
                 f'({cv_results["n_folds"]}-fold Cross-Validation)')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.02,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{cv_results['dataset']}_gnb_kfold_performance.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved to: {plot_path}")


# ==============================================================================
# RUN ALL DATASETS
# ==============================================================================
def run_all_datasets(n_splits=4, n_runs=50, verbose=True):
    """
    Run k-fold cross-validation for all three datasets.
    
    Args:
        n_splits: Number of folds
        n_runs: Number of runs per fold (repeated for consistency)
        verbose: Print progress
        
    Returns:
        Dictionary with results for all datasets
    """
    all_results = {}
    output_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)
    
    for dataset_name in ["cancer", "heart", "diabetes"]:
        print("\n" + "=" * 70)
        print(f"PROCESSING DATASET: {dataset_name.upper()}")
        print("=" * 70)
        
        try:
            cv_results = kfold_cross_validation_gnb(
                dataset_name, n_splits=n_splits, n_runs=n_runs, verbose=verbose
            )
            
            # Print results
            print_results(cv_results)
            
            # Save results
            save_results_to_txt(cv_results, output_dir)
            save_results_to_csv(cv_results, output_dir)
            create_performance_plot(cv_results, output_dir)
            
            all_results[dataset_name] = cv_results
            
        except FileNotFoundError as e:
            print(f"Warning: Could not process {dataset_name}: {e}")
    
    return all_results


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print("Gaussian Naive Bayes - K-Fold Cross-Validation")
    print("=" * 70)
    print("This script evaluates the custom MyGaussianNB model using")
    print("4-fold stratified cross-validation with 50 'runs' per fold.")
    print("\nNote: GNB is deterministic, so results within each fold are identical.")
    print("      Results are repeated 50 times for consistency with ACO evaluation.")
    print("=" * 70)
    
    # Ask user which dataset to process
    print("\nSelect dataset:")
    print("  [0] Cancer")
    print("  [1] Heart")
    print("  [2] Diabetes")
    print("  [3] All datasets")
    
    choice = input("Enter choice (0-3): ").strip()
    
    output_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)
    
    if choice == "3":
        # Run all datasets
        all_results = run_all_datasets(n_splits=4, n_runs=50, verbose=True)
        
        # Print summary comparison
        print("\n" + "=" * 70)
        print("SUMMARY ACROSS ALL DATASETS")
        print("=" * 70)
        
        for dataset_name, cv_results in all_results.items():
            ar = cv_results['all_results']
            print(f"\n{dataset_name.upper()}:")
            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                mean_val = np.mean(ar[metric])
                std_val = np.std(ar[metric])
                print(f"  {metric.capitalize():12}: {mean_val:.4f} ± {std_val:.4f}")
    else:
        # Run single dataset
        dataset_map = {"0": "cancer", "1": "heart", "2": "diabetes"}
        if choice not in dataset_map:
            print("Invalid choice. Defaulting to cancer dataset.")
            choice = "0"
        
        dataset_name = dataset_map[choice]
        
        cv_results = kfold_cross_validation_gnb(
            dataset_name, n_splits=4, n_runs=50, verbose=True
        )
        
        # Print and save results
        print_results(cv_results)
        save_results_to_txt(cv_results, output_dir)
        save_results_to_csv(cv_results, output_dir)
        create_performance_plot(cv_results, output_dir)
    
    print("\n" + "=" * 70)
    print("K-FOLD CROSS-VALIDATION COMPLETE")
    print("=" * 70)
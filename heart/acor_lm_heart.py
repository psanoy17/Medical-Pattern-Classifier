"""
ACOR-LM with Multiple Colonies for Heart Disease Classification - 4-Fold Cross-Validation

This implementation compares:
- Baseline ACOR (SOCHA-ACOR)
- Hybrid ACOR-LM (Multiple Colony ACOR with Levenberg-Marquardt local search)

Evaluation:
- 4-fold stratified cross-validation
- 50 independent runs per fold for each algorithm
- FNN Architecture: 35 inputs, 6 hidden (ReLU), 1 output (Sigmoid)
- Binary Cross-Entropy Loss
- Uses preprocessed heart1.dat with 35 features
"""

import numpy as np
import pandas as pd
import os
import sys
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
import matplotlib.pyplot as plt
import pickle

# Add parent directory to path to import lm_local_search
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from lm_local_search import MultipleColonyACOR, LevenbergMarquardt

# Import baseline ACOR from acor_heart.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from acor_heart import SOCHA_ACOR, FNN, objective_function as baseline_objective_function

# Set random seed for reproducibility
np.random.seed(42)

# 1. Load and preprocess the data from heart1.dat
# --------------------------------------------------
# Load the preprocessed data (space-separated, 35 features + 2 one-hot encoded target)
data = pd.read_csv(
    os.path.join(os.path.dirname(__file__), 'heart1.dat'),
    sep=' ',
    header=None
)

# The last TWO columns (36 and 37) are the one-hot encoded target [class_0, class_1]
X = data.iloc[:, :-2].values  # First 35 columns are features
y_onehot = data.iloc[:, -2:].values   # Last 2 columns are one-hot encoded target

# Convert one-hot encoding back to single label (0 or 1)
y = np.argmax(y_onehot, axis=1)

print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Target distribution: Class 0: {np.sum(y==0)}, Class 1: {np.sum(y==1)}")

# Note: Standardization will be done per fold to avoid data leakage
# Initialize 4-fold cross-validation
kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

print(f"Using 4-fold cross-validation")
print(f"Total samples: {len(X)}")

# 2. Define FNN matching thesis specifications (35, 6, 1)
# --------------------------------------------------
class FNN_Thesis:
    """
    Feedforward Neural Network matching thesis specifications.
    Architecture: Input(35) -> Hidden(6, ReLU) -> Output(1, Sigmoid)
    Total weights: 35*6 + 6 + 6*1 + 1 = 223 weights
    """
    def __init__(self, input_dim=35, hidden_dim=6, output_dim=1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.total_weights = input_dim * hidden_dim + hidden_dim + hidden_dim * output_dim + output_dim

    def set_weights(self, weights):
        """Set weights from flat vector matching thesis structure"""
        if len(weights) != self.total_weights:
            raise ValueError(f"Expected {self.total_weights} weights, got {len(weights)}")
        
        idx = 0
        # Input to hidden weights
        self.W1 = weights[idx:idx + self.input_dim * self.hidden_dim].reshape(self.input_dim, self.hidden_dim)
        idx += self.input_dim * self.hidden_dim
        
        # Hidden bias
        self.b1 = weights[idx:idx + self.hidden_dim]
        idx += self.hidden_dim
        
        # Hidden to output weights
        self.W2 = weights[idx:idx + self.hidden_dim * self.output_dim].reshape(self.hidden_dim, self.output_dim)
        idx += self.hidden_dim * self.output_dim
        
        # Output bias
        self.b2 = weights[idx:idx + self.output_dim]

    def _stable_sigmoid(self, z):
        """Numerically stable sigmoid function that prevents overflow"""
        # Clip values to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        """Forward pass with ReLU and Sigmoid activations"""
        # Hidden layer with ReLU activation
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0, z1)  # ReLU activation
        
        # Output layer with Sigmoid activation (numerically stable)
        z2 = a1 @ self.W2 + self.b2
        a2 = self._stable_sigmoid(z2)
        
        return a2.squeeze()

    def predict(self, X):
        """Predict class based on output threshold"""
        output = self.forward(X)
        return (output > 0.5).astype(int)

    @staticmethod
    def get_num_weights(input_dim=35, hidden_dim=6, output_dim=1):
        return input_dim * hidden_dim + hidden_dim + hidden_dim * output_dim + output_dim

# 3. Objective function for ACOR-LM (Binary Cross-Entropy Loss)
# --------------------------------------------------
def objective_function(weights, model, X_train, y_train):
    """Binary Cross-Entropy Loss as fitness function"""
    model.set_weights(weights)
    y_pred = model.forward(X_train)
    eps = 1e-8
    loss = -np.mean(y_train * np.log(y_pred + eps) + (1 - y_train) * np.log(1 - y_pred + eps))
    return loss

# 4. Baseline ACOR evaluation function
# --------------------------------------------------
def evaluate_baseline_acor(X_train, X_test, y_train, y_test, n_runs=50):
    """
    Evaluate Baseline ACOR (SOCHA-ACOR) using 50 independent runs
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        n_runs: Number of independent runs
        
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
    
    input_dim = 35
    hidden_dim = 6
    output_dim = 1
    num_weights = FNN.get_num_weights(input_dim, hidden_dim, output_dim)
    
    for run in range(n_runs):
        # Initialize model
        model = FNN(input_dim, hidden_dim, output_dim)
        
        # Create objective function
        def obj_func(weights):
            if weights.ndim == 1:
                return baseline_objective_function(weights, model, X_train, y_train)
            else:
                losses = []
                for w in weights:
                    losses.append(baseline_objective_function(w, model, X_train, y_train))
                return np.array(losses)
        
        # Initialize Baseline ACOR
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
        
        # Optimize
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
        
        # Store results
        results['accuracy'].append(acc)
        results['precision'].append(prec)
        results['recall'].append(rec)
        results['f1_score'].append(f1)
        results['confusion_matrices'].append(cm)
        results['best_losses'].append(best_loss)
        results['iterations'].append(iterations)
    
    return results

# 5. Hybrid ACOR-LM evaluation function
# --------------------------------------------------
def evaluate_acor_lm(X_train, X_test, y_train, y_test, n_runs=50):
    """
    Evaluate ACOR-LM with multiple colonies using 50 independent runs
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        n_runs: Number of independent runs
        
    Returns:
        Dictionary with evaluation results and best model
    """
    # Initialize results storage
    results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'confusion_matrices': [],
        'best_losses': [],
        'iterations': []
    }
    
    # Track best model across all runs
    best_overall_accuracy = -1
    best_model_weights = None
    best_run_idx = -1
    
    print(f"\nRunning {n_runs} independent experiments...")
    print("=" * 60)
    
    for run in range(n_runs):
        print(f"Run {run + 1}/{n_runs}", end=" ")
        
        # Initialize model
        model = FNN_Thesis(input_dim=35, hidden_dim=6, output_dim=1)
        num_weights = FNN_Thesis.get_num_weights(35, 6, 1)
        
        # Create objective function
        def obj_func(weights):
            return objective_function(weights, model, X_train, y_train)
        
        # Initialize Multiple Colony ACOR-LM
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
            sharing_frequency=10,
            sharing_ratio=0.1,
            seed=42 + run  # Different seed for each run
        )
        
        # Optimize
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
        
        # Store results
        results['accuracy'].append(acc)
        results['precision'].append(prec)
        results['recall'].append(rec)
        results['f1_score'].append(f1)
        results['confusion_matrices'].append(cm)
        results['best_losses'].append(best_loss)
        results['iterations'].append(iterations)
        
        # Track best model
        if acc > best_overall_accuracy:
            best_overall_accuracy = acc
            best_model_weights = best_weights.copy()
            best_run_idx = run
        
        print(f"Acc: {acc:.3f}, Prec: {prec:.3f}, Rec: {rec:.3f}, F1: {f1:.3f}, Loss: {best_loss:.3f}")
    
    # Add best model info to results
    results['best_model_weights'] = best_model_weights
    results['best_run_index'] = best_run_idx
    results['best_overall_accuracy'] = best_overall_accuracy
    
    return results

# 6. 4-Fold Cross-Validation with Both Algorithms
# --------------------------------------------------
def kfold_cross_validation_comparison(X, y, n_splits=4, n_runs=50):
    """
    Perform 4-fold cross-validation comparing Baseline ACOR vs Hybrid ACOR-LM
    
    For each fold:
    - For each run (1-50):
      - Train and test Baseline ACOR
      - Train and test Hybrid ACOR-LM
    - Calculate averages for the fold
    
    Then calculate overall averages across all folds.
    
    Args:
        X: Full feature matrix
        y: Full target vector
        n_splits: Number of folds (default: 4)
        n_runs: Number of independent runs per fold
        
    Returns:
        Dictionary with cross-validation results for both algorithms
    """
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Store results for Baseline ACOR
    baseline_fold_results = {
        'fold_accuracy': [],
        'fold_precision': [],
        'fold_recall': [],
        'fold_f1_score': [],
        'fold_iterations': [],
        'fold_losses': []
    }
    
    # Store results for Hybrid ACOR-LM
    hybrid_fold_results = {
        'fold_accuracy': [],
        'fold_precision': [],
        'fold_recall': [],
        'fold_f1_score': [],
        'fold_iterations': [],
        'fold_losses': []
    }
    
    # Store all individual run results across folds
    baseline_all_results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'iterations': [],
        'best_losses': []
    }
    
    hybrid_all_results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'iterations': [],
        'best_losses': []
    }
    
    print(f"\nPerforming {n_splits}-fold cross-validation...")
    print("Comparing Baseline ACOR vs Hybrid ACOR-LM")
    print("=" * 60)
    
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{n_splits}")
        print(f"{'='*60}")
        print(f"Training samples: {len(train_idx)}, Test samples: {len(test_idx)}")
        
        # Split data for this fold
        X_train_fold = X[train_idx]
        X_test_fold = X[test_idx]
        y_train_fold = y[train_idx]
        y_test_fold = y[test_idx]
        
        # Standardize features (fit on training fold only, transform both)
        scaler_fold = StandardScaler()
        X_train_fold = scaler_fold.fit_transform(X_train_fold)
        X_test_fold = scaler_fold.transform(X_test_fold)
        
        print(f"\nRunning {n_runs} independent experiments for this fold...")
        print("-" * 60)
        
        # Run Baseline ACOR
        print("Training Baseline ACOR...")
        baseline_results = evaluate_baseline_acor(
            X_train_fold, X_test_fold, y_train_fold, y_test_fold, n_runs=n_runs
        )
        
        # Run Hybrid ACOR-LM
        print("Training Hybrid ACOR-LM...")
        hybrid_results = evaluate_acor_lm(
            X_train_fold, X_test_fold, y_train_fold, y_test_fold, n_runs=n_runs
        )
        
        # Store fold-level averages for Baseline
        baseline_fold_results['fold_accuracy'].append(np.mean(baseline_results['accuracy']))
        baseline_fold_results['fold_precision'].append(np.mean(baseline_results['precision']))
        baseline_fold_results['fold_recall'].append(np.mean(baseline_results['recall']))
        baseline_fold_results['fold_f1_score'].append(np.mean(baseline_results['f1_score']))
        baseline_fold_results['fold_iterations'].append(np.mean(baseline_results['iterations']))
        baseline_fold_results['fold_losses'].append(np.mean(baseline_results['best_losses']))
        
        # Store fold-level averages for Hybrid
        hybrid_fold_results['fold_accuracy'].append(np.mean(hybrid_results['accuracy']))
        hybrid_fold_results['fold_precision'].append(np.mean(hybrid_results['precision']))
        hybrid_fold_results['fold_recall'].append(np.mean(hybrid_results['recall']))
        hybrid_fold_results['fold_f1_score'].append(np.mean(hybrid_results['f1_score']))
        hybrid_fold_results['fold_iterations'].append(np.mean(hybrid_results['iterations']))
        hybrid_fold_results['fold_losses'].append(np.mean(hybrid_results['best_losses']))
        
        # Store all individual run results
        baseline_all_results['accuracy'].extend(baseline_results['accuracy'])
        baseline_all_results['precision'].extend(baseline_results['precision'])
        baseline_all_results['recall'].extend(baseline_results['recall'])
        baseline_all_results['f1_score'].extend(baseline_results['f1_score'])
        baseline_all_results['iterations'].extend(baseline_results['iterations'])
        baseline_all_results['best_losses'].extend(baseline_results['best_losses'])
        
        hybrid_all_results['accuracy'].extend(hybrid_results['accuracy'])
        hybrid_all_results['precision'].extend(hybrid_results['precision'])
        hybrid_all_results['recall'].extend(hybrid_results['recall'])
        hybrid_all_results['f1_score'].extend(hybrid_results['f1_score'])
        hybrid_all_results['iterations'].extend(hybrid_results['iterations'])
        hybrid_all_results['best_losses'].extend(hybrid_results['best_losses'])
        
        # Print fold summary
        print(f"\nFold {fold_idx + 1} Results (Averaged across {n_runs} runs):")
        print(f"  Baseline ACOR:  Acc={baseline_fold_results['fold_accuracy'][-1]:.4f}, "
              f"Prec={baseline_fold_results['fold_precision'][-1]:.4f}, "
              f"Rec={baseline_fold_results['fold_recall'][-1]:.4f}, "
              f"F1={baseline_fold_results['fold_f1_score'][-1]:.4f}, "
              f"Iter={baseline_fold_results['fold_iterations'][-1]:.1f}")
        print(f"  Hybrid ACOR-LM: Acc={hybrid_fold_results['fold_accuracy'][-1]:.4f}, "
              f"Prec={hybrid_fold_results['fold_precision'][-1]:.4f}, "
              f"Rec={hybrid_fold_results['fold_recall'][-1]:.4f}, "
              f"F1={hybrid_fold_results['fold_f1_score'][-1]:.4f}, "
              f"Iter={hybrid_fold_results['fold_iterations'][-1]:.1f}")
    
    # Compile final results
    cv_results = {
        'baseline': {
            'fold_results': baseline_fold_results,
            'all_results': baseline_all_results
        },
        'hybrid': {
            'fold_results': hybrid_fold_results,
            'all_results': hybrid_all_results
        },
        'n_folds': n_splits,
        'n_runs_per_fold': n_runs
    }
    
    return cv_results

# 7. Main execution
# --------------------------------------------------
if __name__ == "__main__":
    print("Heart Disease Classification - Baseline ACOR vs Hybrid ACOR-LM")
    print("4-Fold Cross-Validation Comparison")
    print("=" * 60)
    print(f"Architecture: 35 inputs, 6 hidden (ReLU), 1 output (Sigmoid)")
    print(f"Total weights: {FNN_Thesis.get_num_weights(35, 6, 1)}")
    print(f"Total samples: {len(X)}")
    print(f"Evaluation: 4-fold CV with 50 runs per fold per algorithm")
    print()
    
    # Run k-fold cross-validation comparison
    cv_results = kfold_cross_validation_comparison(X, y, n_splits=4, n_runs=50)
    
    # Calculate and display final statistics
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS - PER-FOLD AVERAGES")
    print("=" * 60)
    
    for fold_idx in range(4):
        print(f"\nFold {fold_idx + 1}:")
        print(f"  Baseline ACOR:  "
              f"Acc={cv_results['baseline']['fold_results']['fold_accuracy'][fold_idx]:.4f}, "
              f"Prec={cv_results['baseline']['fold_results']['fold_precision'][fold_idx]:.4f}, "
              f"Rec={cv_results['baseline']['fold_results']['fold_recall'][fold_idx]:.4f}, "
              f"F1={cv_results['baseline']['fold_results']['fold_f1_score'][fold_idx]:.4f}, "
              f"Iter={cv_results['baseline']['fold_results']['fold_iterations'][fold_idx]:.1f}")
        print(f"  Hybrid ACOR-LM: "
              f"Acc={cv_results['hybrid']['fold_results']['fold_accuracy'][fold_idx]:.4f}, "
              f"Prec={cv_results['hybrid']['fold_results']['fold_precision'][fold_idx]:.4f}, "
              f"Rec={cv_results['hybrid']['fold_results']['fold_recall'][fold_idx]:.4f}, "
              f"F1={cv_results['hybrid']['fold_results']['fold_f1_score'][fold_idx]:.4f}, "
              f"Iter={cv_results['hybrid']['fold_results']['fold_iterations'][fold_idx]:.1f}")
    
    print("\n" + "=" * 60)
    print("OVERALL RESULTS (Averaged across all folds and runs)")
    print("=" * 60)
    
    baseline_all = cv_results['baseline']['all_results']
    hybrid_all = cv_results['hybrid']['all_results']
    
    print("\nBaseline ACOR:")
    print("-" * 60)
    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        mean_val = np.mean(baseline_all[metric])
        std_val = np.std(baseline_all[metric])
        print(f"  {metric.capitalize()}: {mean_val:.4f} ± {std_val:.4f}")
    print(f"  Convergence (Iterations): {np.mean(baseline_all['iterations']):.1f} ± {np.std(baseline_all['iterations']):.1f}")
    print(f"  Loss: {np.mean(baseline_all['best_losses']):.6f} ± {np.std(baseline_all['best_losses']):.6f}")
    
    print("\nHybrid ACOR-LM:")
    print("-" * 60)
    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        mean_val = np.mean(hybrid_all[metric])
        std_val = np.std(hybrid_all[metric])
        print(f"  {metric.capitalize()}: {mean_val:.4f} ± {std_val:.4f}")
    print(f"  Convergence (Iterations): {np.mean(hybrid_all['iterations']):.1f} ± {np.std(hybrid_all['iterations']):.1f}")
    print(f"  Loss: {np.mean(hybrid_all['best_losses']):.6f} ± {np.std(hybrid_all['best_losses']):.6f}")
    
    # Save results
    output_dir = os.path.dirname(__file__)
    
    with open(os.path.join(output_dir, 'heart_kfold_comparison_results.txt'), 'w') as f:
        f.write("Heart Disease Classification - Baseline ACOR vs Hybrid ACOR-LM\n")
        f.write("4-Fold Cross-Validation Comparison\n")
        f.write("=" * 60 + "\n")
        f.write(f"Architecture: 35-6-1 (Total weights: 223)\n")
        f.write(f"Total samples: {len(X)}\n")
        f.write(f"Number of folds: 4\n")
        f.write(f"Number of runs per fold per algorithm: 50\n\n")
        
        f.write("PER-FOLD RESULTS\n")
        f.write("=" * 60 + "\n")
        for fold_idx in range(4):
            f.write(f"\nFold {fold_idx + 1}:\n")
            f.write(f"  Baseline ACOR:\n")
            f.write(f"    Accuracy: {cv_results['baseline']['fold_results']['fold_accuracy'][fold_idx]:.4f}\n")
            f.write(f"    Precision: {cv_results['baseline']['fold_results']['fold_precision'][fold_idx]:.4f}\n")
            f.write(f"    Recall: {cv_results['baseline']['fold_results']['fold_recall'][fold_idx]:.4f}\n")
            f.write(f"    F1-Score: {cv_results['baseline']['fold_results']['fold_f1_score'][fold_idx]:.4f}\n")
            f.write(f"    Iterations: {cv_results['baseline']['fold_results']['fold_iterations'][fold_idx]:.1f}\n")
            f.write(f"  Hybrid ACOR-LM:\n")
            f.write(f"    Accuracy: {cv_results['hybrid']['fold_results']['fold_accuracy'][fold_idx]:.4f}\n")
            f.write(f"    Precision: {cv_results['hybrid']['fold_results']['fold_precision'][fold_idx]:.4f}\n")
            f.write(f"    Recall: {cv_results['hybrid']['fold_results']['fold_recall'][fold_idx]:.4f}\n")
            f.write(f"    F1-Score: {cv_results['hybrid']['fold_results']['fold_f1_score'][fold_idx]:.4f}\n")
            f.write(f"    Iterations: {cv_results['hybrid']['fold_results']['fold_iterations'][fold_idx]:.1f}\n")
        
        f.write("\n\nOVERALL RESULTS (Mean ± Std across all folds and runs)\n")
        f.write("=" * 60 + "\n")
        f.write("\nBaseline ACOR:\n")
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            mean_val = np.mean(baseline_all[metric])
            std_val = np.std(baseline_all[metric])
            f.write(f"  {metric.capitalize()}: {mean_val:.4f} ± {std_val:.4f}\n")
        f.write(f"  Convergence (Iterations): {np.mean(baseline_all['iterations']):.1f} ± {np.std(baseline_all['iterations']):.1f}\n")
        f.write(f"  Loss: {np.mean(baseline_all['best_losses']):.6f} ± {np.std(baseline_all['best_losses']):.6f}\n")
        
        f.write("\nHybrid ACOR-LM:\n")
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            mean_val = np.mean(hybrid_all[metric])
            std_val = np.std(hybrid_all[metric])
            f.write(f"  {metric.capitalize()}: {mean_val:.4f} ± {std_val:.4f}\n")
        f.write(f"  Convergence (Iterations): {np.mean(hybrid_all['iterations']):.1f} ± {np.std(hybrid_all['iterations']):.1f}\n")
        f.write(f"  Loss: {np.mean(hybrid_all['best_losses']):.6f} ± {np.std(hybrid_all['best_losses']):.6f}\n")
    
    print(f"\nResults saved to: heart_kfold_comparison_results.txt")
    
    # Create comparison plot
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    x = np.arange(len(metrics))
    width = 0.35
    
    baseline_means = [np.mean(baseline_all[m]) for m in metrics]
    baseline_stds = [np.std(baseline_all[m]) for m in metrics]
    hybrid_means = [np.mean(hybrid_all[m]) for m in metrics]
    hybrid_stds = [np.std(hybrid_all[m]) for m in metrics]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, baseline_means, width, yerr=baseline_stds, 
                   label='Baseline ACOR', capsize=5, alpha=0.7, color='skyblue')
    bars2 = ax.bar(x + width/2, hybrid_means, width, yerr=hybrid_stds,
                   label='Hybrid ACOR-LM', capsize=5, alpha=0.7, color='orange')
    
    ax.set_ylabel('Score')
    ax.set_title('Heart Disease Classification - Baseline ACOR vs Hybrid ACOR-LM\n(4-fold CV, 50 runs/fold)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
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
    plt.savefig(os.path.join(output_dir, 'heart_kfold_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Comparison plot saved to: heart_kfold_comparison.png")
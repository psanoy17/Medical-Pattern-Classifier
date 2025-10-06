"""
ACOR-LM with Multiple Colonies for Diabetes Classification

This implementation follows the thesis specifications:
- Multiple Colony ACOR with Levenberg-Marquardt local search
- FNN Architecture: 8 inputs, 6 hidden (ReLU), 1 output (Sigmoid)
- Binary Cross-Entropy Loss
- 4-fold cross-validation
- 50 independent runs
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
import matplotlib.pyplot as plt
import pickle
import sys

# Add parent directory to path to import lm_local_search
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from cancer.lm_local_search import MultipleColonyACOR, LevenbergMarquardt

# Set random seed for reproducibility
np.random.seed(42)

# 1. Load and preprocess the data
# --------------------------------------------------
data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'diabetes.csv'))

# Target variable is 'Outcome' (already 0/1)
X = data.drop('Outcome', axis=1).values
y = data['Outcome'].values

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 2. Define FNN matching thesis specifications (8, 6, 1)
# --------------------------------------------------
class FNN_Thesis:
    """
    Feedforward Neural Network matching thesis specifications.
    Architecture: Input(8) -> Hidden(6, ReLU) -> Output(1, Sigmoid)
    Total weights: 8*6 + 6 + 6*1 + 1 = 61 weights
    """
    def __init__(self, input_dim=8, hidden_dim=6, output_dim=1):
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

    def forward(self, X):
        """Forward pass with ReLU and Sigmoid activations"""
        # Hidden layer with ReLU activation
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0, z1)  # ReLU activation
        
        # Output layer with Sigmoid activation
        z2 = a1 @ self.W2 + self.b2
        a2 = 1 / (1 + np.exp(-z2))  # Sigmoid activation
        
        return a2.squeeze()

    def predict(self, X):
        """Predict class based on output threshold"""
        output = self.forward(X)
        return (output > 0.5).astype(int)

    @staticmethod
    def get_num_weights(input_dim=8, hidden_dim=6, output_dim=1):
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

# 4. Cross-validation and multiple runs evaluation
# --------------------------------------------------
def evaluate_acor_lm(X, y, n_folds=4, n_runs=50):
    """
    Evaluate ACOR-LM with multiple colonies using 4-fold CV and 50 runs
    
    Args:
        X: Input features
        y: Target labels
        n_folds: Number of cross-validation folds
        n_runs: Number of independent runs
        
    Returns:
        Dictionary with evaluation results
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
    
    # 4-fold cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\n=== FOLD {fold + 1}/{n_folds} ===")
        
        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]
        
        # 50 independent runs per fold
        fold_results = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'best_losses': [],
            'iterations': []
        }
        
        for run in range(n_runs):
            print(f"  Run {run + 1}/{n_runs}", end=" ")
            
            # Initialize model
            model = FNN_Thesis(input_dim=8, hidden_dim=6, output_dim=1)
            num_weights = FNN_Thesis.get_num_weights(8, 6, 1)
            
            # Create objective function for this fold
            def obj_func(weights):
                return objective_function(weights, model, X_train_fold, y_train_fold)
            
            # Initialize Multiple Colony ACOR-LM
            acor_lm = MultipleColonyACOR(
                obj_func=obj_func,
                dim=num_weights,
                n_colonies=3,
                n_ants=30,
                n_samples=80,
                q=0.1,
                xi=0.85,
                max_iter=100,
                patience=15,
                sharing_frequency=10,
                sharing_ratio=0.1,
                seed=42 + run  # Different seed for each run
            )
            
            # Optimize
            best_weights, best_loss, iterations = acor_lm.optimize(
                lb=-3.0, ub=3.0, model=model, X_train=X_train_fold, y_train=y_train_fold
            )
            
            # Evaluate on test set
            model.set_weights(best_weights)
            y_pred = model.predict(X_test_fold)
            
            # Calculate metrics
            acc = accuracy_score(y_test_fold, y_pred)
            prec = precision_score(y_test_fold, y_pred, zero_division=0)
            rec = recall_score(y_test_fold, y_pred, zero_division=0)
            f1 = f1_score(y_test_fold, y_pred, zero_division=0)
            
            # Store results
            fold_results['accuracy'].append(acc)
            fold_results['precision'].append(prec)
            fold_results['recall'].append(rec)
            fold_results['f1_score'].append(f1)
            fold_results['best_losses'].append(best_loss)
            fold_results['iterations'].append(iterations)
            
            print(f"Acc: {acc:.3f}, Loss: {best_loss:.3f}")
        
        # Average results for this fold
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'best_losses', 'iterations']:
            results[metric].append(np.mean(fold_results[metric]))
    
    return results

# 5. Main execution
# --------------------------------------------------
if __name__ == "__main__":
    print("ACOR-LM with Multiple Colonies for Diabetes Classification")
    print("=" * 60)
    print(f"Architecture: 8 inputs, 6 hidden (ReLU), 1 output (Sigmoid)")
    print(f"Total weights: {FNN_Thesis.get_num_weights(8, 6, 1)}")
    print(f"Evaluation: 4-fold CV × 50 runs = 200 total experiments")
    print()
    
    # Run evaluation
    results = evaluate_acor_lm(X, y, n_folds=4, n_runs=50)
    
    # Calculate final statistics
    print("\n" + "=" * 60)
    print("FINAL RESULTS (Averaged across 4 folds and 50 runs)")
    print("=" * 60)
    
    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        mean_val = np.mean(results[metric])
        std_val = np.std(results[metric])
        print(f"{metric.capitalize()}: {mean_val:.4f} ± {std_val:.4f}")
    
    print(f"Best Loss: {np.mean(results['best_losses']):.6f} ± {np.std(results['best_losses']):.6f}")
    print(f"Iterations: {np.mean(results['iterations']):.1f} ± {np.std(results['iterations']):.1f}")
    
    # Save results
    output_dir = os.path.dirname(__file__)
    results_data = {
        'results': results,
        'architecture': {'input': 8, 'hidden': 6, 'output': 1, 'weights': 61},
        'evaluation': {'folds': 4, 'runs': 50, 'total_experiments': 200},
        'algorithm': 'ACOR-LM with Multiple Colonies'
    }
    
    with open(os.path.join(output_dir, 'diabetes_acor_lm_results.pkl'), 'wb') as f:
        pickle.dump(results_data, f)
    
    print(f"\nResults saved to: {os.path.join(output_dir, 'diabetes_acor_lm_results.pkl')}")
    
    # Create summary plot
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    means = [np.mean(results[m]) for m in metrics]
    stds = [np.std(results[m]) for m in metrics]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, means, yerr=stds, capsize=5, 
                   color=['skyblue', 'orange', 'green', 'red'], alpha=0.7)
    plt.ylim(0, 1)
    plt.title('ACOR-LM with Multiple Colonies - Diabetes Classification Performance')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        plt.text(bar.get_x() + bar.get_width() / 2, mean + std + 0.02, 
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'diabetes_acor_lm_performance.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nPerformance plot saved to: diabetes_acor_lm_performance.png")




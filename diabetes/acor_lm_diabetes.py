"""
ACOR-LM with Multiple Colonies for Diabetes Classification

This implementation follows the thesis specifications:
- Multiple Colony ACOR with Levenberg-Marquardt local search
- FNN Architecture: 8 inputs, 6 hidden (ReLU), 1 output (Sigmoid)
- Binary Cross-Entropy Loss
- Single train-test split (80-20)
- 50 independent runs
- Uses preprocessed diabetes1.dat with 8 features

Reference: Hybrid ACOR-LM algorithm combining:
- SOCHA's ACOR (Socha & Dorigo, 2008)
- Levenberg-Marquardt local search for fine-tuning
"""

import numpy as np
import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
import matplotlib.pyplot as plt

# Add parent directory to path to import lm_local_search
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lm_local_search import MultipleColonyACOR, LevenbergMarquardt

# Set random seed for reproducibility
np.random.seed(42)

# Static loss threshold from baseline ACOR (UPDATE THIS VALUE after running baseline)
LOSS_THRESHOLD = 1.341828


# ==============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ==============================================================================
data = pd.read_csv(
    os.path.join(os.path.dirname(__file__), 'diabetes1.dat'),
    sep=' ',
    header=None
)

X = data.iloc[:, :-2].values  # First 8 columns are features
y_onehot = data.iloc[:, -2:].values  # Last 2 columns are one-hot encoded target
y = np.argmax(y_onehot, axis=1)  # Convert to single label

print(f"Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Target distribution: Class 0: {np.sum(y==0)}, Class 1: {np.sum(y==1)}")

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Stratified train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")


# ==============================================================================
# 2. FEEDFORWARD NEURAL NETWORK (FNN)
# ==============================================================================
class FNN:
    """
    Feedforward Neural Network matching thesis specifications.
    
    Architecture: Input(8) -> Hidden(6, ReLU) -> Output(1, Sigmoid)
    Total weights: 8*6 + 6 + 6*1 + 1 = 61 weights
    
    Weight vector structure:
    - W1: input-to-hidden weights (8*6 = 48)
    - b1: hidden biases (6)
    - W2: hidden-to-output weights (6*1 = 6)
    - b2: output bias (1)
    """
    def __init__(self, input_dim=8, hidden_dim=6, output_dim=1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def set_weights(self, weights):
        """Unpack flat weight vector into layer matrices"""
        idx = 0
        self.W1 = weights[idx:idx+self.input_dim*self.hidden_dim].reshape(self.input_dim, self.hidden_dim)
        idx += self.input_dim*self.hidden_dim
        self.b1 = weights[idx:idx+self.hidden_dim]
        idx += self.hidden_dim
        self.W2 = weights[idx:idx+self.hidden_dim*self.output_dim].reshape(self.hidden_dim, self.output_dim)
        idx += self.hidden_dim*self.output_dim
        self.b2 = weights[idx:idx+self.output_dim]
    
    def _stable_sigmoid(self, z):
        """Numerically stable sigmoid to prevent overflow"""
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        """Forward pass: Input -> ReLU -> Sigmoid"""
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0, z1)  # ReLU activation
        z2 = a1 @ self.W2 + self.b2
        a2 = self._stable_sigmoid(z2)  # Sigmoid activation
        return a2.squeeze()

    def predict(self, X):
        """Binary classification with 0.5 threshold"""
        return (self.forward(X) > 0.5).astype(int)

    @staticmethod
    def get_num_weights(input_dim=8, hidden_dim=6, output_dim=1):
        """Calculate total number of weights in the network"""
        return input_dim*hidden_dim + hidden_dim + hidden_dim*output_dim + output_dim


# ==============================================================================
# 3. OBJECTIVE FUNCTION (Binary Cross-Entropy Loss)
# ==============================================================================
def objective_function(weights, model, X_train, y_train):
    """
    Binary Cross-Entropy Loss as fitness function
    
    BCE = -mean(y * log(p) + (1-y) * log(1-p))
    
    Args:
        weights: Flat weight vector for FNN
        model: FNN instance
        X_train: Training features
        y_train: Training labels
        
    Returns:
        BCE loss value (lower is better)
    """
    model.set_weights(weights)
    y_pred = model.forward(X_train)
    eps = 1e-8  # Small constant to prevent log(0)
    loss = -np.mean(y_train*np.log(y_pred+eps) + (1-y_train)*np.log(1-y_pred+eps))
    return loss


# ==============================================================================
# 4. EVALUATION FUNCTION
# ==============================================================================
def evaluate_hybrid_acor_lm(X_train, X_test, y_train, y_test, n_runs=50, loss_threshold=LOSS_THRESHOLD):
    """
    Evaluate Hybrid ACOR-LM using 50 independent runs
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        n_runs: Number of independent runs
        loss_threshold: Static threshold for iteration counting
        
    Returns:
        Dictionary with evaluation results including iteration-to-threshold
    """
    results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'confusion_matrices': [],
        'best_losses': [],
        'iterations': [],
        'iterations_to_threshold': [],
        'threshold_reached': []
    }
    
    input_dim = 8
    hidden_dim = 6
    output_dim = 1
    num_weights = FNN.get_num_weights(input_dim, hidden_dim, output_dim)
    
    print(f"\nRunning {n_runs} independent experiments...")
    print(f"Loss Threshold: {loss_threshold:.6f}")
    print("=" * 60)
    
    for run in range(n_runs):
        print(f"Run {run + 1}/{n_runs}", end=" ")
        
        # Initialize model
        model = FNN(input_dim, hidden_dim, output_dim)
        
        # Create objective function wrapper
        def obj_func(weights):
            return objective_function(weights, model, X_train, y_train)
        
        # Initialize and run ACOR-LM with threshold tracking
        acor_lm = MultipleColonyACOR(
            obj_func=obj_func,
            dim=num_weights,
            n_colonies=3,
            n_ants=2,
            n_samples=136,
            q=0.01,
            xi=0.95,
            max_iter=100,
            patience=15,
            local_patience=5,
            sharing_frequency=10,
            sharing_ratio=0.1,
            seed=42 + run  # Different seed for each run
        )
        
        best_weights, best_loss, iterations, iter_to_threshold = acor_lm.optimize(
            lb=-3, ub=3, model=model, X_train=X_train, y_train=y_train,
            loss_threshold=loss_threshold
        )
        
        # Determine if threshold was reached (101 is penalty for not reached)
        threshold_reached = iter_to_threshold < 101
        
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
        results['iterations_to_threshold'].append(iter_to_threshold)
        results['threshold_reached'].append(threshold_reached)
        
        status = f"Iter2Thresh={iter_to_threshold}" if threshold_reached else "NOT REACHED (penalty=101)"
        print(f"Acc: {acc:.3f}, Loss: {best_loss:.4f}, Iter: {iterations}, {status}")
    
    return results


# ==============================================================================
# 5. MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("HYBRID ACOR-LM FOR DIABETES CLASSIFICATION")
    print("=" * 70)
    print(f"Architecture: 8 inputs, 6 hidden (ReLU), 1 output (Sigmoid)")
    print(f"Total weights: {FNN.get_num_weights(8, 6, 1)}")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Evaluation: 50 independent runs")
    print(f"Loss Threshold (from Baseline): {LOSS_THRESHOLD:.6f}")
    print()
    
    # Run evaluation
    results = evaluate_hybrid_acor_lm(X_train, X_test, y_train, y_test, n_runs=50)
    
    # ==================================================================
    # PRINT FINAL RESULTS
    # ==================================================================
    print("\n" + "=" * 70)
    print("FINAL RESULTS (Averaged across 50 runs)")
    print("=" * 70)
    
    # Performance metrics
    print("\n--- Performance Metrics ---")
    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        mean_val = np.mean(results[metric])
        std_val = np.std(results[metric])
        print(f"{metric.capitalize():12}: {mean_val:.4f} ± {std_val:.4f}")
    
    print(f"\n{'Best Loss':12}: {np.mean(results['best_losses']):.6f} ± {np.std(results['best_losses']):.6f}")
    print(f"{'Iterations':12}: {np.mean(results['iterations']):.1f} ± {np.std(results['iterations']):.1f}")
    
    # Iteration-based Time-to-Target Analysis
    print("\n--- Time-to-Target Analysis (Iterations) ---")
    print(f"Target Threshold (from Baseline): {LOSS_THRESHOLD:.6f}")
    
    # Success rate
    success_count = sum(results['threshold_reached'])
    success_rate = success_count / len(results['threshold_reached']) * 100
    print(f"Success Rate: {success_rate:.1f}% ({success_count}/50 runs reached threshold)")
    
    # Compute average iterations INCLUDING penalty (101) for runs that didn't reach threshold
    avg_iter_with_penalty = np.mean(results['iterations_to_threshold'])
    std_iter_with_penalty = np.std(results['iterations_to_threshold'])
    min_iter = np.min(results['iterations_to_threshold'])
    max_iter = np.max(results['iterations_to_threshold'])
    
    print(f"\nIterations to Threshold (ALL runs, penalty=101 for failures):")
    print(f"  Mean: {avg_iter_with_penalty:.1f} ± {std_iter_with_penalty:.1f}")
    print(f"  Min:  {min_iter}, Max: {max_iter}")
    
    # Stats for successful runs only (for reference)
    successful_iters = [i for i, reached in zip(results['iterations_to_threshold'], results['threshold_reached']) if reached]
    if successful_iters:
        avg_successful = np.mean(successful_iters)
        std_successful = np.std(successful_iters)
        print(f"\nIterations to Threshold (successful runs only, for reference):")
        print(f"  Mean: {avg_successful:.1f} ± {std_successful:.1f}")
        
        # Show if threshold was reached during initialization vs optimization
        reached_in_init = sum(1 for i in successful_iters if i == 0)
        reached_in_optim = len(successful_iters) - reached_in_init
        print(f"\n  Reached during initialization (Iter = 0): {reached_in_init}")
        print(f"  Reached during optimization (Iter > 0): {reached_in_optim}")
    
    # Calculate and print average confusion matrix
    avg_cm = np.mean(results['confusion_matrices'], axis=0)
    print(f"\nAverage Confusion Matrix:")
    print(avg_cm)
    
    # Find best run
    best_run_idx = np.argmax(results['accuracy'])
    best_accuracy = results['accuracy'][best_run_idx]
    print(f"\nBest Run: {best_run_idx + 1} with accuracy {best_accuracy:.4f}")
    
    # ==================================================================
    # CREATE PERFORMANCE PLOT
    # ==================================================================
    output_dir = os.path.dirname(__file__)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    means = [np.mean(results[m]) for m in metrics]
    stds = [np.std(results[m]) for m in metrics]
    
    bars = ax.bar(metrics, means, yerr=stds, capsize=5, 
                  color=['skyblue', 'orange', 'green', 'red'], alpha=0.7)
    ax.set_ylim(0, 1)
    ax.set_title('Hybrid ACOR-LM - Diabetes Classification\n(50 runs)')
    ax.set_ylabel('Score')
    ax.grid(True, alpha=0.3)
    
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + std + 0.02, 
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'diabetes_acor_lm_performance.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nPerformance plot saved to: diabetes_acor_lm_performance.png")
    
    # ==================================================================
    # SUMMARY FOR COMPARISON WITH BASELINE
    # ==================================================================
    print("\n" + "=" * 70)
    print("SUMMARY FOR COMPARISON WITH BASELINE")
    print("=" * 70)
    print(f"Algorithm: Hybrid ACOR-LM")
    print(f"Loss Threshold (from Baseline): {LOSS_THRESHOLD:.6f}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Avg Iterations to Threshold (with penalty): {avg_iter_with_penalty:.1f}")
    print(f"Avg Total Iterations: {np.mean(results['iterations']):.1f}")
    print(f"Avg Final Accuracy: {np.mean(results['accuracy']):.4f}")
    print(f"Avg Final Loss: {np.mean(results['best_losses']):.6f}")
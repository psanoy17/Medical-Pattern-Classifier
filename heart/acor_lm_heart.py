"""
ACOR-LM with Multiple Colonies for Heart Disease Classification

This implementation follows the thesis specifications:
- Multiple Colony ACOR with Levenberg-Marquardt local search
- FNN Architecture: 35 inputs, 6 hidden (ReLU), 1 output (Sigmoid)
- Binary Cross-Entropy Loss
- Single train-test split (80-20)
- 50 independent runs
- Uses preprocessed heart1.dat with 35 features
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
import pickle

# Add parent directory to path to import lm_local_search
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from lm_local_search import MultipleColonyACOR, LevenbergMarquardt

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

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into train and test sets (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

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

# 4. Multiple runs evaluation
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

# 5. Main execution
# --------------------------------------------------
if __name__ == "__main__":
    print("ACOR-LM with Multiple Colonies for Heart Disease Classification")
    print("=" * 60)
    print(f"Architecture: 35 inputs, 6 hidden (ReLU), 1 output (Sigmoid)")
    print(f"Total weights: {FNN_Thesis.get_num_weights(35, 6, 1)}")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Evaluation: 50 independent runs")
    print()
    
    # Run evaluation
    results = evaluate_acor_lm(X_train, X_test, y_train, y_test, n_runs=50)
    
    # Calculate final statistics
    print("\n" + "=" * 60)
    print("FINAL RESULTS (Averaged across 50 runs)")
    print("=" * 60)
    
    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        mean_val = np.mean(results[metric])
        std_val = np.std(results[metric])
        print(f"{metric.capitalize()}: {mean_val:.4f} ± {std_val:.4f}")
    
    print(f"Best Loss: {np.mean(results['best_losses']):.6f} ± {np.std(results['best_losses']):.6f}")
    print(f"Iterations: {np.mean(results['iterations']):.1f} ± {np.std(results['iterations']):.1f}")
    
    # Calculate average confusion matrix
    avg_cm = np.mean(results['confusion_matrices'], axis=0)
    print(f"\nAverage Confusion Matrix:")
    print(avg_cm)
    
    print(f"\nBest Model: Run {results['best_run_index'] + 1} with accuracy {results['best_overall_accuracy']:.4f}")
    
    # Save results
    output_dir = os.path.dirname(__file__)
    
    # Save pickle with best model
    model_data = {
        'best_model_weights': results['best_model_weights'],
        'best_run_index': results['best_run_index'],
        'best_overall_accuracy': results['best_overall_accuracy'],
        'metrics_summary': {
            'accuracy_mean': np.mean(results['accuracy']),
            'accuracy_std': np.std(results['accuracy']),
            'precision_mean': np.mean(results['precision']),
            'precision_std': np.std(results['precision']),
            'recall_mean': np.mean(results['recall']),
            'recall_std': np.std(results['recall']),
            'f1_mean': np.mean(results['f1_score']),
            'f1_std': np.std(results['f1_score']),
            'loss_mean': np.mean(results['best_losses']),
            'loss_std': np.std(results['best_losses']),
            'iterations_mean': np.mean(results['iterations']),
            'iterations_std': np.std(results['iterations']),
        },
        'average_confusion_matrix': avg_cm,
        'architecture': {'input': 35, 'hidden': 6, 'output': 1, 'weights': 223},
        'evaluation': {'train_samples': len(X_train), 'test_samples': len(X_test), 'runs': 50},
        'algorithm': 'ACOR-LM with Multiple Colonies',
        'dataset': 'heart1.dat (35 features)',
        'scaler': scaler  # Save scaler for future predictions
    }
    
    with open(os.path.join(output_dir, 'heart_acor_lm_model.pkl'), 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\nModel saved to: heart_acor_lm_model.pkl")
    
    # Create summary plot
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    means = [np.mean(results[m]) for m in metrics]
    stds = [np.std(results[m]) for m in metrics]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, means, yerr=stds, capsize=5, 
                   color=['skyblue', 'orange', 'green', 'red'], alpha=0.7)
    plt.ylim(0, 1)
    plt.title('ACOR-LM with Multiple Colonies - Heart Disease (35 features)')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        plt.text(bar.get_x() + bar.get_width() / 2, mean + std + 0.02, 
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heart_acor_lm_performance.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nPerformance plot saved to: heart_acor_lm_performance.png")
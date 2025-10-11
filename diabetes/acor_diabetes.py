import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
import matplotlib.pyplot as plt
import pickle

# Set random seed for reproducibility
np.random.seed(42)

# 1. Load and preprocess the data from diabetes1.dat
# --------------------------------------------------
data = pd.read_csv(
    os.path.join(os.path.dirname(__file__), 'diabetes1.dat'),
    sep=' ',
    header=None
)

X = data.iloc[:, :-2].values
y_onehot = data.iloc[:, -2:].values
y = np.argmax(y_onehot, axis=1)

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

# 2. Define a simple FNN (8, 6, 1)
# --------------------------------------------------
class FNN:
    """
    Feedforward Neural Network matching thesis specifications.
    Architecture: Input(8) -> Hidden(6, ReLU) -> Output(1, Sigmoid)
    Total weights: 8*6 + 6 + 6*1 + 1 = 61 weights
    """
    def __init__(self, input_dim=8, hidden_dim=6, output_dim=1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def set_weights(self, weights):
        idx = 0
        self.W1 = weights[idx:idx+self.input_dim*self.hidden_dim].reshape(self.input_dim, self.hidden_dim)
        idx += self.input_dim*self.hidden_dim
        self.b1 = weights[idx:idx+self.hidden_dim]
        idx += self.hidden_dim
        self.W2 = weights[idx:idx+self.hidden_dim*self.output_dim].reshape(self.hidden_dim, self.output_dim)
        idx += self.hidden_dim*self.output_dim
        self.b2 = weights[idx:idx+self.output_dim]

    def _stable_sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self._stable_sigmoid(z2)
        return a2.squeeze()

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)

    @staticmethod
    def get_num_weights(input_dim=8, hidden_dim=6, output_dim=1):
        return input_dim*hidden_dim + hidden_dim + hidden_dim*output_dim + output_dim

# 3. SOCHA-ACOR implementation
# --------------------------------------------------
class SOCHA_ACOR:
    def __init__(self, obj_func, dim, n_ants=2, n_samples=136, q=0.8, xi=0.7, max_iter=100, patience=15, seed=42):
        self.obj_func = obj_func
        self.dim = dim
        self.n_ants = n_ants
        self.n_samples = n_samples
        self.q = q
        self.xi = xi
        self.max_iter = max_iter
        self.patience = patience
        self.seed = seed
        
        if seed > 0:
            np.random.seed(seed)

    def optimize(self, lb, ub):
        """SOCHA-ACOR optimization matching R implementation"""
        e_abs = 1e-6
        e_rel = 1e-6
        max_value = 0
        eval_count = 0
        last_impr = self.max_iter
        nl = np.empty((self.n_samples, self.n_samples - 1), dtype=int)
        iteration = 0

        max_X = np.full(self.dim, np.nan)
        max_y = np.inf

        p_X = None
        p_v = []

        # Initialize archive
        for i in range(self.n_samples):
            X = np.random.uniform(lb, ub, self.dim)
            y = self.obj_func(X)
            eval_count += 1
            
            if p_X is None:
                p_X = X.reshape(1, -1)
            else:
                p_X = np.vstack([p_X, X.reshape(1, -1)])
            p_v.append(float(y))

        p_v = np.array(p_v, dtype=float)
        p_gr = self._rank_asc_with_random_ties(p_v)
        
        for i in range(self.n_samples):
            nl[i] = np.delete(np.arange(self.n_samples), i)

        imax0 = int(np.argmin(p_v))
        max_y = float(p_v[imax0])
        max_X = p_X[imax0]
        best_iter = 0

        # Main optimization loop
        for iteration in range(self.max_iter):
            dist_mean = p_X
            if np.sum(np.std(dist_mean, axis=0)) == 0:
                return max_X, max_y, iteration + 1
            
            dist_rank = p_gr
            o_X = self._gen_X(dist_mean, dist_rank, nl, self.n_ants, self.q, self.n_samples, self.xi)

            if o_X is None or len(o_X) == 0:
                return max_X, max_y, iteration + 1

            y = self.obj_func(o_X)
            eval_count += len(o_X)

            p_X = np.vstack([p_X, o_X])
            p_v = np.concatenate([p_v, y])
            p_gr = self._rank_asc_with_random_ties(p_v)

            idx_final = p_gr <= self.n_samples
            p_v = p_v[idx_final]
            p_gr = p_gr[idx_final]
            p_X = p_X[idx_final]

            for i in range(self.n_samples):
                nl[i] = np.delete(np.arange(self.n_samples), i)

            if np.min(y) < max_y:
                max_y = float(np.min(y))
                imax = int(np.argmin(y))
                max_X = o_X[imax]
                best_iter = iteration
                last_impr = eval_count
                
                if (abs(max_y - max_value) < abs(e_rel * max_value + e_abs)) or (max_y < max_value):
                    return max_X, max_y, iteration + 1

            if iteration - best_iter > self.patience:
                return max_X, max_y, iteration + 1

        return max_X, max_y, self.max_iter

    def _rank_desc_with_random_ties(self, values):
        n = len(values)
        perm = np.random.permutation(n)
        shuffled_vals = values[perm]
        order = np.argsort(-shuffled_vals, kind='mergesort')
        ranks = np.empty(n, dtype=int)
        ranks[order] = np.arange(1, n + 1)
        inv_perm = np.empty(n, dtype=int)
        inv_perm[perm] = np.arange(n)
        return ranks[inv_perm]

    def _rank_asc_with_random_ties(self, values):
        n = len(values)
        perm = np.random.permutation(n)
        shuffled_vals = values[perm]
        order = np.argsort(shuffled_vals, kind='mergesort')
        ranks = np.empty(n, dtype=int)
        ranks[order] = np.arange(1, n + 1)
        inv_perm = np.empty(n, dtype=int)
        inv_perm[perm] = np.arange(n)
        return ranks[inv_perm]

    def _gen_X(self, dist_mean, dist_rank, nl, n_of_points, q, k, xi):
        num_dists, num_dims = dist_mean.shape
        X = np.empty((n_of_points, num_dims), dtype=float)

        probs = self._normal_pdf(np.arange(1, num_dists + 1), mean=1.0, sd=q * k)
        probs = probs / probs.sum()
        idx = np.random.choice(num_dists, size=n_of_points, replace=True, p=probs)

        for l in range(n_of_points):
            j = idx[l]
            o_dist_mean = dist_mean - dist_mean[j]
            r_dist_mean = o_dist_mean.copy()
            available = nl[j]
            vec = None
            R = np.eye(num_dims)
            
            for m in range(num_dims - 1):
                if available.size == 0:
                    return None
                sub = r_dist_mean[available, m:]
                if sub.shape[0] == 0 or sub.shape[1] == 0:
                    return None
                dis = np.apply_along_axis(self._euc_dist, 1, sub)
                if np.sum(dis) == 0.0:
                    return None
                if available.size > 1:
                    p_choice = np.power(dis, 4.0)
                    p_choice = p_choice / p_choice.sum()
                    choose_idx = np.random.choice(len(available), p=p_choice)
                    choice = available[choose_idx]
                else:
                    choice = available[0]
                new_vec = o_dist_mean[choice]
                vec = new_vec[None, :] if vec is None else np.vstack([vec, new_vec])
                Q, _ = np.linalg.qr(vec.T, mode='complete')
                R = Q
                if np.linalg.det(R) < 0:
                    R[:, 0] *= -1
                r_dist_mean = o_dist_mean @ R
                available = available[available != choice]

            dist_sd = np.array([
                np.sum(np.abs(r_dist_mean[nl[j], i] - r_dist_mean[j, i])) / (k - 1)
                for i in range(num_dims)
            ])
            n_x = np.random.normal(loc=r_dist_mean[j], scale=dist_sd * xi, size=(num_dims,))
            n_x = (R @ n_x) + dist_mean[j]
            X[l] = n_x
        return X

    def _normal_pdf(self, x, mean, sd):
        if sd <= 0:
            out = np.zeros_like(x, dtype=float)
            out[np.isclose(x, mean)] = 1.0
            return out
        z = (x - mean) / sd
        return np.exp(-0.5 * z * z) / (sd * np.sqrt(2.0 * np.pi))

    def _euc_dist(self, d):
        return float(np.sqrt(np.sum(np.square(d))))

# 4. Objective function (binary cross-entropy loss)
# --------------------------------------------------
def objective_function(weights, model, X_train, y_train):
    """Binary Cross-Entropy Loss"""
    model.set_weights(weights)
    y_pred = model.forward(X_train)
    eps = 1e-8
    loss = -np.mean(y_train*np.log(y_pred+eps) + (1-y_train)*np.log(1-y_pred+eps))
    return loss

# 5. Multiple runs evaluation
# --------------------------------------------------
def evaluate_acor(X_train, X_test, y_train, y_test, n_runs=50):
    """
    Evaluate ACOR using 50 independent runs
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        n_runs: Number of independent runs
        
    Returns:
        Dictionary with evaluation results and best model
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
    
    # Track best model across all runs
    best_overall_accuracy = -1
    best_model_weights = None
    best_run_idx = -1
    
    input_dim = 8
    hidden_dim = 6
    output_dim = 1
    num_weights = FNN.get_num_weights(input_dim, hidden_dim, output_dim)
    
    print(f"\nRunning {n_runs} independent experiments...")
    print("=" * 60)
    
    for run in range(n_runs):
        print(f"Run {run + 1}/{n_runs}", end=" ")
        
        # Initialize model
        model = FNN(input_dim, hidden_dim, output_dim)
        
        # Create objective function
        def obj_func(weights):
            if weights.ndim == 1:
                return objective_function(weights, model, X_train, y_train)
            else:
                losses = []
                for w in weights:
                    losses.append(objective_function(w, model, X_train, y_train))
                return np.array(losses)
        
        # Initialize ACOR
        acor = SOCHA_ACOR(
            obj_func=obj_func,
            dim=num_weights,
            n_ants=2,
            n_samples=136,
            q=0.8,
            xi=0.7,
            max_iter=100,
            patience=15,
            seed=42 + run  # Different seed for each run
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

# 6. Main execution
# --------------------------------------------------
if __name__ == "__main__":
    print("ACOR for Diabetes Classification")
    print("=" * 60)
    print(f"Architecture: 8 inputs, 6 hidden (ReLU), 1 output (Sigmoid)")
    print(f"Total weights: {FNN.get_num_weights(8, 6, 1)}")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Evaluation: 50 independent runs")
    print()
    
    # Run evaluation
    results = evaluate_acor(X_train, X_test, y_train, y_test, n_runs=50)
    
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
    
    with open(os.path.join(output_dir, 'diabetes_acor_results.txt'), 'w') as f:
        f.write("Diabetes Classification - ACOR (8 Features)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Architecture: 8-6-1 (Total weights: 61)\n")
        f.write(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}\n")
        f.write(f"Number of runs: 50\n\n")
        f.write("RESULTS (Mean ± Std)\n")
        f.write("=" * 60 + "\n")
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            mean_val = np.mean(results[metric])
            std_val = np.std(results[metric])
            f.write(f"{metric.capitalize()}: {mean_val:.4f} ± {std_val:.4f}\n")
        f.write(f"Best Loss: {np.mean(results['best_losses']):.6f} ± {np.std(results['best_losses']):.6f}\n")
        f.write(f"Iterations: {np.mean(results['iterations']):.1f} ± {np.std(results['iterations']):.1f}\n")
        f.write("\nAverage Confusion Matrix:\n")
        f.write(str(avg_cm))
        f.write(f"\n\nBest Model: Run {results['best_run_index'] + 1} with accuracy {results['best_overall_accuracy']:.4f}\n")
    
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
        'architecture': {'input': 8, 'hidden': 6, 'output': 1, 'weights': 61},
        'evaluation': {'train_samples': len(X_train), 'test_samples': len(X_test), 'runs': 50},
        'algorithm': 'ACOR',
        'dataset': 'diabetes1.dat (8 features)',
        'scaler': scaler  # Save scaler for future predictions
    }
    
    with open(os.path.join(output_dir, 'diabetes_acor_model.pkl'), 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\nModel saved to: diabetes_acor_model.pkl")
    print(f"Results saved to: diabetes_acor_results.txt")
    
    # Create summary plot
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    means = [np.mean(results[m]) for m in metrics]
    stds = [np.std(results[m]) for m in metrics]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, means, yerr=stds, capsize=5, 
                   color=['skyblue', 'orange', 'green', 'red'], alpha=0.7)
    plt.ylim(0, 1)
    plt.title('ACOR - Diabetes Classification (8 features, 50 runs)')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3)
    
    for bar, mean, std in zip(bars, means, stds):
        plt.text(bar.get_x() + bar.get_width() / 2, mean + std + 0.02, 
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'diabetes_acor_performance.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Performance plot saved to: diabetes_acor_performance.png")
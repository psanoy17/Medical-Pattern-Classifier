"""
Baseline ACOR (SOCHA-ACOR) for Heart Disease Classification

This implementation follows SOCHA's original ACOR algorithm:
- Ant Colony Optimization for Continuous Domains (ACOR)
- Uses Gaussian kernel PDF for probabilistic solution selection
- QR decomposition for orthogonal rotation transformation
- Neighbor list-based exploration with distance weighting

Reference: Socha, K., & Dorigo, M. (2008). Ant colony optimization for continuous domains.

Architecture: 35 inputs, 6 hidden (ReLU), 1 output (Sigmoid)
Total weights: 35*6 + 6 + 6*1 + 1 = 223 weights
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# ==============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ==============================================================================
data = pd.read_csv(
    os.path.join(os.path.dirname(__file__), 'heart1.dat'),
    sep=' ',
    header=None
)

X = data.iloc[:, :-2].values  # First 35 columns are features
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
    
    Architecture: Input(35) -> Hidden(6, ReLU) -> Output(1, Sigmoid)
    Total weights: 35*6 + 6 + 6*1 + 1 = 223 weights
    
    Weight vector structure:
    - W1: input-to-hidden weights (35*6 = 210)
    - b1: hidden biases (6)
    - W2: hidden-to-output weights (6*1 = 6)
    - b2: output bias (1)
    """
    def __init__(self, input_dim=35, hidden_dim=6, output_dim=1):
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
    def get_num_weights(input_dim=35, hidden_dim=6, output_dim=1):
        """Calculate total number of weights in the network"""
        return input_dim*hidden_dim + hidden_dim + hidden_dim*output_dim + output_dim


# ==============================================================================
# 3. SOCHA-ACOR ALGORITHM (Baseline)
# ==============================================================================
class SOCHA_ACOR:
    """
    SOCHA's Ant Colony Optimization for Continuous Domains (ACOR)
    
    This is the baseline algorithm based on Socha & Dorigo (2008).
    
    Key Components:
    1. Solution Archive: Stores k best solutions found so far
    2. Gaussian Kernel PDF: Probabilistic selection based on solution rank
    3. QR Decomposition: Orthogonal rotation for correlated sampling
    4. Neighbor List: All other solutions in archive (for distance computation)
    5. Adaptive Standard Deviation: Computed from neighbor distances
    
    Parameters:
        obj_func: Objective function to minimize (BCE loss)
        dim: Dimensionality of search space (number of weights)
        n_ants: Number of new solutions generated per iteration (default: 2)
        n_samples: Archive size k (default: 230, based on 80% of 288 samples)
        q: Locality parameter for Gaussian kernel (default: 0.6)
        xi: Convergence speed parameter (default: 0.9)
        max_iter: Maximum iterations (default: 100)
        patience: Iterations without improvement before stopping (default: 15)
        seed: Random seed for reproducibility
    """
    
    def __init__(self, obj_func, dim, n_ants=2, n_samples=230, q=0.6, xi=0.9, 
                 max_iter=100, patience=15, seed=42):
        self.obj_func = obj_func
        self.dim = dim
        self.n_ants = n_ants
        self.n_samples = n_samples  # Archive size (k)
        self.q = q  # Locality parameter for Gaussian kernel
        self.xi = xi  # Convergence speed parameter
        self.max_iter = max_iter
        self.patience = patience
        self.seed = seed
        
        if seed > 0:
            np.random.seed(seed)

    def optimize(self, lb, ub):
        """
        Main SOCHA-ACOR optimization loop
        
        Args:
            lb: Lower bound for weight initialization
            ub: Upper bound for weight initialization
            
        Returns:
            Tuple of (best_weights, best_loss, iterations_used)
        """
        # Initialize neighbor list array
        neighbor_list = np.empty((self.n_samples, self.n_samples - 1), dtype=int)
        
        # Initialize tracking variables
        best_weights = np.full(self.dim, np.nan)
        best_loss = np.inf
        best_iteration = 0
        
        # Initialize solution archive and fitness values
        archive_solutions = None
        archive_fitness = []
        
        # ==================================================================
        # PHASE 1: Archive Initialization
        # ==================================================================
        for i in range(self.n_samples):
            solution = np.random.uniform(lb, ub, self.dim)
            fitness = self.obj_func(solution)
            
            if archive_solutions is None:
                archive_solutions = solution.reshape(1, -1)
            else:
                archive_solutions = np.vstack([archive_solutions, solution.reshape(1, -1)])
            archive_fitness.append(float(fitness))

        archive_fitness = np.array(archive_fitness, dtype=float)
        archive_ranks = self._rank_ascending_with_random_ties(archive_fitness)
        
        for i in range(self.n_samples):
            neighbor_list[i] = np.delete(np.arange(self.n_samples), i)

        best_idx = int(np.argmin(archive_fitness))
        best_loss = float(archive_fitness[best_idx])
        best_weights = archive_solutions[best_idx].copy()

        # ==================================================================
        # PHASE 2: Main Optimization Loop
        # ==================================================================
        for iteration in range(self.max_iter):
            
            if np.sum(np.std(archive_solutions, axis=0)) == 0:
                print(f"  [Iter {iteration+1}] Archive converged - all solutions identical")
                return best_weights, best_loss, iteration + 1
            
            new_solutions = self._generate_new_solutions(
                archive_solutions, archive_ranks, neighbor_list,
                self.n_ants, self.q, self.n_samples, self.xi
            )

            if new_solutions is None or len(new_solutions) == 0:
                print(f"  [Iter {iteration+1}] Solution generation failed")
                return best_weights, best_loss, iteration + 1

            new_fitness = self.obj_func(new_solutions)

            archive_solutions = np.vstack([archive_solutions, new_solutions])
            archive_fitness = np.concatenate([archive_fitness, new_fitness])
            archive_ranks = self._rank_ascending_with_random_ties(archive_fitness)

            keep_indices = archive_ranks <= self.n_samples
            archive_fitness = archive_fitness[keep_indices]
            archive_ranks = archive_ranks[keep_indices]
            archive_solutions = archive_solutions[keep_indices]

            for i in range(self.n_samples):
                neighbor_list[i] = np.delete(np.arange(self.n_samples), i)

            current_best_fitness = np.min(new_fitness)
            if current_best_fitness < best_loss:
                best_loss = float(current_best_fitness)
                best_idx = int(np.argmin(new_fitness))
                best_weights = new_solutions[best_idx].copy()
                best_iteration = iteration

            if iteration - best_iteration > self.patience:
                print(f"  [Iter {iteration+1}] Stopping: No improvement for {self.patience} iterations")
                return best_weights, best_loss, iteration + 1

        print(f"  [Iter {self.max_iter}] Stopping: Reached maximum iterations")
        return best_weights, best_loss, self.max_iter

    def _rank_ascending_with_random_ties(self, values):
        """Rank values in ascending order with random tie-breaking"""
        n = len(values)
        permutation = np.random.permutation(n)
        shuffled_values = values[permutation]
        
        sort_order = np.argsort(shuffled_values, kind='mergesort')
        ranks = np.empty(n, dtype=int)
        ranks[sort_order] = np.arange(1, n + 1)
        
        inverse_permutation = np.empty(n, dtype=int)
        inverse_permutation[permutation] = np.arange(n)
        
        return ranks[inverse_permutation]

    def _generate_new_solutions(self, archive_solutions, archive_ranks, neighbor_list,
                                 n_new_solutions, q, k, xi):
        """Generate new solutions using SOCHA's ACOR mechanism"""
        num_solutions, num_dimensions = archive_solutions.shape
        new_solutions = np.empty((n_new_solutions, num_dimensions), dtype=float)

        ranks = np.arange(1, num_solutions + 1)
        selection_probs = self._gaussian_kernel_pdf(ranks, mean=1.0, std=q * k)
        selection_probs = selection_probs / selection_probs.sum()
        
        selected_indices = np.random.choice(num_solutions, size=n_new_solutions, 
                                            replace=True, p=selection_probs)

        for solution_idx in range(n_new_solutions):
            guide_idx = selected_indices[solution_idx]
            
            centered_archive = archive_solutions - archive_solutions[guide_idx]
            rotated_archive = centered_archive.copy()
            
            available_neighbors = neighbor_list[guide_idx].copy()
            
            basis_vectors = None
            rotation_matrix = np.eye(num_dimensions)
            
            for dim_idx in range(num_dimensions - 1):
                if available_neighbors.size == 0:
                    return None
                
                subspace = rotated_archive[available_neighbors, dim_idx:]
                if subspace.shape[0] == 0 or subspace.shape[1] == 0:
                    return None
                
                distances = np.apply_along_axis(self._euclidean_distance, 1, subspace)
                if np.sum(distances) == 0.0:
                    return None
                
                if available_neighbors.size > 1:
                    distance_probs = np.power(distances, 4.0)
                    distance_probs = distance_probs / distance_probs.sum()
                    chosen_neighbor_idx = np.random.choice(len(available_neighbors), p=distance_probs)
                    chosen_neighbor = available_neighbors[chosen_neighbor_idx]
                else:
                    chosen_neighbor = available_neighbors[0]
                
                new_basis_vector = centered_archive[chosen_neighbor]
                if basis_vectors is None:
                    basis_vectors = new_basis_vector[None, :]
                else:
                    basis_vectors = np.vstack([basis_vectors, new_basis_vector])
                
                Q, _ = np.linalg.qr(basis_vectors.T, mode='complete')
                rotation_matrix = Q
                
                if np.linalg.det(rotation_matrix) < 0:
                    rotation_matrix[:, 0] *= -1
                
                rotated_archive = centered_archive @ rotation_matrix
                
                available_neighbors = available_neighbors[available_neighbors != chosen_neighbor]

            neighbor_indices = neighbor_list[guide_idx]
            adaptive_std = np.array([
                np.sum(np.abs(rotated_archive[neighbor_indices, d] - rotated_archive[guide_idx, d])) / (k - 1)
                for d in range(num_dimensions)
            ])
            
            new_solution_rotated = np.random.normal(
                loc=rotated_archive[guide_idx],
                scale=adaptive_std * xi,
                size=(num_dimensions,)
            )
            
            new_solution = (rotation_matrix @ new_solution_rotated) + archive_solutions[guide_idx]
            new_solutions[solution_idx] = new_solution
            
        return new_solutions

    def _gaussian_kernel_pdf(self, x, mean, std):
        """Compute Gaussian probability density function"""
        if std <= 0:
            out = np.zeros_like(x, dtype=float)
            out[np.isclose(x, mean)] = 1.0
            return out
        z = (x - mean) / std
        return np.exp(-0.5 * z * z) / (std * np.sqrt(2.0 * np.pi))

    def _euclidean_distance(self, vector):
        """Compute Euclidean norm of a vector"""
        return float(np.sqrt(np.sum(np.square(vector))))


# ==============================================================================
# 4. OBJECTIVE FUNCTION (Binary Cross-Entropy Loss)
# ==============================================================================
def objective_function(weights, model, X_train, y_train):
    """Binary Cross-Entropy Loss as fitness function"""
    model.set_weights(weights)
    y_pred = model.forward(X_train)
    eps = 1e-8
    loss = -np.mean(y_train*np.log(y_pred+eps) + (1-y_train)*np.log(1-y_pred+eps))
    return loss


# ==============================================================================
# 5. EVALUATION FUNCTION
# ==============================================================================
def evaluate_baseline_acor(X_train, X_test, y_train, y_test, n_runs=50):
    """Evaluate Baseline SOCHA-ACOR using 50 independent runs"""
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
    
    print(f"\nRunning {n_runs} independent experiments...")
    print("=" * 60)
    
    for run in range(n_runs):
        print(f"Run {run + 1}/{n_runs}", end=" ")
        
        model = FNN(input_dim, hidden_dim, output_dim)
        
        def obj_func(weights):
            if weights.ndim == 1:
                return objective_function(weights, model, X_train, y_train)
            else:
                return np.array([objective_function(w, model, X_train, y_train) for w in weights])
        
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
        
        best_weights, best_loss, iterations = acor.optimize(lb=-3, ub=3)
        
        model.set_weights(best_weights)
        y_pred = model.predict(X_test)
        
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
        
        print(f"Acc: {acc:.3f}, Prec: {prec:.3f}, Rec: {rec:.3f}, F1: {f1:.3f}, Loss: {best_loss:.3f}")
    
    return results


# ==============================================================================
# 6. MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("BASELINE ACOR (SOCHA-ACOR) FOR HEART DISEASE CLASSIFICATION")
    print("=" * 60)
    print(f"Architecture: 35 inputs, 6 hidden (ReLU), 1 output (Sigmoid)")
    print(f"Total weights: {FNN.get_num_weights(35, 6, 1)}")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Evaluation: 50 independent runs")
    print()
    
    results = evaluate_baseline_acor(X_train, X_test, y_train, y_test, n_runs=50)
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS (Averaged across 50 runs)")
    print("=" * 60)
    
    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        mean_val = np.mean(results[metric])
        std_val = np.std(results[metric])
        print(f"{metric.capitalize()}: {mean_val:.4f} ± {std_val:.4f}")
    
    print(f"Best Loss: {np.mean(results['best_losses']):.6f} ± {np.std(results['best_losses']):.6f}")
    print(f"Iterations: {np.mean(results['iterations']):.1f} ± {np.std(results['iterations']):.1f}")
    
    avg_cm = np.mean(results['confusion_matrices'], axis=0)
    print(f"\nAverage Confusion Matrix:")
    print(avg_cm)
    
    best_run_idx = np.argmax(results['accuracy'])
    best_accuracy = results['accuracy'][best_run_idx]
    print(f"\nBest Run: {best_run_idx + 1} with accuracy {best_accuracy:.4f}")
    
    output_dir = os.path.dirname(__file__)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    means = [np.mean(results[m]) for m in metrics]
    stds = [np.std(results[m]) for m in metrics]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, means, yerr=stds, capsize=5, 
                   color=['skyblue', 'orange', 'green', 'red'], alpha=0.7)
    plt.ylim(0, 1)
    plt.title('Baseline SOCHA-ACOR - Heart Disease Classification (35 features, 50 runs)')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3)
    
    for bar, mean, std in zip(bars, means, stds):
        plt.text(bar.get_x() + bar.get_width() / 2, mean + std + 0.02, 
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heart_acor_performance.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nPerformance plot saved to: heart_acor_performance.png")
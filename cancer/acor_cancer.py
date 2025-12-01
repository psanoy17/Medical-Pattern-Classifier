"""
Baseline ACOR (SOCHA-ACOR) for Cancer Classification

This implementation follows SOCHA's original ACOR algorithm:
- Ant Colony Optimization for Continuous Domains (ACOR)
- Uses Gaussian kernel PDF for probabilistic solution selection
- QR decomposition for orthogonal rotation transformation
- Neighbor list-based exploration with distance weighting

Reference: Socha, K., & Dorigo, M. (2008). Ant colony optimization for continuous domains.

Architecture: 9 inputs, 6 hidden (ReLU), 1 output (Sigmoid)
Total weights: 9*6 + 6 + 6*1 + 1 = 67 weights
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
    os.path.join(os.path.dirname(__file__), 'cancer1.dat'),
    sep=' ',
    header=None
)

X = data.iloc[:, :-2].values  # First 9 columns are features
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
    
    Architecture: Input(9) -> Hidden(6, ReLU) -> Output(1, Sigmoid)
    Total weights: 9*6 + 6 + 6*1 + 1 = 67 weights
    
    Weight vector structure:
    - W1: input-to-hidden weights (9*6 = 54)
    - b1: hidden biases (6)
    - W2: hidden-to-output weights (6*1 = 6)
    - b2: output bias (1)
    """
    def __init__(self, input_dim=9, hidden_dim=6, output_dim=1):
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
    def get_num_weights(input_dim=9, hidden_dim=6, output_dim=1):
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
        n_samples: Archive size k (default: 148, matching training samples)
        q: Locality parameter for Gaussian kernel (default: 0.95)
            - Controls exploration/exploitation balance
            - Higher q -> more exploration (flatter distribution)
            - Lower q -> more exploitation (peaked distribution)
        xi: Convergence speed parameter (default: 0.98)
            - Scales standard deviation for new solution generation
            - Higher xi -> larger search steps
            - Lower xi -> finer search steps
        max_iter: Maximum iterations (default: 100)
        patience: Iterations without improvement before stopping (default: 15)
        seed: Random seed for reproducibility
    """
    
    def __init__(self, obj_func, dim, n_ants=2, n_samples=148, q=0.95, xi=0.98, 
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
        # Each solution has (n_samples - 1) neighbors (all other archive members)
        neighbor_list = np.empty((self.n_samples, self.n_samples - 1), dtype=int)
        
        # Initialize tracking variables
        best_weights = np.full(self.dim, np.nan)
        best_loss = np.inf
        best_iteration = 0
        
        # Initialize solution archive (p_X) and fitness values (p_v)
        archive_solutions = None  # Will hold (n_samples, dim) array
        archive_fitness = []  # Will hold fitness values
        
        # ==================================================================
        # PHASE 1: Archive Initialization
        # Generate n_samples random solutions and evaluate them
        # ==================================================================
        for i in range(self.n_samples):
            # Generate random solution within bounds
            solution = np.random.uniform(lb, ub, self.dim)
            fitness = self.obj_func(solution)
            
            # Add to archive
            if archive_solutions is None:
                archive_solutions = solution.reshape(1, -1)
            else:
                archive_solutions = np.vstack([archive_solutions, solution.reshape(1, -1)])
            archive_fitness.append(float(fitness))

        archive_fitness = np.array(archive_fitness, dtype=float)
        
        # Rank solutions (ascending - lower loss is better)
        archive_ranks = self._rank_ascending_with_random_ties(archive_fitness)
        
        # Initialize neighbor list (all other solutions are neighbors)
        for i in range(self.n_samples):
            neighbor_list[i] = np.delete(np.arange(self.n_samples), i)

        # Find initial best solution
        best_idx = int(np.argmin(archive_fitness))
        best_loss = float(archive_fitness[best_idx])
        best_weights = archive_solutions[best_idx].copy()

        # ==================================================================
        # PHASE 2: Main Optimization Loop
        # ==================================================================
        for iteration in range(self.max_iter):
            
            # Check for archive convergence (all solutions identical)
            if np.sum(np.std(archive_solutions, axis=0)) == 0:
                print(f"  [Iter {iteration+1}] Archive converged - all solutions identical")
                return best_weights, best_loss, iteration + 1
            
            # Generate new solutions using ACOR mechanism
            new_solutions = self._generate_new_solutions(
                archive_solutions, archive_ranks, neighbor_list,
                self.n_ants, self.q, self.n_samples, self.xi
            )

            # Check if solution generation failed
            if new_solutions is None or len(new_solutions) == 0:
                print(f"  [Iter {iteration+1}] Solution generation failed")
                return best_weights, best_loss, iteration + 1

            # Evaluate new solutions
            new_fitness = self.obj_func(new_solutions)

            # Combine archive with new solutions
            archive_solutions = np.vstack([archive_solutions, new_solutions])
            archive_fitness = np.concatenate([archive_fitness, new_fitness])
            archive_ranks = self._rank_ascending_with_random_ties(archive_fitness)

            # Keep only top n_samples solutions (elitism)
            keep_indices = archive_ranks <= self.n_samples
            archive_fitness = archive_fitness[keep_indices]
            archive_ranks = archive_ranks[keep_indices]
            archive_solutions = archive_solutions[keep_indices]

            # Update neighbor list for new archive
            for i in range(self.n_samples):
                neighbor_list[i] = np.delete(np.arange(self.n_samples), i)

            # Check for improvement
            current_best_fitness = np.min(new_fitness)
            if current_best_fitness < best_loss:
                best_loss = float(current_best_fitness)
                best_idx = int(np.argmin(new_fitness))
                best_weights = new_solutions[best_idx].copy()
                best_iteration = iteration

            # Patience-based stopping (no improvement for patience iterations)
            if iteration - best_iteration > self.patience:
                print(f"  [Iter {iteration+1}] Stopping: No improvement for {self.patience} iterations")
                return best_weights, best_loss, iteration + 1

        print(f"  [Iter {self.max_iter}] Stopping: Reached maximum iterations")
        return best_weights, best_loss, self.max_iter

    def _rank_ascending_with_random_ties(self, values):
        """
        Rank values in ascending order with random tie-breaking
        
        This ensures that solutions with identical fitness get random ranks,
        maintaining diversity in the selection process.
        
        Args:
            values: Array of fitness values
            
        Returns:
            Array of ranks (1 = best, n = worst)
        """
        n = len(values)
        # Random permutation for tie-breaking
        permutation = np.random.permutation(n)
        shuffled_values = values[permutation]
        
        # Sort and assign ranks
        sort_order = np.argsort(shuffled_values, kind='mergesort')
        ranks = np.empty(n, dtype=int)
        ranks[sort_order] = np.arange(1, n + 1)
        
        # Restore original order
        inverse_permutation = np.empty(n, dtype=int)
        inverse_permutation[permutation] = np.arange(n)
        
        return ranks[inverse_permutation]

    def _generate_new_solutions(self, archive_solutions, archive_ranks, neighbor_list,
                                 n_new_solutions, q, k, xi):
        """
        Generate new solutions using SOCHA's ACOR mechanism
        
        This is the core of the ACOR algorithm:
        1. Select a guiding solution using Gaussian kernel PDF
        2. Apply QR decomposition for orthogonal rotation
        3. Compute adaptive standard deviation from neighbor distances
        4. Sample new solution in rotated space
        5. Transform back to original space
        
        Args:
            archive_solutions: Current solution archive (k x dim)
            archive_ranks: Ranks of archive solutions
            neighbor_list: Neighbor indices for each solution
            n_new_solutions: Number of new solutions to generate
            q: Locality parameter for Gaussian kernel
            k: Archive size
            xi: Convergence speed parameter
            
        Returns:
            Array of new solutions (n_new_solutions x dim)
        """
        num_solutions, num_dimensions = archive_solutions.shape
        new_solutions = np.empty((n_new_solutions, num_dimensions), dtype=float)

        # Compute selection probabilities using Gaussian kernel PDF
        # Solutions with better rank (lower) have higher probability
        ranks = np.arange(1, num_solutions + 1)
        selection_probs = self._gaussian_kernel_pdf(ranks, mean=1.0, std=q * k)
        selection_probs = selection_probs / selection_probs.sum()
        
        # Select guiding solutions probabilistically
        selected_indices = np.random.choice(num_solutions, size=n_new_solutions, 
                                            replace=True, p=selection_probs)

        # Generate each new solution
        for solution_idx in range(n_new_solutions):
            guide_idx = selected_indices[solution_idx]
            
            # Center archive around guiding solution
            centered_archive = archive_solutions - archive_solutions[guide_idx]
            rotated_archive = centered_archive.copy()
            
            # Get available neighbors for orthogonal basis construction
            available_neighbors = neighbor_list[guide_idx].copy()
            
            # Build orthogonal rotation matrix using QR decomposition
            basis_vectors = None
            rotation_matrix = np.eye(num_dimensions)
            
            for dim_idx in range(num_dimensions - 1):
                if available_neighbors.size == 0:
                    return None  # Not enough neighbors
                
                # Get subspace of remaining dimensions
                subspace = rotated_archive[available_neighbors, dim_idx:]
                if subspace.shape[0] == 0 or subspace.shape[1] == 0:
                    return None
                
                # Compute distances in subspace
                distances = np.apply_along_axis(self._euclidean_distance, 1, subspace)
                if np.sum(distances) == 0.0:
                    return None  # All neighbors at same point
                
                # Select neighbor based on distance (farther = more likely)
                if available_neighbors.size > 1:
                    distance_probs = np.power(distances, 4.0)  # Emphasize distant neighbors
                    distance_probs = distance_probs / distance_probs.sum()
                    chosen_neighbor_idx = np.random.choice(len(available_neighbors), p=distance_probs)
                    chosen_neighbor = available_neighbors[chosen_neighbor_idx]
                else:
                    chosen_neighbor = available_neighbors[0]
                
                # Add chosen direction to basis
                new_basis_vector = centered_archive[chosen_neighbor]
                if basis_vectors is None:
                    basis_vectors = new_basis_vector[None, :]
                else:
                    basis_vectors = np.vstack([basis_vectors, new_basis_vector])
                
                # Compute orthogonal rotation matrix via QR decomposition
                Q, _ = np.linalg.qr(basis_vectors.T, mode='complete')
                rotation_matrix = Q
                
                # Ensure proper rotation (det = +1, not reflection)
                if np.linalg.det(rotation_matrix) < 0:
                    rotation_matrix[:, 0] *= -1
                
                # Apply rotation to centered archive
                rotated_archive = centered_archive @ rotation_matrix
                
                # Remove chosen neighbor from available list
                available_neighbors = available_neighbors[available_neighbors != chosen_neighbor]

            # Compute adaptive standard deviation from neighbor distances in rotated space
            neighbor_indices = neighbor_list[guide_idx]
            adaptive_std = np.array([
                np.sum(np.abs(rotated_archive[neighbor_indices, d] - rotated_archive[guide_idx, d])) / (k - 1)
                for d in range(num_dimensions)
            ])
            
            # Sample new solution in rotated space
            new_solution_rotated = np.random.normal(
                loc=rotated_archive[guide_idx],
                scale=adaptive_std * xi,  # Scale by convergence parameter
                size=(num_dimensions,)
            )
            
            # Transform back to original space
            new_solution = (rotation_matrix @ new_solution_rotated) + archive_solutions[guide_idx]
            new_solutions[solution_idx] = new_solution
            
        return new_solutions

    def _gaussian_kernel_pdf(self, x, mean, std):
        """
        Compute Gaussian probability density function
        
        Used for probabilistic selection of guiding solutions.
        Solutions with rank closer to mean (1 = best) have higher probability.
        
        Args:
            x: Rank values
            mean: Mean of Gaussian (typically 1.0 for best rank)
            std: Standard deviation (q * k)
            
        Returns:
            Probability density values
        """
        if std <= 0:
            # Degenerate case: return 1 only for mean
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
# 5. EVALUATION FUNCTION
# ==============================================================================
def evaluate_baseline_acor(X_train, X_test, y_train, y_test, n_runs=50):
    """
    Evaluate Baseline SOCHA-ACOR using 50 independent runs
    
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
    
    input_dim = 9
    hidden_dim = 6
    output_dim = 1
    num_weights = FNN.get_num_weights(input_dim, hidden_dim, output_dim)
    
    print(f"\nRunning {n_runs} independent experiments...")
    print("=" * 60)
    
    for run in range(n_runs):
        print(f"Run {run + 1}/{n_runs}", end=" ")
        
        # Initialize model
        model = FNN(input_dim, hidden_dim, output_dim)
        
        # Create objective function wrapper
        def obj_func(weights):
            if weights.ndim == 1:
                return objective_function(weights, model, X_train, y_train)
            else:
                return np.array([objective_function(w, model, X_train, y_train) for w in weights])
        
        # Initialize and run SOCHA-ACOR
        acor = SOCHA_ACOR(
            obj_func=obj_func,
            dim=num_weights,
            n_ants=2,
            n_samples=148,
            q=0.95,
            xi=0.98,
            max_iter=100,
            patience=15,
            seed=42 + run  # Different seed for each run
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
        
        # Store results
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
    print("BASELINE ACOR (SOCHA-ACOR) FOR CANCER CLASSIFICATION")
    print("=" * 60)
    print(f"Architecture: 9 inputs, 6 hidden (ReLU), 1 output (Sigmoid)")
    print(f"Total weights: {FNN.get_num_weights(9, 6, 1)}")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Evaluation: 50 independent runs")
    print()
    
    # Run evaluation
    results = evaluate_baseline_acor(X_train, X_test, y_train, y_test, n_runs=50)
    
    # ==================================================================
    # PRINT FINAL RESULTS (Terminal Only - No File Saving)
    # ==================================================================
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
    
    # Calculate and print average confusion matrix
    avg_cm = np.mean(results['confusion_matrices'], axis=0)
    print(f"\nAverage Confusion Matrix:")
    print(avg_cm)
    
    # Find best run
    best_run_idx = np.argmax(results['accuracy'])
    best_accuracy = results['accuracy'][best_run_idx]
    print(f"\nBest Run: {best_run_idx + 1} with accuracy {best_accuracy:.4f}")
    
    # ==================================================================
    # CREATE PERFORMANCE PLOT (Optional - Still Saved)
    # ==================================================================
    output_dir = os.path.dirname(__file__)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    means = [np.mean(results[m]) for m in metrics]
    stds = [np.std(results[m]) for m in metrics]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, means, yerr=stds, capsize=5, 
                   color=['skyblue', 'orange', 'green', 'red'], alpha=0.7)
    plt.ylim(0, 1)
    plt.title('Baseline SOCHA-ACOR - Cancer Classification (9 features, 50 runs)')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3)
    
    for bar, mean, std in zip(bars, means, stds):
        plt.text(bar.get_x() + bar.get_width() / 2, mean + std + 0.02, 
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cancer_acor_performance.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nPerformance plot saved to: cancer_acor_performance.png")
"""
Baseline ACOR (SOCHA-ACOR) for Cancer Classification

This implementation follows SOCHA's original ACOR algorithm:
- Ant Colony Optimization for Continuous Domains (ACOR)
- Uses Gaussian kernel PDF for probabilistic solution selection
- Independent Gaussian sampling per dimension (standard ACOR)
- Adaptive standard deviation computed from archive distances

References:
- Socha, K., & Dorigo, M. (2008). Ant colony optimization for continuous domains.
  European Journal of Operational Research, 185(3), 1155-1173.
- Socha, K. (2008). Ant Colony Optimization for Continuous and Mixed-Variable Domains.
  PhD Thesis, Université Libre de Bruxelles.

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
    3. Independent Gaussian Sampling: Each dimension sampled independently
    4. Adaptive Standard Deviation: σ_i = ξ * Σ|s_e^i - s_l^i| / (k-1)
    
    Parameters:
        obj_func: Objective function to minimize (BCE loss)
        dim: Dimensionality of search space (number of weights)
        n_ants: Number of new solutions generated per iteration (m in thesis)
        n_samples: Archive size k
        q: Locality parameter for Gaussian kernel (controls selection pressure)
        xi: Convergence speed parameter (ξ in thesis)
        max_iter: Maximum iterations
        patience: Iterations without improvement before stopping
        seed: Random seed for reproducibility
    """
    
    def __init__(self, obj_func, dim, n_ants=2, n_samples=148, q=0.01, xi=0.95, 
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
        
        # Loss history for post-hoc threshold analysis
        self.loss_history = []  # List of (iteration, best_loss_at_that_point)
        
        if seed > 0:
            np.random.seed(seed)

    def optimize(self, lb, ub):
        """
        Main SOCHA-ACOR optimization loop
        
        Args:
            lb: Lower bound for weight initialization
            ub: Upper bound for weight initialization
            
        Returns:
            Tuple of (best_weights, best_loss, iterations_used, loss_history)
        """
        # Reset loss history
        self.loss_history = []
        
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
            
            # Track best during initialization
            if fitness < best_loss:
                best_loss = float(fitness)
                best_weights = solution.copy()

        archive_fitness = np.array(archive_fitness, dtype=float)
        
        # Record initial best loss (iteration 0 = after initialization)
        self.loss_history.append((0, best_loss))
        
        # Rank solutions (ascending - lower loss is better)
        archive_ranks = self._rank_ascending_with_random_ties(archive_fitness)
        
        # Initialize neighbor list
        for i in range(self.n_samples):
            neighbor_list[i] = np.delete(np.arange(self.n_samples), i)

        # ==================================================================
        # PHASE 2: Main Optimization Loop
        # ==================================================================
        iteration = 0
        for iteration in range(self.max_iter):
            
            # Check for archive convergence (all solutions identical)
            if np.sum(np.std(archive_solutions, axis=0)) == 0:
                break
            
            # Generate new solutions using ACOR mechanism
            new_solutions = self._generate_new_solutions(
                archive_solutions, archive_ranks,
                self.n_ants, self.q, self.n_samples, self.xi
            )

            # Check if solution generation failed
            if new_solutions is None or len(new_solutions) == 0:
                break

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
            
            # Record loss history after each iteration (1-indexed)
            self.loss_history.append((iteration + 1, best_loss))

            # Patience-based stopping
            if iteration - best_iteration > self.patience:
                break

        return best_weights, best_loss, iteration + 1, self.loss_history

    def _rank_ascending_with_random_ties(self, values):
        """
        Rank values in ascending order with random tie-breaking
        """
        n = len(values)
        permutation = np.random.permutation(n)
        shuffled_values = values[permutation]
        
        sort_order = np.argsort(shuffled_values, kind='mergesort')
        ranks = np.empty(n, dtype=int)
        ranks[sort_order] = np.arange(1, n + 1)
        
        inverse_permutation = np.empty(n, dtype=int)
        inverse_permutation[permutation] = np.arange(n)
        
        return ranks[inverse_permutation]

    def _generate_new_solutions(self, archive_solutions, archive_ranks, 
                                      n_new_solutions, q, k, xi):
        """
        Generate new solutions using Standard ACOR (Socha & Dorigo, 2008)
        
        Algorithm (Section 3.2 of the paper):
        1. Compute selection probabilities using Gaussian kernel:
           w_l = (1 / (q*k*sqrt(2π))) * exp(-((l-1)^2) / (2*(q*k)^2))
           where l is the rank of solution (1 = best)
        
        2. Select guide solution l probabilistically based on weights
        
        3. For each dimension i, compute adaptive standard deviation:
           σ_i^l = ξ * Σ_{e=1}^{k} |s_e^i - s_l^i| / (k-1)
           This is the average distance from guide to all other solutions
        
        4. Sample new solution:
           s_new^i ~ N(s_l^i, σ_i^l)
           Each dimension is sampled independently
        
        Args:
            archive_solutions: Current archive (k x dim)
            archive_ranks: Ranks of archive solutions (1 = best)
            n_new_solutions: Number of new solutions to generate (m)
            q: Locality parameter for selection
            k: Archive size
            xi: Convergence speed parameter (ξ)
            
        Returns:
            Array of new solutions (n_new_solutions x dim)
        """
        num_solutions, num_dimensions = archive_solutions.shape
        new_solutions = np.empty((n_new_solutions, num_dimensions), dtype=float)
        
        # Compute selection probabilities using Gaussian kernel
        # w_l ∝ exp(-((rank-1)^2) / (2*(q*k)^2))
        ranks = np.arange(1, num_solutions + 1)
        weights = self._gaussian_kernel_pdf(ranks, mean=1.0, std=q * k)
        weights = weights / weights.sum()
        
        for ant_idx in range(n_new_solutions):
            # Select guide solution probabilistically
            guide_idx = np.random.choice(num_solutions, p=weights)
            guide_solution = archive_solutions[guide_idx]
            
            # Compute standard deviation for each dimension
            # σ_i^l = ξ * Σ_{e=1}^{k} |s_e^i - s_l^i| / (k-1)
            # Note: Sum includes guide (distance = 0), so effectively sums over k-1 non-zero terms
            sigma = np.zeros(num_dimensions)
            for d in range(num_dimensions):
                distances = np.abs(archive_solutions[:, d] - guide_solution[d])
                sigma[d] = xi * np.sum(distances) / (k - 1)
            
            # Sample new solution from independent Gaussians
            # s_new^i ~ N(s_l^i, σ_i^l)
            new_solutions[ant_idx] = np.random.normal(loc=guide_solution, scale=sigma)
        
        return new_solutions

    def _gaussian_kernel_pdf(self, x, mean, std):
        """
        Compute Gaussian probability density function
        
        Used to weight solutions by rank for probabilistic selection.
        Formula: p(x) = (1 / (σ√(2π))) * exp(-(x-μ)² / (2σ²))
        
        For ACOR: μ=1 (best rank), σ=q*k
        - Smaller q → sharper distribution → stronger preference for best
        - Larger q → flatter distribution → more uniform selection
        
        Args:
            x: Rank values (1 = best, k = worst)
            mean: Mean of Gaussian (1.0 for best rank)
            std: Standard deviation (q * k)
            
        Returns:
            Unnormalized probability density values
        """
        if std <= 0:
            out = np.zeros_like(x, dtype=float)
            out[np.isclose(x, mean)] = 1.0
            return out
        z = (x - mean) / std
        return np.exp(-0.5 * z * z) / (std * np.sqrt(2.0 * np.pi))


# ==============================================================================
# 4. OBJECTIVE FUNCTION (Binary Cross-Entropy Loss)
# ==============================================================================
def objective_function(weights, model, X_train, y_train):
    """
    Binary Cross-Entropy Loss as fitness function
    """
    model.set_weights(weights)
    y_pred = model.forward(X_train)
    eps = 1e-8
    loss = -np.mean(y_train*np.log(y_pred+eps) + (1-y_train)*np.log(1-y_pred+eps))
    return loss


# ==============================================================================
# 5. HELPER FUNCTIONS
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
    return 101 # Penalty value for runs that never reached threshold


def run_single_acor_experiment(X_train, y_train, X_test, y_test, run_seed):
    """
    Run a single ACOR experiment and return results including loss history
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        run_seed: Random seed for this run
        
    Returns:
        Dictionary with run results including loss_history for post-hoc analysis
    """
    input_dim = 9
    hidden_dim = 6
    output_dim = 1
    num_weights = FNN.get_num_weights(input_dim, hidden_dim, output_dim)
    
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
        q=0.01,
        xi=0.95,
        max_iter=100,
        patience=15,
        seed=run_seed
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
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'confusion_matrix': cm,
        'best_loss': best_loss,
        'iterations': iterations,
        'loss_history': loss_history,
        'best_weights': best_weights
    }


# ==============================================================================
# 6. EVALUATION FUNCTION (Single-Phase with Post-Hoc Analysis)
# ==============================================================================
def evaluate_baseline_acor_single_phase(X_train, X_test, y_train, y_test, n_runs=50):
    """
    Evaluate Baseline SOCHA-ACOR using Single-Phase approach with post-hoc analysis
    
    This approach:
    1. Runs all experiments ONCE and stores loss history
    2. Computes threshold from average final loss
    3. Analyzes stored history to find iterations-to-threshold (NO RE-RUNNING)
    
    Args:
        X_train, X_test, y_train, y_test: Data splits
        n_runs: Number of independent runs
        
    Returns:
        Dictionary with evaluation results including iteration-based convergence metrics
    """
    print(f"\n{'='*70}")
    print(f"Running {n_runs} experiments (with loss history tracking)...")
    print("="*70)
    
    # Store all results including loss histories
    all_results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'confusion_matrices': [],
        'best_losses': [],
        'iterations': [],
        'loss_histories': []
    }
    
    for run in range(n_runs):
        print(f"  Run {run + 1}/{n_runs}", end=" ")
        
        run_result = run_single_acor_experiment(
            X_train, y_train, X_test, y_test, 
            run_seed=42 + run
        )
        
        all_results['accuracy'].append(run_result['accuracy'])
        all_results['precision'].append(run_result['precision'])
        all_results['recall'].append(run_result['recall'])
        all_results['f1_score'].append(run_result['f1_score'])
        all_results['confusion_matrices'].append(run_result['confusion_matrix'])
        all_results['best_losses'].append(run_result['best_loss'])
        all_results['iterations'].append(run_result['iterations'])
        all_results['loss_histories'].append(run_result['loss_history'])
        
        print(f"Loss: {run_result['best_loss']:.4f}, Acc: {run_result['accuracy']:.3f}, Iter: {run_result['iterations']}")
    
    # ==================================================================
    # POST-HOC THRESHOLD ANALYSIS (No re-running!)
    # ==================================================================
    average_final_loss = np.mean(all_results['best_losses'])
    loss_std = np.std(all_results['best_losses'])
    
    print(f"\n{'='*70}")
    print("POST-HOC THRESHOLD ANALYSIS")
    print("="*70)
    print(f"  Average Final Loss: {average_final_loss:.6f} ± {loss_std:.6f}")
    print(f"  Min Loss: {np.min(all_results['best_losses']):.6f}")
    print(f"  Max Loss: {np.max(all_results['best_losses']):.6f}")
    print(f"  >>> TARGET THRESHOLD SET TO: {average_final_loss:.6f} <<<")
    print(f"\n  Analyzing stored loss histories (no re-running)...")
    
    # Analyze each run's loss history to find iterations-to-threshold
    iterations_to_threshold_list = []
    threshold_reached_list = []
    
    for run_idx, loss_history in enumerate(all_results['loss_histories']):
        iter_to_thresh = find_iteration_to_threshold(loss_history, average_final_loss)
        iterations_to_threshold_list.append(iter_to_thresh)
        
        reached = iter_to_thresh < 101  # 101 is the penalty value
        threshold_reached_list.append(reached)
        
        status = f"Iter={iter_to_thresh}" if reached else "NOT REACHED (penalty=101)"
        print(f"    Run {run_idx + 1}: {status}")
    
    # Compile final results - include ALL runs (with penalty) in average
    successful_iters = [i for i in iterations_to_threshold_list if i < 101]
    success_count = len(successful_iters)
    success_rate = success_count / n_runs * 100
    
    final_results = {
        # Core metrics
        'accuracy': all_results['accuracy'],
        'precision': all_results['precision'],
        'recall': all_results['recall'],
        'f1_score': all_results['f1_score'],
        'confusion_matrices': all_results['confusion_matrices'],
        'best_losses': all_results['best_losses'],
        'iterations': all_results['iterations'],
        
        # Iteration-based tracking (ALL runs including penalty)
        'iterations_to_threshold': iterations_to_threshold_list,
        'threshold_reached': threshold_reached_list,
        'loss_histories': all_results['loss_histories'],
        
        # Threshold metadata
        'loss_threshold': average_final_loss,
        'loss_threshold_std': loss_std,
        
        # Success rate analysis
        'success_count': success_count,
        'success_rate': success_rate,
        'successful_iterations': successful_iters,
    }
    
    return final_results


# ==============================================================================
# 7. MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("BASELINE ACOR (SOCHA-ACOR) FOR CANCER CLASSIFICATION")
    print("Single-Phase Evaluation with Post-Hoc Iteration Analysis")
    print("=" * 70)
    print(f"Architecture: 9 inputs, 6 hidden (ReLU), 1 output (Sigmoid)")
    print(f"Total weights: {FNN.get_num_weights(9, 6, 1)}")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Evaluation: 50 independent runs")
    print()
    
    # Run single-phase evaluation with post-hoc analysis
    results = evaluate_baseline_acor_single_phase(X_train, X_test, y_train, y_test, n_runs=50)
    
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
    print(f"Target Threshold (Avg Final Loss): {results['loss_threshold']:.6f}")
    print(f"Success Rate: {results['success_rate']:.1f}% ({results['success_count']}/50 runs reached threshold)")
    
    # Compute average iterations INCLUDING penalty (101) for runs that didn't reach threshold
    avg_iter_with_penalty = np.mean(results['iterations_to_threshold'])
    std_iter_with_penalty = np.std(results['iterations_to_threshold'])
    min_iter = np.min(results['iterations_to_threshold'])
    max_iter = np.max(results['iterations_to_threshold'])
    
    print(f"\nIterations to Threshold (ALL runs, penalty=101 for failures):")
    print(f"  Mean: {avg_iter_with_penalty:.1f} ± {std_iter_with_penalty:.1f}")
    print(f"  Min:  {min_iter}, Max: {max_iter}")
    
    if results['successful_iterations']:
        # Also show stats for successful runs only (for reference)
        avg_successful = np.mean(results['successful_iterations'])
        std_successful = np.std(results['successful_iterations'])
        print(f"\nIterations to Threshold (successful runs only, for reference):")
        print(f"  Mean: {avg_successful:.1f} ± {std_successful:.1f}")
        
        # Show if threshold was reached during initialization vs optimization
        reached_in_init = sum(1 for i in results['successful_iterations'] if i == 0)
        reached_in_optim = len(results['successful_iterations']) - reached_in_init
        print(f"\n  Reached during initialization (Iter = 0): {reached_in_init}")
        print(f"  Reached during optimization (Iter > 0): {reached_in_optim}")
    
    # Confusion matrix
    avg_cm = np.mean(results['confusion_matrices'], axis=0)
    print(f"\nAverage Confusion Matrix:")
    print(avg_cm)
    
    # Best run
    best_run_idx = np.argmax(results['accuracy'])
    best_accuracy = results['accuracy'][best_run_idx]
    print(f"\nBest Run: {best_run_idx + 1} with accuracy {best_accuracy:.4f}")
    
    # ==================================================================
    # CREATE PERFORMANCE PLOT (Single plot - no pie chart)
    # ==================================================================
    output_dir = os.path.dirname(__file__)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Performance metrics bar chart
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    means = [np.mean(results[m]) for m in metrics]
    stds = [np.std(results[m]) for m in metrics]
    
    bars = ax.bar(metrics, means, yerr=stds, capsize=5, 
                  color=['skyblue', 'orange', 'green', 'red'], alpha=0.7)
    ax.set_ylim(0, 1)
    ax.set_title('Baseline SOCHA-ACOR - Performance Metrics\n(50 runs)')
    ax.set_ylabel('Score')
    ax.grid(True, alpha=0.3)
    
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + std + 0.02, 
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cancer_acor_performance.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nPerformance plot saved to: cancer_acor_performance.png")
    
    # ==================================================================
    # SUMMARY FOR COMPARISON WITH HYBRID
    # ==================================================================
    print("\n" + "=" * 70)
    print("SUMMARY FOR COMPARISON (Use these values for Hybrid evaluation)")
    print("=" * 70)
    print(f"Algorithm: Baseline SOCHA-ACOR")
    print(f"Loss Threshold (for Hybrid): {results['loss_threshold']:.6f}")
    print(f"Success Rate: {results['success_rate']:.1f}%")
    print(f"Avg Iterations to Threshold (with penalty): {avg_iter_with_penalty:.1f}")
    print(f"Avg Total Iterations: {np.mean(results['iterations']):.1f}")
    print(f"Avg Final Accuracy: {np.mean(results['accuracy']):.4f}")
    print(f"Avg Final Loss: {np.mean(results['best_losses']):.6f}")
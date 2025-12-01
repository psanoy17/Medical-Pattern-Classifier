"""
Levenberg-Marquardt Local Search Module for ACOR-LM Algorithm

This module implements the LM Fine-Tuning module as described in the thesis:
- Computes Jacobian matrix of FNN's error vector with respect to weights
- Approximates Hessian matrix
- Implements trust-region update that interpolates between Gauss-Newton and gradient-descent
- Dynamically adjusts damping factor μ based on step success
"""

import numpy as np
from typing import Tuple, Optional
import warnings


class LevenbergMarquardt:
    """
    Levenberg-Marquardt optimizer for local search refinement
    
    Implements trust-region update that interpolates between:
    - Gauss-Newton method (fast convergence near optimum)
    - Gradient descent (robust far from optimum)
    
    The damping factor μ controls this interpolation:
    - Small μ: More like Gauss-Newton
    - Large μ: More like gradient descent
    """
    
    def __init__(self, 
                 initial_mu: float = 0.001,
                 max_iterations: int = 50,
                 convergence_threshold: float = 1e-6,
                 mu_increase_factor: float = 10.0,
                 mu_decrease_factor: float = 0.1,
                 max_mu: float = 1e10,
                 min_mu: float = 1e-20):
        """
        Initialize Levenberg-Marquardt optimizer
        
        Args:
            initial_mu: Initial damping factor
            max_iterations: Maximum LM iterations
            convergence_threshold: Loss threshold for convergence
            mu_increase_factor: Factor to increase μ when step fails
            mu_decrease_factor: Factor to decrease μ when step succeeds
            max_mu: Maximum allowed damping factor
            min_mu: Minimum allowed damping factor
        """
        self.initial_mu = initial_mu
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.mu_increase_factor = mu_increase_factor
        self.mu_decrease_factor = mu_decrease_factor
        self.max_mu = max_mu
        self.min_mu = min_mu
        
    def optimize(self, 
                 model, 
                 X_train: np.ndarray, 
                 y_train: np.ndarray,
                 initial_weights: np.ndarray) -> Tuple[np.ndarray, float, int]:
        """
        Optimize weights using Levenberg-Marquardt algorithm
        
        Args:
            model: Neural network model with set_weights() and forward() methods
            X_train: Training features
            y_train: Training labels
            initial_weights: Starting weight vector
            
        Returns:
            Tuple of (optimized_weights, final_loss, iterations_used)
        """
        weights = initial_weights.copy()
        mu = self.initial_mu
        iteration = 0
        
        model.set_weights(weights)
        current_loss = self._compute_loss(model, X_train, y_train)
        
        for iteration in range(self.max_iterations):
            # Compute Jacobian and error vector
            jacobian = self._compute_jacobian(model, X_train, y_train, weights)
            error_vector = self._compute_error_vector(model, X_train, y_train)
            
            # Approximate Hessian: H ≈ J^T * J
            hessian_approx = jacobian.T @ jacobian
            
            # Damped Hessian: H + μI
            hessian_damped = hessian_approx + mu * np.eye(hessian_approx.shape[0])
            
            # Solve for weight update: (H + μI) * Δw = J^T * e
            try:
                delta_weights = np.linalg.solve(hessian_damped, jacobian.T @ error_vector)
            except np.linalg.LinAlgError:
                # Use pseudo-inverse if matrix is singular
                delta_weights = np.linalg.pinv(hessian_damped) @ (jacobian.T @ error_vector)
            
            # Apply weight update
            new_weights = weights - delta_weights
            model.set_weights(new_weights)
            new_loss = self._compute_loss(model, X_train, y_train)
            
            # Adaptive damping factor adjustment
            if new_loss < current_loss:
                # Step succeeded: accept update and decrease μ
                weights = new_weights
                current_loss = new_loss
                mu = max(mu * self.mu_decrease_factor, self.min_mu)
                
                # Check convergence
                if current_loss < self.convergence_threshold:
                    break
            else:
                # Step failed: reject update and increase μ
                mu = min(mu * self.mu_increase_factor, self.max_mu)
                if mu >= self.max_mu:
                    break
        
        return weights, current_loss, iteration + 1
    
    def _compute_loss(self, model, X_train: np.ndarray, y_train: np.ndarray) -> float:
        """
        Compute binary cross-entropy loss
        
        BCE = -mean(y * log(p) + (1-y) * log(1-p))
        """
        y_pred = model.forward(X_train)
        eps = 1e-8
        loss = -np.mean(y_train * np.log(y_pred + eps) + (1 - y_train) * np.log(1 - y_pred + eps))
        return loss
    
    def _compute_error_vector(self, model, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Compute error vector (predicted - actual)"""
        y_pred = model.forward(X_train)
        return y_pred - y_train
    
    def _compute_jacobian(self, model, X_train: np.ndarray, y_train: np.ndarray, 
                          weights: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian matrix using finite differences
        
        J[i,j] = ∂error[i] / ∂weight[j]
        """
        n_samples = X_train.shape[0]
        n_weights = len(weights)
        epsilon = 1e-6
        
        jacobian = np.zeros((n_samples, n_weights))
        
        model.set_weights(weights)
        error_base = self._compute_error_vector(model, X_train, y_train)
        
        for i in range(n_weights):
            weights_perturbed = weights.copy()
            weights_perturbed[i] += epsilon
            model.set_weights(weights_perturbed)
            error_perturbed = self._compute_error_vector(model, X_train, y_train)
            jacobian[:, i] = (error_perturbed - error_base) / epsilon
        
        return jacobian


class MultipleColonyACOR:
    """
    Multiple Colony ACOR with Levenberg-Marquardt Local Search
    
    This is the hybrid ACOR-LM algorithm that combines:
    1. Multiple parallel ACOR colonies for diverse exploration
    2. Levenberg-Marquardt local search for fine-tuning
    3. Inter-colony solution sharing for information exchange
    
    Key Components:
    - Global stagnation tracking across all colonies
    - Local stagnation detection per colony (triggers LM)
    - Periodic solution sharing between colonies
    
    Parameters:
        obj_func: Objective function to minimize (BCE loss)
        dim: Dimensionality of search space (number of weights)
        n_colonies: Number of parallel colonies (default: 3)
        n_ants: Number of new solutions generated per iteration per colony (default: 2)
        n_samples: Archive size k per colony (default: 148)
        q: Locality parameter for Gaussian kernel (default: 0.95)
        xi: Convergence speed parameter (default: 0.98)
        max_iter: Maximum global iterations (default: 100)
        patience: Global stagnation patience (default: 15)
        local_patience: Local stagnation patience for LM trigger (default: 5)
        sharing_frequency: Iterations between inter-colony sharing (default: 10)
        sharing_ratio: Fraction of solutions to share (default: 0.1)
        seed: Random seed for reproducibility
    """
    
    def __init__(self, 
                 obj_func,
                 dim: int,
                 n_colonies: int = 3,
                 n_ants: int = 2,
                 n_samples: int = 148,
                 q: float = 0.95,
                 xi: float = 0.98,
                 max_iter: int = 100,
                 patience: int = 15,
                 local_patience: int = 5,
                 sharing_frequency: int = 10,
                 sharing_ratio: float = 0.1,
                 seed: int = 42):
        """Initialize Multiple Colony ACOR with LM local search"""
        self.obj_func = obj_func
        self.dim = dim
        self.n_colonies = n_colonies
        self.n_ants = n_ants
        self.n_samples = n_samples
        self.q = q
        self.xi = xi
        self.max_iter = max_iter
        self.patience = patience
        self.local_patience = local_patience
        self.sharing_frequency = sharing_frequency
        self.sharing_ratio = sharing_ratio
        self.seed = seed
        
        # Initialize LM optimizer
        self.lm_optimizer = LevenbergMarquardt()
        
        # Initialize colonies
        self.colonies = []
        for i in range(n_colonies):
            colony = SingleColonyACOR(
                obj_func=obj_func,
                dim=dim,
                n_ants=n_ants,
                n_samples=n_samples,
                q=q,
                xi=xi,
                max_iter=max_iter,
                patience=local_patience,
                seed=seed + i
            )
            self.colonies.append(colony)
    
    def optimize(self, lb: float, ub: float, model, X_train: np.ndarray, y_train: np.ndarray):
        """
        Main optimization loop with multiple colonies and LM local search
        
        Args:
            lb: Lower bound for weight initialization
            ub: Upper bound for weight initialization
            model: Neural network model
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Tuple of (best_weights, best_loss, iterations_used)
        """
        # Initialize all colony archives
        for colony in self.colonies:
            colony.initialize_archive(lb, ub)
        
        # Global tracking variables
        best_global_weights = None
        best_global_loss = np.inf
        global_stagnation_counter = 0
        best_global_iteration = 0
        
        # Statistics tracking
        total_lm_applications = 0
        
        # ==================================================================
        # Main Optimization Loop
        # ==================================================================
        for iteration in range(self.max_iter):
            global_improved_this_iteration = False
            
            # Run each colony for one iteration
            for colony_idx, colony in enumerate(self.colonies):
                colony.run_iteration()
                
                # Check for local stagnation -> trigger LM
                if colony.has_local_stagnation():
                    lm_weights, lm_loss, lm_iters = self.lm_optimizer.optimize(
                        model, X_train, y_train, colony.best_weights
                    )
                    total_lm_applications += 1
                    
                    # Inject LM solution if it improved
                    if lm_loss < colony.best_loss:
                        colony.inject_lm_solution(lm_weights, lm_loss)
                
                # Update global best
                if colony.best_loss < best_global_loss:
                    best_global_loss = colony.best_loss
                    best_global_weights = colony.best_weights.copy()
                    best_global_iteration = iteration
                    global_improved_this_iteration = True
            
            # Update global stagnation counter
            if global_improved_this_iteration:
                global_stagnation_counter = 0
            else:
                global_stagnation_counter += 1
            
            # Inter-colony solution sharing
            if (iteration + 1) % self.sharing_frequency == 0:
                self._share_solutions_between_colonies()
            
            # Global stopping criteria (patience-based)
            if global_stagnation_counter > self.patience:
                break
        
        return best_global_weights, best_global_loss, iteration + 1
    
    def _share_solutions_between_colonies(self):
        """
        Share best solutions between colonies
        
        Collects best solutions from all colonies and injects them
        into other colonies to promote information exchange.
        """
        # Collect best solutions from all colonies
        all_best_solutions = []
        for colony in self.colonies:
            if colony.best_weights is not None:
                all_best_solutions.append((colony.best_weights.copy(), colony.best_loss))
        
        # Sort by fitness (ascending - lower is better)
        all_best_solutions.sort(key=lambda x: x[1])
        
        # Determine number of solutions to share
        n_share = max(1, int(self.n_samples * self.sharing_ratio))
        top_solutions = all_best_solutions[:n_share]
        
        # Inject top solutions into each colony
        for colony in self.colonies:
            for weights, loss in top_solutions:
                # Replace worst solution in archive
                worst_idx = np.argmax(colony.archive_fitness)
                if loss < colony.archive_fitness[worst_idx]:
                    colony.archive_solutions[worst_idx] = weights.copy()
                    colony.archive_fitness[worst_idx] = loss
                    
                    # Update ranks and neighbor list
                    colony.archive_ranks = colony._rank_ascending_with_random_ties(colony.archive_fitness)
                    colony._update_neighbor_list()
                    
                    # Update colony best if improved
                    if loss < colony.best_loss:
                        colony.best_weights = weights.copy()
                        colony.best_loss = loss


class SingleColonyACOR:
    """
    Single Colony ACOR with SOCHA-ACOR Implementation
    
    This class implements SOCHA's original ACOR algorithm for a single colony,
    matching the baseline implementation exactly.
    
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
        n_samples: Archive size k (default: 148)
        q: Locality parameter for Gaussian kernel (default: 0.95)
        xi: Convergence speed parameter (default: 0.98)
        max_iter: Maximum iterations (default: 100)
        patience: Local stagnation patience for LM trigger (default: 5)
        seed: Random seed for reproducibility
    """
    
    def __init__(self, obj_func, dim, n_ants, n_samples, q, xi, max_iter, patience, seed):
        self.obj_func = obj_func
        self.dim = dim
        self.n_ants = n_ants
        self.n_samples = n_samples  # Archive size (k)
        self.q = q  # Locality parameter for Gaussian kernel
        self.xi = xi  # Convergence speed parameter
        self.max_iter = max_iter
        self.patience = patience  # Local stagnation threshold for LM trigger
        self.seed = seed
        
        if seed > 0:
            np.random.seed(seed)
        
        # Solution archive (matching baseline naming)
        self.archive_solutions = None  # Will hold (n_samples, dim) array
        self.archive_fitness = None  # Will hold fitness values
        self.archive_ranks = None  # Solution ranks
        self.neighbor_list = None  # Neighbor indices for each solution
        
        # Best solution tracking
        self.best_weights = None
        self.best_loss = np.inf
        
        # Iteration tracking (matching baseline approach)
        self.iteration = 0
        self.best_iteration = 0  # Track when last improvement occurred
    
    def initialize_archive(self, lb, ub):
        """
        Initialize the solution archive
        
        Generates n_samples random solutions within bounds and evaluates them.
        
        Args:
            lb: Lower bound for weight initialization
            ub: Upper bound for weight initialization
        """
        # Generate random solutions
        self.archive_solutions = np.random.uniform(lb, ub, (self.n_samples, self.dim))
        
        # Evaluate all solutions
        self.archive_fitness = np.array([self.obj_func(w) for w in self.archive_solutions])
        
        # Initialize ranks and neighbor list
        self.archive_ranks = self._rank_ascending_with_random_ties(self.archive_fitness)
        self._update_neighbor_list()
        
        # Find initial best solution
        best_idx = np.argmin(self.archive_fitness)
        self.best_weights = self.archive_solutions[best_idx].copy()
        self.best_loss = self.archive_fitness[best_idx]
        self.best_iteration = 0
    
    def run_iteration(self):
        """
        Run one ACOR iteration using SOCHA-ACOR mechanism
        
        This matches the baseline implementation exactly:
        1. Check for archive convergence
        2. Generate new solutions using ACOR mechanism
        3. Evaluate new solutions
        4. Update archive (keep best n_samples)
        5. Track improvements for stagnation detection
        """
        if self.archive_solutions is None:
            raise ValueError("Archive not initialized. Call initialize_archive first.")
        
        # Check for archive convergence (all solutions identical)
        if np.sum(np.std(self.archive_solutions, axis=0)) == 0:
            return
        
        # Generate new solutions using ACOR mechanism
        new_solutions = self._generate_new_solutions()
        
        if new_solutions is None or len(new_solutions) == 0:
            return
        
        # Evaluate new solutions
        new_fitness = np.array([self.obj_func(w) for w in new_solutions])
        
        # Combine archive with new solutions
        all_solutions = np.vstack([self.archive_solutions, new_solutions])
        all_fitness = np.concatenate([self.archive_fitness, new_fitness])
        
        # Rank all solutions
        all_ranks = self._rank_ascending_with_random_ties(all_fitness)
        
        # Keep only top n_samples solutions (elitism)
        keep_indices = all_ranks <= self.n_samples
        self.archive_solutions = all_solutions[keep_indices]
        self.archive_fitness = all_fitness[keep_indices]
        self.archive_ranks = self._rank_ascending_with_random_ties(self.archive_fitness)
        
        # Update neighbor list for new archive
        self._update_neighbor_list()
        
        # Check for improvement (matching baseline approach)
        current_best_fitness = np.min(new_fitness)
        if current_best_fitness < self.best_loss:
            self.best_loss = float(current_best_fitness)
            best_idx = int(np.argmin(new_fitness))
            self.best_weights = new_solutions[best_idx].copy()
            self.best_iteration = self.iteration
        
        self.iteration += 1
    
    def _generate_new_solutions(self):
        """
        Generate new solutions using SOCHA's ACOR mechanism
        
        This is the core of the ACOR algorithm (matching baseline exactly):
        1. Select a guiding solution using Gaussian kernel PDF
        2. Apply QR decomposition for orthogonal rotation
        3. Compute adaptive standard deviation from neighbor distances
        4. Sample new solution in rotated space
        5. Transform back to original space
        
        Returns:
            Array of new solutions (n_ants x dim) or None if generation fails
        """
        num_solutions, num_dimensions = self.archive_solutions.shape
        new_solutions = np.empty((self.n_ants, num_dimensions), dtype=float)

        # Compute selection probabilities using Gaussian kernel PDF
        # Solutions with better rank (lower) have higher probability
        ranks = np.arange(1, num_solutions + 1)
        selection_probs = self._gaussian_kernel_pdf(ranks, mean=1.0, std=self.q * self.n_samples)
        selection_probs = selection_probs / selection_probs.sum()
        
        # Select guiding solutions probabilistically
        selected_indices = np.random.choice(num_solutions, size=self.n_ants, 
                                            replace=True, p=selection_probs)

        # Generate each new solution
        for solution_idx in range(self.n_ants):
            guide_idx = selected_indices[solution_idx]
            
            # Center archive around guiding solution
            centered_archive = self.archive_solutions - self.archive_solutions[guide_idx]
            rotated_archive = centered_archive.copy()
            
            # Get available neighbors for orthogonal basis construction
            available_neighbors = self.neighbor_list[guide_idx].copy()
            
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
            neighbor_indices = self.neighbor_list[guide_idx]
            adaptive_std = np.array([
                np.sum(np.abs(rotated_archive[neighbor_indices, d] - rotated_archive[guide_idx, d])) / (self.n_samples - 1)
                for d in range(num_dimensions)
            ])
            
            # Sample new solution in rotated space
            new_solution_rotated = np.random.normal(
                loc=rotated_archive[guide_idx],
                scale=adaptive_std * self.xi,  # Scale by convergence parameter
                size=(num_dimensions,)
            )
            
            # Transform back to original space
            new_solution = (rotation_matrix @ new_solution_rotated) + self.archive_solutions[guide_idx]
            new_solutions[solution_idx] = new_solution
            
        return new_solutions
    
    def _update_neighbor_list(self):
        """
        Update neighbor list for all archive solutions
        
        Each solution's neighbors are all other solutions in the archive.
        """
        self.neighbor_list = np.empty((self.n_samples, self.n_samples - 1), dtype=int)
        for i in range(self.n_samples):
            self.neighbor_list[i] = np.delete(np.arange(self.n_samples), i)
    
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
    
    def has_local_stagnation(self) -> bool:
        """
        Check if colony has local stagnation (for LM trigger)
        
        Uses the same approach as baseline ACOR:
        - Compares current iteration with last improvement iteration
        - Returns True when no improvement for 'patience' iterations
        
        Returns:
            True if local stagnation detected, False otherwise
        """
        return (self.iteration - self.best_iteration) > self.patience
    
    def inject_lm_solution(self, lm_weights: np.ndarray, lm_loss: float):
        """
        Inject LM-refined solution back into archive
        
        Replaces the worst solution in the archive with the LM solution
        and updates tracking variables.
        
        Args:
            lm_weights: Weight vector from LM optimization
            lm_loss: Loss value of LM solution
        """
        # Replace worst solution with LM solution
        worst_idx = np.argmax(self.archive_fitness)
        self.archive_solutions[worst_idx] = lm_weights.copy()
        self.archive_fitness[worst_idx] = lm_loss
        
        # Update ranks and neighbor list
        self.archive_ranks = self._rank_ascending_with_random_ties(self.archive_fitness)
        self._update_neighbor_list()
        
        # Update colony's best if LM improved
        if lm_loss < self.best_loss:
            self.best_weights = lm_weights.copy()
            self.best_loss = lm_loss
            self.best_iteration = self.iteration  # Reset stagnation tracking
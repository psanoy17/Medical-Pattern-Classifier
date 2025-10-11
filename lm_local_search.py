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
            model: FNN model with forward and predict methods
            X_train: Training input data
            y_train: Training target data
            initial_weights: Initial weight vector from ACOR
            
        Returns:
            Tuple of (optimized_weights, final_loss, iterations_used)
        """
        weights = initial_weights.copy()
        mu = self.initial_mu
        iteration = 0
        
        # Compute initial loss
        model.set_weights(weights)
        current_loss = self._compute_loss(model, X_train, y_train)
        
        for iteration in range(self.max_iterations):
            # Compute Jacobian matrix
            J = self._compute_jacobian(model, X_train, y_train, weights)
            
            # Compute error vector
            error = self._compute_error_vector(model, X_train, y_train)
            
            # Compute approximate Hessian: H ≈ J^T * J
            H = J.T @ J
            
            # Add damping: H + μI
            H_damped = H + mu * np.eye(H.shape[0])
            
            # Solve for weight update: (H + μI) * Δw = J^T * e
            try:
                delta_w = np.linalg.solve(H_damped, J.T @ error)
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse if matrix is singular
                delta_w = np.linalg.pinv(H_damped) @ (J.T @ error)
            
            # Try the update
            new_weights = weights - delta_w
            
            # Compute new loss
            model.set_weights(new_weights)
            new_loss = self._compute_loss(model, X_train, y_train)
            
            # Check if step was successful
            if new_loss < current_loss:
                # Step successful: decrease μ (favor Gauss-Newton)
                weights = new_weights
                current_loss = new_loss
                mu = max(mu * self.mu_decrease_factor, self.min_mu)
                
                # Check convergence
                if current_loss < self.convergence_threshold:
                    break
                    
            else:
                # Step failed: increase μ (favor gradient descent)
                mu = min(mu * self.mu_increase_factor, self.max_mu)
                
                # If μ becomes too large, stop
                if mu >= self.max_mu:
                    break
        
        return weights, current_loss, iteration + 1
    
    def _compute_loss(self, model, X_train: np.ndarray, y_train: np.ndarray) -> float:
        """Compute binary cross-entropy loss"""
        y_pred = model.forward(X_train)
        eps = 1e-8
        loss = -np.mean(y_train * np.log(y_pred + eps) + (1 - y_train) * np.log(1 - y_pred + eps))
        return loss
    
    def _compute_error_vector(self, model, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Compute error vector (predicted - actual)"""
        y_pred = model.forward(X_train)
        return y_pred - y_train
    
    def _compute_jacobian(self, model, X_train: np.ndarray, y_train: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian matrix of error vector with respect to weights
        Using finite differences for numerical stability
        """
        n_samples = X_train.shape[0]
        n_weights = len(weights)
        epsilon = 1e-6
        
        # Initialize Jacobian matrix
        J = np.zeros((n_samples, n_weights))
        
        # Compute error at current weights
        model.set_weights(weights)
        error_base = self._compute_error_vector(model, X_train, y_train)
        
        # Compute Jacobian using finite differences
        for i in range(n_weights):
            # Create perturbed weights
            weights_pert = weights.copy()
            weights_pert[i] += epsilon
            
            # Compute error with perturbed weights
            model.set_weights(weights_pert)
            error_pert = self._compute_error_vector(model, X_train, y_train)
            
            # Compute partial derivative
            J[:, i] = (error_pert - error_base) / epsilon
        
        return J

class MultipleColonyACOR:
    """
    Multiple Colony ACOR with Levenberg-Marquardt Local Search
    
    Implements the parallel multiple colony strategy as described in the thesis:
    - 3 separate ant colonies
    - Colonies share best solutions every 10 rounds
    - 10% of best solutions from one colony replace weaker ones in another
    - Each colony uses ACOR with LM local search
    """
    
    def __init__(self, 
                 obj_func,
                 dim: int,
                 n_colonies: int = 3,
                 n_ants: int = 30,
                 n_samples: int = 120,
                 q: float = 0.1,
                 xi: float = 0.85,
                 max_iter: int = 100,
                 patience: int = 15,
                 sharing_frequency: int = 10,
                 sharing_ratio: float = 0.1,
                 seed: int = 42):
        """
        Initialize Multiple Colony ACOR
        
        Args:
            obj_func: Objective function to optimize
            dim: Dimension of the problem
            n_colonies: Number of parallel colonies
            n_ants: Number of ants per colony
            n_samples: Archive size per colony
            q: Locality parameter
            xi: Convergence pressure
            max_iter: Maximum iterations
            patience: Early stopping patience
            sharing_frequency: How often colonies share solutions (every N rounds)
            sharing_ratio: Fraction of solutions to share (0.1 = 10%)
            seed: Random seed
        """
        self.obj_func = obj_func
        self.dim = dim
        self.n_colonies = n_colonies
        self.n_ants = n_ants
        self.n_samples = n_samples
        self.q = q
        self.xi = xi
        self.max_iter = max_iter
        self.patience = patience
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
                patience=patience,
                seed=seed + i  # Different seed for each colony
            )
            self.colonies.append(colony)
    
    def optimize(self, lb: float, ub: float, model, X_train: np.ndarray, y_train: np.ndarray):
        """
        Optimize using multiple colonies with LM local search
        
        Args:
            lb: Lower bound for weights
            ub: Upper bound for weights
            model: FNN model
            X_train: Training data
            y_train: Training labels
            
        Returns:
            Tuple of (best_weights, best_loss, total_iterations)
        """
        # Initialize all colonies
        for colony in self.colonies:
            colony.initialize_archive(lb, ub)
        
        best_global_weights = None
        best_global_loss = np.inf
        total_iterations = 0
        
        for iteration in range(self.max_iter):
            # Run one iteration for each colony
            for colony_idx, colony in enumerate(self.colonies):
                # Run ACOR iteration
                colony.run_iteration()
                total_iterations += 1
                
                # Apply LM local search to best solution
                if colony.best_weights is not None:
                    lm_weights, lm_loss, lm_iterations = self.lm_optimizer.optimize(
                        model, X_train, y_train, colony.best_weights
                    )
                    
                    # Update colony's best if LM improved it
                    if lm_loss < colony.best_loss:
                        colony.best_weights = lm_weights
                        colony.best_loss = lm_loss
                    
                    # Update global best
                    if lm_loss < best_global_loss:
                        best_global_loss = lm_loss
                        best_global_weights = lm_weights.copy()
            
            # Share solutions between colonies every sharing_frequency iterations
            if (iteration + 1) % self.sharing_frequency == 0:
                self._share_solutions()
            
            # Check for early stopping
            if self._check_convergence():
                break
        
        return best_global_weights, best_global_loss, total_iterations
    
    def _share_solutions(self):
        """Share best solutions between colonies"""
        # Collect best solutions from all colonies
        all_solutions = []
        for colony in self.colonies:
            if colony.best_weights is not None:
                all_solutions.append((colony.best_weights, colony.best_loss))
        
        # Sort by fitness (lower loss is better)
        all_solutions.sort(key=lambda x: x[1])
        
        # Share top solutions
        n_share = max(1, int(self.n_samples * self.sharing_ratio))
        top_solutions = all_solutions[:n_share]
        
        # Replace worst solutions in each colony with shared solutions
        for colony in self.colonies:
            if len(top_solutions) > 0:
                # Replace worst solutions with shared ones
                for i, (weights, loss) in enumerate(top_solutions):
                    if i < len(colony.archive_weights):
                        # Replace worst solution
                        worst_idx = np.argmax(colony.archive_losses)
                        colony.archive_weights[worst_idx] = weights.copy()
                        colony.archive_losses[worst_idx] = loss
                        
                        # Update colony's best if needed
                        if loss < colony.best_loss:
                            colony.best_weights = weights.copy()
                            colony.best_loss = loss
    
    def _check_convergence(self) -> bool:
        """Check if all colonies have converged"""
        for colony in self.colonies:
            if not colony.has_converged():
                return False
        return True

class SingleColonyACOR:
    """Single colony ACOR implementation for use in multiple colony system"""
    
    def __init__(self, obj_func, dim, n_ants, n_samples, q, xi, max_iter, patience, seed):
        self.obj_func = obj_func
        self.dim = dim
        self.n_ants = n_ants
        self.n_samples = n_samples
        self.q = q
        self.xi = xi
        self.max_iter = max_iter
        self.patience = patience
        self.seed = seed
        
        # Initialize random number generator
        if seed > 0:
            np.random.seed(seed)
        
        # Colony state
        self.archive_weights = None
        self.archive_losses = None
        self.best_weights = None
        self.best_loss = np.inf
        self.iteration = 0
        self.no_improvement_count = 0
        
    def initialize_archive(self, lb, ub):
        """Initialize the solution archive"""
        self.archive_weights = np.random.uniform(lb, ub, (self.n_samples, self.dim))
        self.archive_losses = np.array([self.obj_func(w) for w in self.archive_weights])
        
        # Find best solution
        best_idx = np.argmin(self.archive_losses)
        self.best_weights = self.archive_weights[best_idx].copy()
        self.best_loss = self.archive_losses[best_idx]
    
    def run_iteration(self):
        """Run one ACOR iteration"""
        if self.archive_weights is None:
            raise ValueError("Archive not initialized. Call initialize_archive first.")
        
        # Generate new solutions using ACOR
        new_weights = self._generate_solutions()
        new_losses = np.array([self.obj_func(w) for w in new_weights])
        
        # Combine with archive
        all_weights = np.vstack([self.archive_weights, new_weights])
        all_losses = np.concatenate([self.archive_losses, new_losses])
        
        # Keep best n_samples solutions
        sorted_indices = np.argsort(all_losses)
        self.archive_weights = all_weights[sorted_indices[:self.n_samples]]
        self.archive_losses = all_losses[sorted_indices[:self.n_samples]]
        
        # Update best solution
        if self.archive_losses[0] < self.best_loss:
            self.best_loss = self.archive_losses[0]
            self.best_weights = self.archive_weights[0].copy()
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
        
        self.iteration += 1
    
    def _generate_solutions(self):
        """Generate new solutions using ACOR mechanism"""
        # Simplified ACOR solution generation
        # In practice, this would implement the full SOCHA-ACOR algorithm
        new_weights = []
        
        for _ in range(self.n_ants):
            # Select a solution from archive based on fitness
            probs = 1.0 / (self.archive_losses + 1e-8)
            probs = probs / probs.sum()
            
            selected_idx = np.random.choice(len(self.archive_weights), p=probs)
            selected_weights = self.archive_weights[selected_idx]
            
            # Add Gaussian noise
            noise = np.random.normal(0, 0.1, self.dim)
            new_weight = selected_weights + noise
            
            new_weights.append(new_weight)
        
        return np.array(new_weights)
    
    def has_converged(self) -> bool:
        """Check if colony has converged"""
        return (self.iteration >= self.max_iter or 
                self.no_improvement_count >= self.patience)



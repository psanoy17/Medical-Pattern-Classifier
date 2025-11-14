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
    def __init__(self, 
                 obj_func,
                 dim: int,
                 n_colonies: int = 3,
                 n_ants: int = 30,
                 n_samples: int = 120,
                 q: float = 0.1,
                 xi: float = 0.85,
                 max_iter: int = 100,
                 patience: int = 15,  # Global stagnation patience
                 local_patience: int = 5,  # Local stagnation threshold for LM trigger
                 sharing_frequency: int = 10,
                 sharing_ratio: float = 0.1,
                 seed: int = 42):
        """
        Initialize Multiple Colony ACOR with Global and Local stagnation tracking
        
        Args:
            patience: Global stagnation patience (15 iterations)
            local_patience: Local stagnation threshold for LM trigger (5-7 iterations)
        """
        self.obj_func = obj_func
        self.dim = dim
        self.n_colonies = n_colonies
        self.n_ants = n_ants
        self.n_samples = n_samples
        self.q = q
        self.xi = xi
        self.max_iter = max_iter
        self.patience = patience  # Global stagnation patience
        self.local_patience = local_patience  # Local LM trigger threshold
        self.sharing_frequency = sharing_frequency
        self.sharing_ratio = sharing_ratio
        self.seed = seed
        
        self.lm_optimizer = LevenbergMarquardt()
        
        # Initialize colonies with LOCAL patience
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
                patience=local_patience,  # ✅ Use local_patience for LM trigger
                seed=seed + i
            )
            self.colonies.append(colony)
    
    def optimize(self, lb: float, ub: float, model, X_train: np.ndarray, y_train: np.ndarray):
        """
        Optimize using multiple colonies with CORRECTED stopping criteria
        
        Stopping Rules (Global):
        1. Global Best Solution hasn't improved for 15 iterations (patience)
        2. Reached 100 maximum global iterations (max_iter)
        
        LM Trigger (Local per colony):
        - Triggered when colony's local best hasn't improved for 5 iterations (local_patience)
        - Does NOT stop the algorithm, only applies LM refinement
        """
        # Initialize all colonies
        for colony in self.colonies:
            colony.initialize_archive(lb, ub)
        
        # ✅ GLOBAL tracking
        best_global_weights = None
        best_global_loss = np.inf
        global_stagnation_counter = 0  # Track global best stagnation
        
        # Statistics
        total_colony_runs = 0
        lm_applications = 0
        
        # ===== PHASE 1: ACOR with Adaptive LM Triggering =====
        for iteration in range(self.max_iter):
            # Track if global best improved this iteration
            global_improved_this_iteration = False
            
            for colony_idx, colony in enumerate(self.colonies):
                # Run ACOR iteration
                colony.run_iteration()
                total_colony_runs += 1
                
                # ✅ ADAPTIVE LM TRIGGER: Based on LOCAL stagnation
                if colony.has_local_stagnation():  # Colony's local best hasn't improved
                    print(f"  [Iter {iteration+1}] Colony {colony_idx+1} local stagnation, applying LM...")
                    
                    lm_weights, lm_loss, lm_iters = self.lm_optimizer.optimize(
                        model, X_train, y_train, colony.best_weights
                    )
                    lm_applications += 1
                    
                    if lm_loss < colony.best_loss:
                        colony.inject_lm_solution(lm_weights, lm_loss)
                        print(f"    → LM improved: {colony.best_loss:.6f} → {lm_loss:.6f}")
                
                # ✅ Update GLOBAL best
                if colony.best_loss < best_global_loss:
                    best_global_loss = colony.best_loss
                    best_global_weights = colony.best_weights.copy()
                    global_improved_this_iteration = True  # Mark improvement
            
            # ✅ Update GLOBAL stagnation counter
            if global_improved_this_iteration:
                global_stagnation_counter = 0  # Reset on global improvement
                print(f"  [Iter {iteration+1}] Global best improved: {best_global_loss:.6f}")
            else:
                global_stagnation_counter += 1  # Increment on global stagnation
            
            # Inter-colony communication
            if (iteration + 1) % self.sharing_frequency == 0:
                self._share_solutions()
            
            # ✅ GLOBAL STOPPING CRITERIA
            # Rule 1: Global best hasn't improved for 'patience' iterations
            if global_stagnation_counter >= self.patience:
                print(f"\n✅ Stopping: Global best stagnated for {self.patience} iterations")
                print(f"Last global improvement at iteration {iteration + 1 - self.patience}")
                break
            
            # Rule 2: Maximum iterations (handled by for loop)
            if iteration + 1 >= self.max_iter:
                print(f"\n✅ Stopping: Reached maximum {self.max_iter} iterations")
                break
        
        # ===== FINAL SUMMARY =====
        print(f"\n" + "=" * 60)
        print(f"OPTIMIZATION COMPLETE")
        print(f"=" * 60)
        print(f"Total LM applications: {lm_applications}")
        print(f"Global stagnation at stopping: {global_stagnation_counter} iterations")
        print(f"Final global best loss: {best_global_loss:.6f}")
        
        # ✅ Return best solution found during optimization (no final LM)
        return best_global_weights, best_global_loss, iteration + 1
    
    def _share_solutions(self):
        """Share best solutions between colonies"""
        all_solutions = []
        for colony in self.colonies:
            if colony.best_weights is not None:
                all_solutions.append((colony.best_weights, colony.best_loss))
        
        all_solutions.sort(key=lambda x: x[1])
        n_share = max(1, int(self.n_samples * self.sharing_ratio))
        top_solutions = all_solutions[:n_share]
        
        for colony in self.colonies:
            if len(top_solutions) > 0:
                for i, (weights, loss) in enumerate(top_solutions):
                    if i < len(colony.archive_weights):
                        worst_idx = np.argmax(colony.archive_losses)
                        colony.archive_weights[worst_idx] = weights.copy()
                        colony.archive_losses[worst_idx] = loss
                        
                        if loss < colony.best_loss:
                            colony.best_weights = weights.copy()
                            colony.best_loss = loss

class SingleColonyACOR:
    """Single colony ACOR with local stagnation tracking for LM trigger"""
    
    def __init__(self, obj_func, dim, n_ants, n_samples, q, xi, max_iter, patience, seed):
        self.obj_func = obj_func
        self.dim = dim
        self.n_ants = n_ants
        self.n_samples = n_samples
        self.q = q
        self.xi = xi
        self.max_iter = max_iter
        self.patience = patience  # ✅ This is now LOCAL stagnation threshold (5-7)
        self.seed = seed
        
        if seed > 0:
            np.random.seed(seed)
        
        # ✅ LOCAL tracking (per colony)
        self.archive_weights = None
        self.archive_losses = None
        self.best_weights = None
        self.best_loss = np.inf
        self.iteration = 0
        self.no_improvement_count = 0  # ✅ LOCAL stagnation counter
    
    def initialize_archive(self, lb, ub):
        """Initialize the solution archive"""
        self.archive_weights = np.random.uniform(lb, ub, (self.n_samples, self.dim))
        self.archive_losses = np.array([self.obj_func(w) for w in self.archive_weights])
        
        best_idx = np.argmin(self.archive_losses)
        self.best_weights = self.archive_weights[best_idx].copy()
        self.best_loss = self.archive_losses[best_idx]
    
    def run_iteration(self):
        """Run one ACOR iteration"""
        if self.archive_weights is None:
            raise ValueError("Archive not initialized. Call initialize_archive first.")
        
        # Generate new solutions
        new_weights = self._generate_solutions()
        new_losses = np.array([self.obj_func(w) for w in new_weights])
        
        # Combine with archive
        all_weights = np.vstack([self.archive_weights, new_weights])
        all_losses = np.concatenate([self.archive_losses, new_losses])
        
        # Keep best n_samples solutions
        sorted_indices = np.argsort(all_losses)
        self.archive_weights = all_weights[sorted_indices[:self.n_samples]]
        self.archive_losses = all_losses[sorted_indices[:self.n_samples]]
        
        # ✅ Update LOCAL best and stagnation counter
        if self.archive_losses[0] < self.best_loss:
            self.best_loss = self.archive_losses[0]
            self.best_weights = self.archive_weights[0].copy()
            self.no_improvement_count = 0  # Reset on local improvement
        else:
            self.no_improvement_count += 1  # Increment on local stagnation
        
        self.iteration += 1
    
    def has_local_stagnation(self) -> bool:
        """
        ✅ Check if colony has local stagnation (for LM trigger)
        
        Returns True when local best hasn't improved for 'patience' iterations
        This triggers LM but does NOT stop the algorithm
        """
        return self.no_improvement_count >= self.patience
    
    def inject_lm_solution(self, lm_weights: np.ndarray, lm_loss: float):
        """Inject LM-refined solution back into archive"""
        # Replace worst solution with LM solution
        worst_idx = np.argmax(self.archive_losses)
        self.archive_weights[worst_idx] = lm_weights.copy()
        self.archive_losses[worst_idx] = lm_loss
        
        # Update colony's local best if LM improved
        if lm_loss < self.best_loss:
            self.best_weights = lm_weights.copy()
            self.best_loss = lm_loss
            self.no_improvement_count = 0  # ✅ Reset local stagnation counter
    
    def _generate_solutions(self):
        """Generate new solutions using ACOR mechanism"""
        new_weights = []
        
        for _ in range(self.n_ants):
            probs = 1.0 / (self.archive_losses + 1e-8)
            probs = probs / probs.sum()
            
            selected_idx = np.random.choice(len(self.archive_weights), p=probs)
            selected_weights = self.archive_weights[selected_idx]
            
            noise = np.random.normal(0, 0.1, self.dim)
            new_weight = selected_weights + noise
            
            new_weights.append(new_weight)
        
        return np.array(new_weights)



import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import warnings
import pickle

# Set random seed for reproducibility
np.random.seed(42)

# 1. Load and preprocess the data
# --------------------------------------------------
data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'heart.csv'))

# Target variable is 'target' (already 0/1)
X = data.drop('target', axis=1).values
y = data['target'].values

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Define a simple FNN (single hidden layer, 6 neurons)
# --------------------------------------------------
class FNN:
    """
    Feedforward Neural Network (FNN) with a single hidden layer (6 neurons) and 1 output neuron.
    Architecture: Input -> Hidden(6) -> Output(1)
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
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

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.tanh(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = 1 / (1 + np.exp(-z2))  # Sigmoid for binary classification
        return a2.squeeze()

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)

    @staticmethod
    def get_num_weights(input_dim, hidden_dim, output_dim):
        return input_dim*hidden_dim + hidden_dim + hidden_dim*output_dim + output_dim

# 3. SOCHA-ACOR implementation (from R) with Medical Pattern Classifier hyperparameters
# --------------------------------------------------
class SOCHA_ACOR:
    def __init__(self, obj_func, dim, n_ants=30, n_samples=80, q=0.1, xi=0.85, max_iter=100, patience=15, seed=42):
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

    def optimize(self, lb, ub):
        """SOCHA-ACOR optimization matching R implementation"""
        e_abs = 1e-6  # Small error threshold
        e_rel = 1e-6
        max_value = 0  # opt value
        eval_count = 0
        last_impr = self.max_iter
        nl = np.empty((self.n_samples, self.n_samples - 1), dtype=int)
        iteration = 0

        # Initialize variables
        max_X = np.full(self.dim, np.nan)
        max_y = np.inf  # Start with high value for minimization

        p_X = None
        p_v = []

        # Initialize archive with random solutions
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

        # Initialize best from archive
        imax0 = int(np.argmin(p_v))
        max_y = float(p_v[imax0])
        max_X = p_X[imax0]
        best_iter = 0

        # Main optimization loop
        for iteration in range(self.max_iter):
            # Generate new points based on chosen distributions
            dist_mean = p_X
            if np.sum(np.std(dist_mean, axis=0)) == 0:
                return max_X, max_y, iteration + 1
            
            dist_rank = p_gr
            o_X = self._gen_X(dist_mean, dist_rank, nl, self.n_ants, self.q, self.n_samples, self.xi)

            if o_X is None or len(o_X) == 0:
                return max_X, max_y, iteration + 1

            # Evaluate new solutions
            y = self.obj_func(o_X)
            eval_count += len(o_X)

            # Add new solutions to population
            p_X = np.vstack([p_X, o_X])
            p_v = np.concatenate([p_v, y])
            p_gr = self._rank_asc_with_random_ties(p_v)

            # Keep best n_samples solutions
            idx_final = p_gr <= self.n_samples
            p_v = p_v[idx_final]
            p_gr = p_gr[idx_final]
            p_X = p_X[idx_final]

            # Recompute neighbor lists
            for i in range(self.n_samples):
                nl[i] = np.delete(np.arange(self.n_samples), i)

            # Check for improvement (for minimization, we want lower values)
            if np.min(y) < max_y:
                max_y = float(np.min(y))
                imax = int(np.argmin(y))
                max_X = o_X[imax]
                best_iter = iteration
                last_impr = eval_count
                
                if (abs(max_y - max_value) < abs(e_rel * max_value + e_abs)) or (max_y < max_value):
                    return max_X, max_y, iteration + 1

            print(f"Iteration {iteration+1}/{self.max_iter}, Best Loss: {max_y:.4f}")
            
            # Early stopping (convergence)
            if iteration - best_iter > self.patience:
                print(f"Early stopping at iteration {iteration+1}")
                return max_X, max_y, iteration + 1

        return max_X, max_y, self.max_iter

    def _rank_desc_with_random_ties(self, values):
        """Rank values in descending order with random ties"""
        n = len(values)
        # Shuffle to break ties randomly
        perm = np.random.permutation(n)
        shuffled_vals = values[perm]
        order = np.argsort(-shuffled_vals, kind='mergesort')
        ranks = np.empty(n, dtype=int)
        ranks[order] = np.arange(1, n + 1)
        # Un-permute back
        inv_perm = np.empty(n, dtype=int)
        inv_perm[perm] = np.arange(n)
        return ranks[inv_perm]

    def _rank_asc_with_random_ties(self, values):
        """Rank values in ascending order with random ties (for minimization)"""
        n = len(values)
        # Shuffle to break ties randomly
        perm = np.random.permutation(n)
        shuffled_vals = values[perm]
        order = np.argsort(shuffled_vals, kind='mergesort')
        ranks = np.empty(n, dtype=int)
        ranks[order] = np.arange(1, n + 1)
        # Un-permute back
        inv_perm = np.empty(n, dtype=int)
        inv_perm[perm] = np.arange(n)
        return ranks[inv_perm]

    def _gen_X(self, dist_mean, dist_rank, nl, n_of_points, q, k, xi):
        """Generate new solutions based on Gaussian mixture distributions (SOCHA-ACOR)"""
        num_dists, num_dims = dist_mean.shape
        X = np.empty((n_of_points, num_dims), dtype=float)

        # Choose distributions according to N(mean=1, sd=q*k)
        probs = self._normal_pdf(np.arange(1, num_dists + 1), mean=1.0, sd=q * k)
        probs = probs / probs.sum()
        idx = np.random.choice(num_dists, size=n_of_points, replace=True, p=probs)

        for l in range(n_of_points):
            j = idx[l]
            o_dist_mean = dist_mean - dist_mean[j]  # translate origin
            r_dist_mean = o_dist_mean.copy()
            available = nl[j]
            vec = None
            R = np.eye(num_dims)
            
            for m in range(num_dims - 1):
                if available.size == 0:
                    return None
                # Distances in rotated space
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
                # Augment direction matrix
                new_vec = o_dist_mean[choice]
                vec = new_vec[None, :] if vec is None else np.vstack([vec, new_vec])
                # QR to get rotation matrix
                Q, _ = np.linalg.qr(vec.T, mode='complete')
                R = Q
                if np.linalg.det(R) < 0:
                    R[:, 0] *= -1
                r_dist_mean = o_dist_mean @ R
                available = available[available != choice]

            # Standard deviations along rotated axes
            dist_sd = np.array([
                np.sum(np.abs(r_dist_mean[nl[j], i] - r_dist_mean[j, i])) / (k - 1)
                for i in range(num_dims)
            ])
            n_x = np.random.normal(loc=r_dist_mean[j], scale=dist_sd * xi, size=(num_dims,))
            n_x = (R @ n_x) + dist_mean[j]
            X[l] = n_x
        return X

    def _normal_pdf(self, x, mean, sd):
        """Normal probability density function"""
        if sd <= 0:
            out = np.zeros_like(x, dtype=float)
            out[np.isclose(x, mean)] = 1.0
            return out
        z = (x - mean) / sd
        return np.exp(-0.5 * z * z) / (sd * np.sqrt(2.0 * np.pi))

    def _euc_dist(self, d):
        """Euclidean distance"""
        return float(np.sqrt(np.sum(np.square(d))))

# 4. Objective function for ACOR (binary cross-entropy loss)
# --------------------------------------------------
def objective(weights):
    model.set_weights(weights)
    y_pred = model.forward(X_train)
    eps = 1e-8
    loss = -np.mean(y_train*np.log(y_pred+eps) + (1-y_train)*np.log(1-y_pred+eps))
    return loss

# 5. Model and ACOR parameters
# --------------------------------------------------
input_dim = X_train.shape[1]
hidden_dim = 6  # single hidden layer size
output_dim = 1
model = FNN(input_dim, hidden_dim, output_dim)
num_weights = FNN.get_num_weights(input_dim, hidden_dim, output_dim)
acor = SOCHA_ACOR(obj_func=objective, dim=num_weights, n_ants=30, n_samples=80, q=0.1, xi=0.85, max_iter=100, patience=15)
lb = -3
ub = 3

# 6. Run ACOR to optimize FNN weights
# --------------------------------------------------
best_weights, best_loss, n_iterations = acor.optimize(lb, ub)

# 7. Evaluate on test set and save results
# --------------------------------------------------
model.set_weights(best_weights)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

# Class distribution
unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
unique_true, counts_true = np.unique(y_test, return_counts=True)

# Warn if only one class is predicted
if len(unique_pred) == 1:
    warnings.warn(f"Model predicted only one class: {unique_pred[0]}. Metrics may be misleading.")

# Print metrics
print(f"Test Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-score: {f1:.4f}")
print("Confusion Matrix:")
print(cm)

# Plot and save metrics bar chart
metrics = [acc, prec, rec, f1]
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-score']
plt.figure(figsize=(6, 4))
bars = plt.bar(metric_names, metrics, color=['skyblue', 'orange', 'green', 'red'])
plt.ylim(0, 1)
plt.title('Heart Model Performance Metrics')
plt.ylabel('Score')
for bar, value in zip(bars, metrics):
    plt.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f'{value:.2f}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'heart_metrics_bar_chart.png'))
plt.show()

# Plot and save confusion matrix
def plot_confusion_matrix(cm, class_names, save_path=None):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names, yticklabels=class_names,
        ylabel='True label', xlabel='Predicted label',
        title='Confusion Matrix'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

plot_confusion_matrix(cm, class_names=["No Heart Disease", "Heart Disease"], save_path=os.path.join(os.path.dirname(__file__), "heart_confusion_matrix.png"))

# Save results to a txt file
output_dir = os.path.dirname(__file__)
with open(os.path.join(output_dir, 'heart_result.txt'), 'w') as f:
    f.write(f"Test Accuracy: {acc:.4f}\n")
    f.write(f"Precision: {prec:.4f}\n")
    f.write(f"Recall: {rec:.4f}\n")
    f.write(f"F1-score: {f1:.4f}\n")
    f.write(f"Number of Iterations until Convergence: {n_iterations}\n")
    f.write("Confusion Matrix (for test set):\n")
    f.write(f"  True Negatives (no heart disease predicted correctly): {cm[0,0]}\n")
    f.write(f"  False Positives (no heart disease predicted as heart disease): {cm[0,1]}\n")
    f.write(f"  False Negatives (heart disease predicted as no heart disease): {cm[1,0]}\n")
    f.write(f"  True Positives (heart disease predicted correctly): {cm[1,1]}\n")
    f.write(f"Predicted class distribution: {y_pred.tolist().count(0)} predicted no heart disease, {y_pred.tolist().count(1)} predicted heart disease\n")
    f.write(f"True class distribution: {y_test.tolist().count(0)} actually no heart disease, {y_test.tolist().count(1)} actually heart disease\n")
    if len(unique_pred) == 1:
        f.write(f"WARNING: Model predicted only one class: {unique_pred[0]}. Metrics may be misleading.\n")

# Save the trained model (weights and scaler)
model_data = {
    'weights': best_weights,
    'scaler': scaler,
    'input_dim': input_dim,
    'hidden_dim': hidden_dim,
    'output_dim': output_dim
}

with open(os.path.join(output_dir, 'heart_model.pkl'), 'wb') as f:
    pickle.dump(model_data, f)

print("Model saved to heart_model.pkl") 
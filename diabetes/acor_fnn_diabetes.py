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
# Load the preprocessed data (space-separated, 8 features + 2 one-hot encoded target)
data = pd.read_csv(
    os.path.join(os.path.dirname(__file__), 'diabetes1.dat'),
    sep=' ',
    header=None
)

# The last TWO columns (9 and 10) are the one-hot encoded target [class_0, class_1]
X = data.iloc[:, :-2].values  # First 8 columns are features
y_onehot = data.iloc[:, -2:].values   # Last 2 columns are one-hot encoded target

# Convert one-hot encoding back to single label (0 or 1)
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
        # Clip values to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0, z1)  # ReLU activation for hidden layer
        z2 = a1 @ self.W2 + self.b2
        a2 = self._stable_sigmoid(z2)  # Sigmoid for binary classification
        return a2.squeeze()

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)

    @staticmethod
    def get_num_weights(input_dim=8, hidden_dim=6, output_dim=1):
        return input_dim*hidden_dim + hidden_dim + hidden_dim*output_dim + output_dim

# 3. SOCHA-ACOR implementation
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

            print(f"Iteration {iteration+1}/{self.max_iter}, Best Loss: {max_y:.4f}")
            
            if iteration - best_iter > self.patience:
                print(f"Early stopping at iteration {iteration+1}")
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
def objective(weights):
    if weights.ndim == 1:
        model.set_weights(weights)
        y_pred = model.forward(X_train)
        eps = 1e-8
        loss = -np.mean(y_train*np.log(y_pred+eps) + (1-y_train)*np.log(1-y_pred+eps))
        return loss
    else:
        losses = []
        for w in weights:
            model.set_weights(w)
            y_pred = model.forward(X_train)
            eps = 1e-8
            loss = -np.mean(y_train*np.log(y_pred+eps) + (1-y_train)*np.log(1-y_pred+eps))
            losses.append(loss)
        return np.array(losses)

# 5. Model and ACOR parameters
# --------------------------------------------------
input_dim = 8  # 8 features from diabetes1.dat
hidden_dim = 6
output_dim = 1
model = FNN(input_dim, hidden_dim, output_dim)
num_weights = FNN.get_num_weights(input_dim, hidden_dim, output_dim)
print(f"\nModel architecture: {input_dim}-{hidden_dim}-{output_dim}")
print(f"Total weights: {num_weights}")

acor = SOCHA_ACOR(obj_func=objective, dim=num_weights, n_ants=30, n_samples=80, q=0.1, xi=0.85, max_iter=100, patience=15)
lb = -3
ub = 3

# 6. Run ACOR to optimize FNN weights
# --------------------------------------------------
print("\nStarting ACOR optimization...")
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

if len(unique_pred) == 1:
    warnings.warn(f"Model predicted only one class: {unique_pred[0]}. Metrics may be misleading.")

print(f"\nTest Accuracy: {acc:.4f}")
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
plt.title('Diabetes Model Performance Metrics (8 features)')
plt.ylabel('Score')
for bar, value in zip(bars, metrics):
    plt.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f'{value:.2f}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'diabetes_metrics_bar_chart.png'))
plt.show()

# Plot confusion matrix
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

output_dir = os.path.dirname(__file__)

plot_confusion_matrix(cm, class_names=["No Diabetes", "Diabetes"], 
                     save_path=os.path.join(output_dir, "diabetes_confusion_matrix.png"))

# Save results
with open(os.path.join(output_dir, 'diabetes_result.txt'), 'w') as f:
    f.write(f"Diabetes Classification - 8 Features\n")
    f.write(f"Architecture: {input_dim}-{hidden_dim}-{output_dim}\n")
    f.write(f"Total weights: {num_weights}\n\n")
    f.write(f"Test Accuracy: {acc:.4f}\n")
    f.write(f"Precision: {prec:.4f}\n")
    f.write(f"Recall: {rec:.4f}\n")
    f.write(f"F1-score: {f1:.4f}\n")
    f.write(f"Number of Iterations until Convergence: {n_iterations}\n")
    f.write("Confusion Matrix (for test set):\n")
    f.write(f"  True Negatives: {cm[0,0]}\n")
    f.write(f"  False Positives: {cm[0,1]}\n")
    f.write(f"  False Negatives: {cm[1,0]}\n")
    f.write(f"  True Positives: {cm[1,1]}\n")
    f.write(f"Predicted distribution: {dict(zip(*np.unique(y_pred, return_counts=True)))}\n")
    f.write(f"True distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}\n")
    if len(unique_pred) == 1:
        f.write(f"WARNING: Model predicted only one class: {unique_pred[0]}\n")

# Save model
model_data = {
    'weights': best_weights,
    'scaler': scaler,
    'input_dim': input_dim,
    'hidden_dim': hidden_dim,
    'output_dim': output_dim,
    'num_features': 8
}

with open(os.path.join(output_dir, 'diabetes_model.pkl'), 'wb') as f:
    pickle.dump(model_data, f)

print("\nModel saved to diabetes_model.pkl")
print(f"Results saved to diabetes_result.txt")
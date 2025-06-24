import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings

# Set random seed for reproducibility
np.random.seed(42)

# 1. Load and preprocess the data
# --------------------------------------------------
data = pd.read_csv('heart.csv')

# Target variable is 'target' (already 0/1)
X = data.drop('target', axis=1).values
y = data['target'].values

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Define a simple FNN (smaller hidden layer)
# --------------------------------------------------
class FNN:
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

# 3. Improved ACOR implementation
# --------------------------------------------------
class ACOR:
    def __init__(self, obj_func, dim, n_ants=40, n_samples=80, q=0.1, xi=0.85, max_iter=60, patience=10):
        self.obj_func = obj_func
        self.dim = dim
        self.n_ants = n_ants
        self.n_samples = n_samples
        self.q = q
        self.xi = xi
        self.max_iter = max_iter
        self.patience = patience

    def optimize(self, lb, ub):
        solutions = np.random.uniform(lb, ub, (self.n_samples, self.dim))
        fitness = np.array([self.obj_func(sol) for sol in solutions])
        idx = np.argsort(fitness)
        solutions = solutions[idx]
        fitness = fitness[idx]
        best_sol = solutions[0].copy()
        best_fit = fitness[0]
        best_iter = 0
        for it in range(self.max_iter):
            w = 1/(self.q*self.n_samples*np.sqrt(2*np.pi)) * np.exp(-0.5*(np.arange(self.n_samples)/(self.q*self.n_samples))**2)
            w /= w.sum()
            s = np.zeros((self.n_samples, self.dim))
            for i in range(self.n_samples):
                s[i] = self.xi * np.std(solutions, axis=0) + 1e-10
            new_solutions = np.zeros((self.n_ants, self.dim))
            for k in range(self.n_ants):
                idx = np.random.choice(self.n_samples, p=w)
                new_solutions[k] = np.random.normal(solutions[idx], s[idx])
                new_solutions[k] = np.clip(new_solutions[k], lb, ub)
            new_fitness = np.array([self.obj_func(sol) for sol in new_solutions])
            all_solutions = np.vstack([solutions, new_solutions])
            all_fitness = np.hstack([fitness, new_fitness])
            idx = np.argsort(all_fitness)
            solutions = all_solutions[idx][:self.n_samples]
            fitness = all_fitness[idx][:self.n_samples]
            if fitness[0] < best_fit:
                best_fit = fitness[0]
                best_sol = solutions[0].copy()
                best_iter = it
            print(f"Iteration {it+1}/{self.max_iter}, Best Loss: {best_fit:.4f}")
            # Early stopping
            if it - best_iter > self.patience:
                print(f"Early stopping at iteration {it+1}")
                break
        return best_sol, best_fit

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
hidden_dim = 6  # smaller hidden layer for easier optimization
output_dim = 1
model = FNN(input_dim, hidden_dim, output_dim)
num_weights = FNN.get_num_weights(input_dim, hidden_dim, output_dim)
acor = ACOR(obj_func=objective, dim=num_weights, n_ants=40, n_samples=80, q=0.1, xi=0.85, max_iter=60, patience=12)
lb = -3
ub = 3

# 6. Run ACOR to optimize FNN weights
# --------------------------------------------------
best_weights, best_loss = acor.optimize(lb, ub)

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
print(f"  True Negatives (no heart disease predicted correctly): {cm[0,0]}")
print(f"  False Positives (no heart disease predicted as heart disease): {cm[0,1]}")
print(f"  False Negatives (heart disease predicted as no heart disease): {cm[1,0]}")
print(f"  True Positives (heart disease predicted correctly): {cm[1,1]}")
print(f"Predicted class distribution: {y_pred.tolist().count(0)} predicted no heart disease, {y_pred.tolist().count(1)} predicted heart disease")
print(f"True class distribution: {y_test.tolist().count(0)} actually no heart disease, {y_test.tolist().count(1)} actually heart disease")

# Save results to a txt file
with open('heart_result.txt', 'w') as f:
    f.write(f"Test Accuracy: {acc:.4f}\n")
    f.write(f"Precision: {prec:.4f}\n")
    f.write(f"Recall: {rec:.4f}\n")
    f.write(f"F1-score: {f1:.4f}\n")
    f.write("Confusion Matrix (for test set):\n")
    f.write(f"  True Negatives (no heart disease predicted correctly): {cm[0,0]}\n")
    f.write(f"  False Positives (no heart disease predicted as heart disease): {cm[0,1]}\n")
    f.write(f"  False Negatives (heart disease predicted as no heart disease): {cm[1,0]}\n")
    f.write(f"  True Positives (heart disease predicted correctly): {cm[1,1]}\n")
    f.write(f"Predicted class distribution: {y_pred.tolist().count(0)} predicted no heart disease, {y_pred.tolist().count(1)} predicted heart disease\n")
    f.write(f"True class distribution: {y_test.tolist().count(0)} actually no heart disease, {y_test.tolist().count(1)} actually heart disease\n")
    if len(unique_pred) == 1:
        f.write(f"WARNING: Model predicted only one class: {unique_pred[0]}. Metrics may be misleading.\n") 
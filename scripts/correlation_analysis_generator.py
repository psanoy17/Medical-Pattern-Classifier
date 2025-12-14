"""
Correlation Analysis Generator for Phase 2C Paper

Generates:
1. Feature-to-feature Pearson correlation matrices for all three datasets
2. Feature-to-model-prediction correlations (hybrid ACOR-LM)
3. Feature-to-model-prediction correlations (GNB baseline)
4. Outputs as CSV + visualizations for paper inclusion

Paper sections supported:
- Feature importance via correlation analysis
- Model interpretability section
- Dataset characteristics comparison
"""

import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lm_local_search import MultipleColonyACOR, LevenbergMarquardt

# =============================================================================
# CONFIGURATION: Validated optimal configs from Phase 2C
# =============================================================================
OPTIMAL_CONFIGS = {
    'heart': {
        'n_colonies': 4,
        'local_patience': 10,
        'sharing_frequency': 15,
        'sharing_ratio': 0.15,
        'initial_mu': 0.1,
        'lm_max_iterations': 10,
        'accuracy': 0.8128,
    },
    'diabetes': {
        'n_colonies': 4,
        'local_patience': 7,
        'sharing_frequency': 10,
        'sharing_ratio': 0.20,
        'initial_mu': 0.001,
        'lm_max_iterations': 10,
        'accuracy': 0.7503,
    },
    'cancer': {
        'n_colonies': 4,
        'local_patience': 7,
        'sharing_frequency': 10,
        'sharing_ratio': 0.20,
        'initial_mu': 0.001,
        'lm_max_iterations': 10,
        'accuracy': 0.9522,
    },
}

# Dataset file paths
DATASET_PATHS = {
    'heart': {
        'path': os.path.join(os.path.dirname(__file__), '..', 'heart', 'heart1.dat'),
        'n_features': 35,
        'target_encoding': 'onehot_2cols',  # Last 2 columns are one-hot
        'feature_names': [
            'Age', 'Age_Padding',
            'Sex', 'Sex_Padding',
            'Chest_Pain_0', 'Chest_Pain_1', 'Chest_Pain_2', 'Chest_Pain_3', 'Chest_Pain_Padding',
            'Resting_BP', 'Resting_BP_Padding',
            'Serum_Chol', 'Serum_Chol_Padding',
            'Fasting_BS', 'Fasting_BS_Padding',
            'Resting_ECG_0', 'Resting_ECG_1', 'Resting_ECG_2', 'Resting_ECG_Padding',
            'Max_Heart_Rate', 'Max_Heart_Rate_Padding',
            'Exercise_Angina', 'Exercise_Angina_Padding',
            'ST_Depression', 'ST_Depression_Padding',
            'ST_Slope_0', 'ST_Slope_1', 'ST_Slope_2', 'ST_Slope_Padding',
            'Num_Vessels', 'Num_Vessels_Padding',
            'Thalassemia_0', 'Thalassemia_1', 'Thalassemia_2', 'Thalassemia_Padding'
        ]
    },
    'diabetes': {
        'path': os.path.join(os.path.dirname(__file__), '..', 'diabetes', 'diabetes1.dat'),
        'n_features': 8,
        'target_encoding': 'onehot_2cols',
        'feature_names': [
            'Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness',
            'Insulin', 'BMI', 'Diabetes Pedigree Function', 'Age'
        ]
    },
    'cancer': {
        'path': os.path.join(os.path.dirname(__file__), '..', 'cancer', 'cancer1.dat'),
        'n_features': 9,
        'target_encoding': 'onehot_2cols',
        'feature_names': [
            'Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness',
            'Compactness', 'Concavity', 'Concave Points', 'Symmetry'
        ]
    },
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def load_dataset(dataset_name):
    """Load and preprocess dataset."""
    config = DATASET_PATHS[dataset_name]
    data = pd.read_csv(config['path'], sep=' ', header=None)
    
    X = data.iloc[:, :config['n_features']].values
    
    if config['target_encoding'] == 'onehot_2cols':
        y_onehot = data.iloc[:, -2:].values
        y = np.argmax(y_onehot, axis=1)
    else:
        y = data.iloc[:, -1].values
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    print(f"  Loaded {dataset_name}: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Class distribution: {np.bincount(y)}")
    
    return X, y, scaler


def compute_feature_correlation_matrix(X, feature_names=None):
    """Compute feature-to-feature Pearson correlation matrix."""
    n_features = X.shape[1]
    if feature_names is None:
        feature_names = [f'Feature_{i+1}' for i in range(n_features)]
    
    corr_matrix = np.corrcoef(X.T)
    corr_df = pd.DataFrame(corr_matrix, index=feature_names, columns=feature_names)
    
    return corr_df


def train_acor_lm_model(X_train, y_train, dataset_name, seed=42):
    """Train hybrid ACOR-LM model with optimal config."""
    np.random.seed(seed)
    config = OPTIMAL_CONFIGS[dataset_name]
    
    # Define FNN based on dataset input dimension
    input_dim = X_train.shape[1]
    hidden_dim = 6
    output_dim = 1
    
    class FNN:
        def __init__(self, input_dim, hidden_dim, output_dim):
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
        
        def set_weights(self, weights):
            idx = 0
            self.W1 = weights[idx:idx+self.input_dim*self.hidden_dim].reshape(
                self.input_dim, self.hidden_dim)
            idx += self.input_dim*self.hidden_dim
            self.b1 = weights[idx:idx+self.hidden_dim]
            idx += self.hidden_dim
            self.W2 = weights[idx:idx+self.hidden_dim*self.output_dim].reshape(
                self.hidden_dim, self.output_dim)
            idx += self.hidden_dim*self.output_dim
            self.b2 = weights[idx:idx+self.output_dim]
        
        def forward(self, X):
            z1 = X @ self.W1 + self.b1
            a1 = np.tanh(z1)
            z2 = a1 @ self.W2 + self.b2
            a2 = 1 / (1 + np.exp(-z2))
            return a2.squeeze()
        
        def predict(self, X):
            return (self.forward(X) > 0.5).astype(int)
        
        @staticmethod
        def get_num_weights(input_dim, hidden_dim, output_dim):
            return (input_dim*hidden_dim + hidden_dim + 
                   hidden_dim*output_dim + output_dim)
    
    model = FNN(input_dim, hidden_dim, output_dim)
    n_weights = FNN.get_num_weights(input_dim, hidden_dim, output_dim)
    
    def objective_function(weights):
        model.set_weights(weights)
        y_pred = model.forward(X_train)
        eps = 1e-8
        loss = -np.mean(y_train * np.log(y_pred + eps) + 
                       (1 - y_train) * np.log(1 - y_pred + eps))
        return loss
    
    # Initialize ACOR-LM with proper API
    n_ants = 2
    n_samples = int(X_train.shape[0] * 0.25)  # 25% of training set
    max_iter = 100
    patience = 15
    
    acor = MultipleColonyACOR(
        obj_func=objective_function,
        dim=n_weights,
        n_colonies=config['n_colonies'],
        n_ants=n_ants,
        n_samples=n_samples,
        max_iter=max_iter,
        patience=patience,
        local_patience=config['local_patience'],
        sharing_frequency=config['sharing_frequency'],
        sharing_ratio=config['sharing_ratio'],
        initial_mu=config['initial_mu'],
        lm_max_iterations=config['lm_max_iterations'],
        seed=seed
    )
    
    # Run optimization with proper bounds and model
    # Weights typically initialized in [-1, 1] range for neural networks
    best_weights, best_loss, iterations, _ = acor.optimize(
        lb=-1.0,
        ub=1.0,
        model=model,
        X_train=X_train,
        y_train=y_train
    )
    
    model.set_weights(best_weights)
    return model, best_weights


def train_gnb_model(X_train, y_train):
    """Train Gaussian Naive Bayes baseline."""
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    return gnb


def compute_feature_model_correlations(X, y_pred, feature_names=None):
    """
    Compute correlation between each feature and model predictions.
    Also includes correlation with actual targets.
    """
    n_features = X.shape[1]
    if feature_names is None:
        feature_names = [f'Feature_{i+1}' for i in range(n_features)]
    
    correlations = {}
    
    # Feature-to-prediction correlation
    for i in range(n_features):
        corr_pred = np.corrcoef(X[:, i], y_pred)[0, 1]
        correlations[f'{feature_names[i]}_to_pred'] = corr_pred
    
    return correlations


def generate_correlation_heatmap(corr_df, output_path, title):
    """Generate and save correlation heatmap."""
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_df, cmap='coolwarm', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved heatmap: {output_path}")


# =============================================================================
# MAIN ANALYSIS
# =============================================================================
def main():
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS GENERATOR - Phase 2C")
    print("="*80)
    
    output_dir = os.path.dirname(__file__)
    results = {}
    
    for dataset_name in ['heart', 'diabetes', 'cancer']:
        print(f"\n{'='*80}")
        print(f"DATASET: {dataset_name.upper()}")
        print(f"{'='*80}")
        
        # Load data
        print(f"\n1. Loading dataset...")
        X, y, scaler = load_dataset(dataset_name)
        
        # Train-test split for model training
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # === Feature-to-Feature Correlations ===
        print(f"\n2. Computing feature-to-feature correlations...")
        config = DATASET_PATHS[dataset_name]
        feature_corr_df = compute_feature_correlation_matrix(X, config['feature_names'])
        
        # Save to CSV
        feature_corr_path = os.path.join(
            output_dir, f'correlation_{dataset_name}_feature_feature.csv')
        feature_corr_df.to_csv(feature_corr_path)
        print(f"  Saved feature-feature correlations: {feature_corr_path}")
        
        # Generate heatmap
        heatmap_path = os.path.join(
            output_dir, f'correlation_{dataset_name}_feature_feature_heatmap.png')
        generate_correlation_heatmap(
            feature_corr_df, heatmap_path,
            f'{dataset_name.upper()}: Feature-to-Feature Correlation Matrix')
        
        # === ACOR-LM Feature-Model Correlations ===
        print(f"\n3. Training hybrid ACOR-LM model...")
        model, best_weights = train_acor_lm_model(X_train, y_train, dataset_name)
        
        # Get predictions
        y_pred_prob_acor = model.forward(X_test)
        y_pred_acor = model.predict(X_test)
        
        # Compute accuracy
        acor_lm_acc = accuracy_score(y_test, y_pred_acor)
        print(f"  ACOR-LM Test Accuracy: {acor_lm_acc:.4f}")
        
        # Compute feature-model correlations
        acor_lm_corr = compute_feature_model_correlations(
            X_test, y_pred_prob_acor, config['feature_names'])
        
        acor_lm_corr_df = pd.DataFrame([acor_lm_corr]).T
        acor_lm_corr_df.columns = ['Correlation']
        
        acor_lm_path = os.path.join(
            output_dir, f'correlation_{dataset_name}_acor_lm_feature_model.csv')
        acor_lm_corr_df.to_csv(acor_lm_path)
        print(f"  Saved ACOR-LM feature-model correlations: {acor_lm_path}")
        
        # === GNB Baseline Feature-Model Correlations ===
        print(f"\n4. Training Gaussian Naive Bayes baseline...")
        gnb = train_gnb_model(X_train, y_train)
        
        # Get predictions
        y_pred_prob_gnb = gnb.predict_proba(X_test)[:, 1]
        y_pred_gnb = gnb.predict(X_test)
        
        # Compute accuracy
        gnb_acc = accuracy_score(y_test, y_pred_gnb)
        print(f"  GNB Test Accuracy: {gnb_acc:.4f}")
        
        # Compute feature-model correlations
        gnb_corr = compute_feature_model_correlations(
            X_test, y_pred_prob_gnb, config['feature_names'])
        
        gnb_corr_df = pd.DataFrame([gnb_corr]).T
        gnb_corr_df.columns = ['Correlation']
        
        gnb_path = os.path.join(
            output_dir, f'correlation_{dataset_name}_gnb_feature_model.csv')
        gnb_corr_df.to_csv(gnb_path)
        print(f"  Saved GNB feature-model correlations: {gnb_path}")
        
        # === Comparison Summary ===
        print(f"\n5. Summary statistics for {dataset_name}:")
        print(f"  Feature-Feature Correlation Range: [{feature_corr_df.values[~np.eye(feature_corr_df.shape[0], dtype=bool)].min():.3f}, {feature_corr_df.values[~np.eye(feature_corr_df.shape[0], dtype=bool)].max():.3f}]")
        print(f"  ACOR-LM Feature-Model Correlation Range: [{acor_lm_corr_df['Correlation'].min():.3f}, {acor_lm_corr_df['Correlation'].max():.3f}]")
        print(f"  GNB Feature-Model Correlation Range: [{gnb_corr_df['Correlation'].min():.3f}, {gnb_corr_df['Correlation'].max():.3f}]")
        print(f"  ACOR-LM vs GNB Accuracy Delta: {acor_lm_acc - gnb_acc:+.4f}")
        
        # Store results
        results[dataset_name] = {
            'feature_corr': feature_corr_df,
            'acor_lm_corr': acor_lm_corr_df,
            'gnb_corr': gnb_corr_df,
            'acor_lm_acc': acor_lm_acc,
            'gnb_acc': gnb_acc,
        }
    
    # === Cross-Dataset Summary ===
    print(f"\n{'='*80}")
    print("CROSS-DATASET COMPARISON")
    print(f"{'='*80}\n")
    
    comparison_data = []
    for dataset_name in ['heart', 'diabetes', 'cancer']:
        r = results[dataset_name]
        comparison_data.append({
            'Dataset': dataset_name.capitalize(),
            'ACOR-LM Acc': f"{r['acor_lm_acc']:.4f}",
            'GNB Acc': f"{r['gnb_acc']:.4f}",
            'Delta': f"{r['acor_lm_acc'] - r['gnb_acc']:+.4f}",
            'Feature-Model Corr Range (ACOR-LM)': 
                f"[{r['acor_lm_corr']['Correlation'].min():.3f}, {r['acor_lm_corr']['Correlation'].max():.3f}]",
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_path = os.path.join(output_dir, 'correlation_comparison_summary.csv')
    comparison_df.to_csv(comparison_path, index=False)
    print(comparison_df.to_string(index=False))
    print(f"\nSaved comparison summary: {comparison_path}")
    
    print(f"\n{'='*80}")
    print("CORRELATION ANALYSIS COMPLETE")
    print(f"{'='*80}\n")
    print("Generated files for paper inclusion:")
    print("  1. correlation_DATASET_feature_feature.csv - Feature correlation matrices")
    print("  2. correlation_DATASET_feature_feature_heatmap.png - Correlation heatmaps")
    print("  3. correlation_DATASET_acor_lm_feature_model.csv - ACOR-LM feature-model correlations")
    print("  4. correlation_DATASET_gnb_feature_model.csv - GNB baseline correlations")
    print("  5. correlation_comparison_summary.csv - Cross-dataset summary")


if __name__ == '__main__':
    main()

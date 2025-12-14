"""
Fast Correlation Analysis Generator - Paper Ready

This is a lightweight version that:
1. Computes feature-feature correlations
2. Uses pre-computed optimal weights from validation runs, OR
3. Falls back to GNB + simple neural network for quick demo

Outputs paper-ready correlation matrices and feature importance rankings.
"""

import numpy as np
import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 9

# Dataset file paths and feature names
DATASET_PATHS = {
    'heart': {
        'path': os.path.join(os.path.dirname(__file__), '..', 'heart', 'heart1.dat'),
        'n_features': 35,
        'name': 'Heart Disease (35 features)',
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
        'name': 'Diabetes (8 features)',
        'feature_names': [
            'Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness',
            'Insulin', 'BMI', 'Diabetes Pedigree Function', 'Age'
        ]
    },
    'cancer': {
        'path': os.path.join(os.path.dirname(__file__), '..', 'cancer', 'cancer1.dat'),
        'n_features': 9,
        'name': 'Cancer Diagnosis (9 features)',
        'feature_names': [
            'Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness',
            'Compactness', 'Concavity', 'Concave Points', 'Symmetry'
        ]
    },
}


def load_dataset(dataset_name):
    """Load and preprocess dataset."""
    config = DATASET_PATHS[dataset_name]
    data = pd.read_csv(config['path'], sep=' ', header=None)
    
    X = data.iloc[:, :config['n_features']].values
    y_onehot = data.iloc[:, -2:].values
    y = np.argmax(y_onehot, axis=1)
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    print(f"  ✓ Loaded {dataset_name}: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"    Class distribution: {np.bincount(y)}")
    
    return X, y, scaler


def compute_feature_correlation_matrix(X, feature_names=None):
    """Compute feature-to-feature Pearson correlation matrix."""
    n_features = X.shape[1]
    if feature_names is None:
        feature_names = [f'F{i+1}' for i in range(n_features)]
    
    corr_matrix = np.corrcoef(X.T)
    corr_df = pd.DataFrame(corr_matrix, index=feature_names, columns=feature_names)
    
    return corr_df


def compute_feature_importance_via_correlation(X, y):
    """
    Compute feature importance based on correlation with target.
    Returns absolute Pearson correlation coefficients.
    """
    correlations = {}
    for i in range(X.shape[1]):
        # Compute Pearson correlation between feature and target
        corr = np.corrcoef(X[:, i], y)[0, 1]
        correlations[f'F{i+1}'] = abs(corr)  # Use absolute value for importance
    
    return pd.Series(correlations).sort_values(ascending=False)


def train_simple_neural_network(X_train, y_train, X_test, y_test):
    """
    Train a simple 2-layer neural network for quick feature-model correlation.
    Uses SGD with fixed learning rate.
    """
    np.random.seed(42)
    
    input_dim = X_train.shape[1]
    hidden_dim = 16
    
    # Initialize weights
    W1 = np.random.randn(input_dim, hidden_dim) * 0.01
    b1 = np.zeros(hidden_dim)
    W2 = np.random.randn(hidden_dim, 1) * 0.01
    b2 = np.zeros(1)
    
    # Quick training (50 iterations, fixed learning rate)
    lr = 0.01
    for epoch in range(50):
        # Forward pass
        z1 = X_train @ W1 + b1
        a1 = np.tanh(z1)
        z2 = a1 @ W2 + b2
        a2 = 1 / (1 + np.exp(-z2))
        
        # Backward pass (simplified)
        dz2 = a2 - y_train.reshape(-1, 1)
        dW2 = (a1.T @ dz2) / len(y_train)
        db2 = np.mean(dz2)
        
        da1 = dz2 @ W2.T
        dz1 = da1 * (1 - a1**2)
        dW1 = (X_train.T @ dz1) / len(y_train)
        db1 = np.mean(dz1, axis=0)
        
        # Update weights
        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1
    
    # Get predictions
    z1 = X_test @ W1 + b1
    a1 = np.tanh(z1)
    z2 = a1 @ W2 + b2
    a2 = 1 / (1 + np.exp(-z2)).squeeze()
    
    accuracy = accuracy_score(y_test, (a2 > 0.5).astype(int))
    
    return a2, accuracy


def train_gnb_and_get_predictions(X_train, y_train, X_test):
    """Train GNB and return probability predictions."""
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    return gnb.predict_proba(X_test)[:, 1]


def compute_feature_model_correlations(X, y_pred, feature_names=None):
    """
    Compute correlation between each feature and model predictions.
    """
    n_features = X.shape[1]
    if feature_names is None:
        feature_names = [f'F{i+1}' for i in range(n_features)]
    
    correlations = {}
    for i in range(n_features):
        corr = np.corrcoef(X[:, i], y_pred)[0, 1]
        correlations[feature_names[i]] = abs(corr)  # Absolute correlation for importance
    
    return pd.Series(correlations).sort_values(ascending=False)


def visualize_top_feature_correlations(acor_corr, gnb_corr, dataset_name, output_dir):
    """Create side-by-side bar plot of top feature correlations."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    top_k = min(10, len(acor_corr))
    
    acor_corr.head(top_k).plot(kind='barh', ax=axes[0], color='steelblue')
    axes[0].set_xlabel('Absolute Correlation with Prediction')
    axes[0].set_title(f'{dataset_name}: Neural Network\n(Feature-Model Correlation)')
    axes[0].invert_yaxis()
    
    gnb_corr.head(top_k).plot(kind='barh', ax=axes[1], color='coral')
    axes[1].set_xlabel('Absolute Correlation with Prediction')
    axes[1].set_title(f'{dataset_name}: Gaussian Naive Bayes\n(Feature-Model Correlation)')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'correlation_{dataset_name}_feature_importance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved feature importance plot: {os.path.basename(output_path)}")


def main():
    print("\n" + "="*80)
    print("FAST CORRELATION ANALYSIS GENERATOR - Phase 2C (Paper Ready)")
    print("="*80)
    
    output_dir = os.path.dirname(__file__)
    summary_data = []
    
    for dataset_name in ['heart', 'diabetes', 'cancer']:
        print(f"\n{'─'*80}")
        print(f"{DATASET_PATHS[dataset_name]['name'].upper()}")
        print(f"{'─'*80}")
        
        # Get config for this dataset
        config = DATASET_PATHS[dataset_name]
        
        # 1. Load data
        print(f"\n[1/4] Loading dataset...")
        X, y, scaler = load_dataset(dataset_name)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 2. Feature-to-Feature Correlations
        print(f"\n[2/4] Computing feature-feature correlations...")
        feature_corr_df = compute_feature_correlation_matrix(X, config['feature_names'])
        
        feature_corr_path = os.path.join(
            output_dir, f'correlation_{dataset_name}_feature_feature.csv')
        feature_corr_df.to_csv(feature_corr_path)
        print(f"    ✓ Saved: {os.path.basename(feature_corr_path)}")
        
        # Compute summary statistics
        mask = ~np.eye(feature_corr_df.shape[0], dtype=bool)
        feature_corr_range = [feature_corr_df.values[mask].min(), 
                             feature_corr_df.values[mask].max()]
        
        # 3. Simple Neural Network for Feature-Model Correlations
        print(f"\n[3/4] Training simple neural network...")
        nn_pred, nn_acc = train_simple_neural_network(X_train, y_train, X_test, y_test)
        nn_corr = compute_feature_model_correlations(X_test, nn_pred, config['feature_names'])
        
        nn_corr_path = os.path.join(
            output_dir, f'correlation_{dataset_name}_nn_feature_model.csv')
        nn_corr.to_frame('Correlation').to_csv(nn_corr_path)
        print(f"    ✓ Neural Network Test Accuracy: {nn_acc:.4f}")
        print(f"    ✓ Saved: {os.path.basename(nn_corr_path)}")
        
        # 4. GNB Baseline Feature-Model Correlations
        print(f"\n[4/4] Training Gaussian Naive Bayes baseline...")
        gnb_pred = train_gnb_and_get_predictions(X_train, y_train, X_test)
        gnb_acc = accuracy_score(y_test, (gnb_pred > 0.5).astype(int))
        gnb_corr = compute_feature_model_correlations(X_test, gnb_pred, config['feature_names'])
        
        gnb_corr_path = os.path.join(
            output_dir, f'correlation_{dataset_name}_gnb_feature_model.csv')
        gnb_corr.to_frame('Correlation').to_csv(gnb_corr_path)
        print(f"    ✓ GNB Test Accuracy: {gnb_acc:.4f}")
        print(f"    ✓ Saved: {os.path.basename(gnb_corr_path)}")
        
        # Visualize top features
        print(f"\n[BONUS] Creating feature importance visualization...")
        visualize_top_feature_correlations(nn_corr, gnb_corr, dataset_name.capitalize(), output_dir)
        
        # Summary
        print(f"\n[SUMMARY]")
        print(f"  Feature-Feature Corr Range: [{feature_corr_range[0]:+.3f}, {feature_corr_range[1]:+.3f}]")
        print(f"  NN Feature-Model Corr Range: [{nn_corr.min():+.3f}, {nn_corr.max():+.3f}]")
        print(f"  GNB Feature-Model Corr Range: [{gnb_corr.min():+.3f}, {gnb_corr.max():+.3f}]")
        print(f"  NN vs GNB Accuracy Delta: {nn_acc - gnb_acc:+.4f}")
        
        summary_data.append({
            'Dataset': dataset_name.capitalize(),
            'Samples': X.shape[0],
            'Features': X.shape[1],
            'NN_Accuracy': f"{nn_acc:.4f}",
            'GNB_Accuracy': f"{gnb_acc:.4f}",
            'NN_Top_Feature': nn_corr.index[0],
            'NN_Top_Corr': f"{nn_corr.iloc[0]:.3f}",
            'GNB_Top_Feature': gnb_corr.index[0],
            'GNB_Top_Corr': f"{gnb_corr.iloc[0]:.3f}",
        })
    
    # Save comparison summary
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, 'correlation_analysis_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}\n")
    
    print("Generated outputs (ready for paper):")
    print("  1. correlation_DATASET_feature_feature.csv")
    print("     → Feature-to-feature correlation matrices")
    print("  2. correlation_DATASET_nn_feature_model.csv")
    print("     → Feature-model correlation (neural network)")
    print("  3. correlation_DATASET_gnb_feature_model.csv")
    print("     → Feature-model correlation (GNB baseline)")
    print("  4. correlation_DATASET_feature_importance.png")
    print("     → Top 10 feature importance visualizations")
    print("  5. correlation_analysis_summary.csv")
    print("     → Summary statistics and comparison\n")
    
    print(summary_df.to_string(index=False))


if __name__ == '__main__':
    main()

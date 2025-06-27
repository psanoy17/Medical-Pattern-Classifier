import numpy as np
import pandas as pd
import pickle
import os

# FNN class for single hidden layer
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
        a2 = 1 / (1 + np.exp(-z2))
        return a2.squeeze()

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)

def load_model(model_path='cancer_model.pkl'):
    model_path = os.path.join(os.path.dirname(__file__), model_path)
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    model = FNN(model_data['input_dim'], model_data['hidden_dim'], model_data['output_dim'])
    model.set_weights(model_data['weights'])
    return model, model_data['scaler']

def predict_cancer(data, model=None, scaler=None):
    if model is None or scaler is None:
        model, scaler = load_model()
    if isinstance(data, pd.DataFrame):
        if 'diagnosis' in data.columns:
            data = data.drop('diagnosis', axis=1)
        if 'id' in data.columns:
            data = data.drop('id', axis=1)
        X = data.values
    else:
        X = data
    X_scaled = scaler.transform(X)
    probabilities = model.forward(X_scaled)
    predictions = (probabilities > 0.5).astype(int)
    return predictions, probabilities

if __name__ == "__main__":
    model, scaler = load_model()
    # Example: Realistic sample for cancer dataset (replace values as needed)
    sample = {
        'radius_mean': 20.6,
        'texture_mean': 29.3,
        'perimeter_mean': 140.1,
        'area_mean': 1265.7,
        'smoothness_mean': 0.127,
        'compactness_mean': 0.284,
        'concavity_mean': 0.343,
        'concave points_mean': 0.155,
        'symmetry_mean': 0.246,
        'fractal_dimension_mean': 0.073,
        'radius_se': 0.763,
        'texture_se': 2.125,
        'perimeter_se': 5.312,
        'area_se': 94.44,
        'smoothness_se': 0.011,
        'compactness_se': 0.036,
        'concavity_se': 0.058,
        'concave points_se': 0.018,
        'symmetry_se': 0.044,
        'fractal_dimension_se': 0.007,
        'radius_worst': 26.23,
        'texture_worst': 38.25,
        'perimeter_worst': 177.4,
        'area_worst': 2051.0,
        'smoothness_worst': 0.187,
        'compactness_worst': 0.657,
        'concavity_worst': 0.712,
        'concave points_worst': 0.239,
        'symmetry_worst': 0.458,
        'fractal_dimension_worst': 0.114
    }
    sample_df = pd.DataFrame([sample])
    pred, prob = predict_cancer(sample_df, model, scaler)
    if np.isscalar(pred):
        print(f"Prediction: {'Malignant' if pred == 1 else 'Benign'}")
        print(f"Probability of malignancy: {prob:.4f}")
    else:
        print(f"Prediction: {'Malignant' if pred[0] == 1 else 'Benign'}")
        print(f"Probability of malignancy: {prob[0]:.4f}") 
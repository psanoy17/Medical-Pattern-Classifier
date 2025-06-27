import numpy as np
import pandas as pd
import pickle

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

def load_model(model_path='heart_model.pkl'):
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    model = FNN(model_data['input_dim'], model_data['hidden_dim'], model_data['output_dim'])
    model.set_weights(model_data['weights'])
    return model, model_data['scaler']

def predict_heart(data, model=None, scaler=None):
    if model is None or scaler is None:
        model, scaler = load_model()
    if isinstance(data, pd.DataFrame):
        if 'target' in data.columns:
            data = data.drop('target', axis=1)
        X = data.values
    else:
        X = data
    X_scaled = scaler.transform(X)
    probabilities = model.forward(X_scaled)
    predictions = (probabilities > 0.5).astype(int)
    return predictions, probabilities

if __name__ == "__main__":
    model, scaler = load_model()
    # Example: Realistic sample for heart dataset (replace values as needed)
    sample = {
        'age': 63,
        'sex': 1,
        'cp': 3,
        'trestbps': 145,
        'chol': 233,
        'fbs': 1,
        'restecg': 0,
        'thalach': 150,
        'exang': 0,
        'oldpeak': 2.3,
        'slope': 0,
        'ca': 0,
        'thal': 1,
        # Add all other features as required by your dataset
    }
    sample_df = pd.DataFrame([sample])
    pred, prob = predict_heart(sample_df, model, scaler)
    if np.isscalar(pred):
        print(f"Prediction: {'Heart Disease' if pred == 1 else 'No Heart Disease'}")
        print(f"Probability of heart disease: {prob:.4f}")
    else:
        print(f"Prediction: {'Heart Disease' if pred[0] == 1 else 'No Heart Disease'}")
        print(f"Probability of heart disease: {prob[0]:.4f}") 
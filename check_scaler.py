import joblib
import numpy as np

# Load scaler
scaler = joblib.load('Model/scaler.joblib')

# Print scaler information
print(f"Scaler feature count: {scaler.n_features_in_}")
print(f"Scaler feature names: {scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else 'Not available'}")
print(f"Scaler mean: {scaler.mean_}")
print(f"Scaler scale: {scaler.scale_}") 
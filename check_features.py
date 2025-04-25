import joblib

# Load feature names
feature_names = joblib.load('Model/feature_names_lightgbm.joblib')
print("Feature names used in training:")
for i, name in enumerate(feature_names):
    print(f"{i+1}. {name}") 
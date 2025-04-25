import lightgbm as lgb
import numpy as np
import os

# Try to load the model
model_path = os.path.join('Model', 'blackhole_detector.txt')
print(f"Loading model from {model_path}")

try:
    model = lgb.Booster(model_file=model_path)
    print("Model loaded successfully")
    
    # Try a test prediction
    test_data = np.random.rand(1, 13)  # 13 features
    prediction = model.predict(test_data)
    print(f"Test prediction: {prediction}")
    
except Exception as e:
    print(f"Error: {str(e)}") 
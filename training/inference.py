import numpy as np
import joblib
import torch

class BlackholeDetector:
    def __init__(self, model_path='models/relation_net_traced.pt', scaler_path='models/scaler.save'):
        # Load model
        self.model = torch.jit.load(model_path)
        self.model.eval()
        
        # Load scaler
        self.scaler = joblib.load(scaler_path)
        
    def preprocess_features(self, features):
        """Preprocess features using the saved scaler"""
        return self.scaler.transform([features])[0]
    
    def detect_blackhole(self, features):
        """Detect if the given features indicate a blackhole attack"""
        # Preprocess features
        processed_features = self.preprocess_features(features)
        
        # Convert to tensor
        x = torch.FloatTensor(processed_features)
        
        # Run inference
        with torch.no_grad():
            # Use the same features as both inputs for single-sample inference
            output = self.model(x.unsqueeze(0), x.unsqueeze(0))
            confidence = float(output.squeeze())
            is_blackhole = confidence > 0.5
        
        return {
            'is_blackhole': bool(is_blackhole),
            'confidence': confidence,
            'features': features.tolist() if hasattr(features, 'tolist') else features
        }

if __name__ == "__main__":
    # Example usage
    detector = BlackholeDetector()
    
    # Example features (replace with actual feature values)
    example_features = [0.0] * 8  # Assuming 8 features
    
    result = detector.detect_blackhole(example_features)
    print("Detection result:", result) 
import logging
import joblib
import numpy as np
from pathlib import Path

class BlackholeDetector:
    def __init__(self, model_path='blackhole_model.joblib'):
        self.model_path = Path(model_path)
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Load the trained model"""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
                
            self.model = joblib.load(self.model_path)
            logging.info(f"Successfully loaded model from {self.model_path}")
            
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise
            
    def predict(self, features):
        """Make predictions using the loaded model"""
        try:
            if self.model is None:
                raise ValueError("Model not loaded")
                
            # Convert features to numpy array if needed
            if not isinstance(features, np.ndarray):
                features = np.array(features)
                
            # Ensure features are 2D
            if features.ndim == 1:
                features = features.reshape(1, -1)
                
            # Make prediction
            prediction = self.model.predict(features)[0]
            probability = self.model.predict_proba(features)[0]
            
            logging.debug(f"Prediction: {prediction}, Probability: {probability}")
            return prediction, probability
            
        except Exception as e:
            logging.error(f"Error making prediction: {str(e)}")
            raise 
import logging
import numpy as np
from feature_extraction import FeatureExtractor
from notifier import Notifier

class BlackholeDetector:
    def __init__(self, model_path=None, threshold=0.5):
        self.logger = logging.getLogger(__name__)
        self.feature_extractor = FeatureExtractor()
        self.notifier = Notifier()
        self.threshold = threshold
        self.model = None
        if model_path:
            self.load_model(model_path)
            
    def load_model(self, model_path):
        """Load a trained model"""
        try:
            self.model = np.load(model_path, allow_pickle=True).item()
            self.logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            
    def process_packet(self, packet):
        """Process a single packet and detect blackholes"""
        try:
            # Extract features for the packet
            features = self.feature_extractor.extract_features(packet)
            if not features:
                self.logger.warning("No features extracted from packet")
                return
                
            self.logger.debug(f"Extracted features: {features}")
            
            # Make prediction if model is loaded
            if self.model:
                prediction = self._make_prediction(features)
                self.logger.debug(f"Made prediction: {prediction}")
                
                # Notify if blackhole detected
                if prediction > self.threshold:
                    self.notifier.notify_blackhole(packet['src_ip'], prediction)
                    
        except Exception as e:
            self.logger.error(f"Error processing packet: {str(e)}")
            
    def _make_prediction(self, features):
        """Make a prediction using the loaded model"""
        try:
            # Convert features to numpy array
            feature_array = np.array(list(features.values()))
            
            # Make prediction using model
            prediction = self.model.predict([feature_array])[0]
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            return 0.0 
import logging
import threading
import numpy as np
import time
import joblib
import os
from datetime import datetime
import lightgbm as lgb

class LightweightModelInfer(threading.Thread):
    def __init__(self, model_path='Model/blackhole_detector.joblib', scaler_path='Model/scaler.joblib', confidence_threshold=0.5):
        """Initialize the blackhole detector"""
        self.logger = logging.getLogger('BlackholeDetector')
        self.confidence_threshold = confidence_threshold
        
        # Define expected feature names in correct order
        self.feature_names = [
            'transmission_rate_per_1000_ms',
            'reception_rate_per_1000_ms',
            'transmission_count_per_sec',
            'reception_count_per_sec',
            'transmission_total_duration_per_sec',
            'reception_total_duration_per_sec',
            'transmission_reception_ratio',
            'count_ratio',
            'duration_ratio',
            'dao',
            'dis',
            'dio',
            'length'
        ]
        
        # Define valid ranges for features
        self.feature_ranges = {
            'transmission_rate_per_1000_ms': (0, 1000),
            'reception_rate_per_1000_ms': (0, 1000),
            'transmission_count_per_sec': (0, 100),
            'reception_count_per_sec': (0, 100),
            'transmission_total_duration_per_sec': (0, 1000),
            'reception_total_duration_per_sec': (0, 1000),
            'dao': (0, 100),
            'dis': (0, 100),
            'dio': (0, 100),
            'length': (0, 1500)
        }
        
        try:
            # Load model
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            self.model = joblib.load(model_path)
            self.logger.info(f"Loaded model from {model_path}")
            
            # Load scaler if available
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                self.logger.info(f"Loaded scaler from {scaler_path}")
            else:
                self.scaler = None
                self.logger.warning(f"No scaler found at {scaler_path}, features will not be scaled")
                
        except Exception as e:
            self.logger.error(f"Error initializing blackhole detector: {str(e)}")
            raise
        
    def validate_features(self, features):
        """Validate feature values are within expected ranges"""
        try:
            for name, (min_val, max_val) in self.feature_ranges.items():
                value = features.get(name, 0)
                if not (min_val <= value <= max_val):
                    self.logger.warning(f"Feature {name} value {value} outside expected range [{min_val}, {max_val}]")
                    # Clip the value to the valid range
                    features[name] = np.clip(value, min_val, max_val)
            return True
        except Exception as e:
            self.logger.error(f"Error validating features: {str(e)}")
            return False

    def preprocess_features(self, features):
        """Preprocess features before model input"""
        try:
            # Ensure all required features are present
            required_features = [
                'transmission_rate_per_1000_ms',
                'reception_rate_per_1000_ms',
                'transmission_count_per_sec',
                'reception_count_per_sec',
                'transmission_total_duration_per_sec',
                'reception_total_duration_per_sec',
                'transmission_reception_ratio',
                'count_ratio',
                'duration_ratio',
                'dao',
                'dis',
                'dio',
                'length'
            ]
            
            # Create feature vector in the exact order expected by the model
            feature_vector = np.array([
                features.get('transmission_rate_per_1000_ms', 0),
                features.get('reception_rate_per_1000_ms', 0),
                features.get('transmission_count_per_sec', 0),
                features.get('reception_count_per_sec', 0),
                features.get('transmission_total_duration_per_sec', 0),
                features.get('reception_total_duration_per_sec', 0),
                features.get('transmission_reception_ratio', 0),
                features.get('count_ratio', 0),
                features.get('duration_ratio', 0),
                features.get('dao', 0),
                features.get('dis', 0),
                features.get('dio', 0),
                features.get('length', 0)
            ]).reshape(1, -1)
            
            # Scale features if scaler is available
            if self.scaler is not None:
                feature_vector = self.scaler.transform(feature_vector)
            
            return feature_vector
            
        except Exception as e:
            self.logger.error(f"Error preprocessing features: {str(e)}")
            return None

    def detect_blackhole(self, features):
        """Detect blackhole attacks based on features"""
        try:
            # Preprocess features
            processed_features = self.preprocess_features(features)
            if processed_features is None:
                return None
                
            # Get prediction probability
            prob = self.model.predict_proba(processed_features.reshape(1, -1))[0][1]
            
            # Create detection result
            detection = {
                'timestamp': datetime.now().isoformat(),
                'source_ip': features.get('source_ip', 'unknown'),
                'dest_ip': features.get('dest_ip', 'unknown'),
                'confidence': float(prob),
                'features': {
                    'transmission_rate': features.get('transmission_rate_per_1000_ms', 0),
                    'reception_rate': features.get('reception_rate_per_1000_ms', 0),
                    'transmission_count': features.get('transmission_count_per_sec', 0),
                    'reception_count': features.get('reception_count_per_sec', 0),
                    'transmission_duration': features.get('transmission_total_duration_per_sec', 0),
                    'reception_duration': features.get('reception_total_duration_per_sec', 0),
                    'dao_count': features.get('dao', 0),
                    'dis_count': features.get('dis', 0),
                    'dio_count': features.get('dio', 0),
                    'transmission_reception_ratio': features.get('transmission_reception_ratio', 0),
                    'count_ratio': features.get('count_ratio', 0),
                    'duration_ratio': features.get('duration_ratio', 0)
                }
            }
            
            # Determine if blackhole exists based on confidence threshold
            detection['is_blackhole'] = prob >= self.confidence_threshold
            
            return detection
            
        except Exception as e:
            self.logger.error(f"Error in blackhole detection: {str(e)}")
            return None
            
    def run(self):
        """Main detection loop"""
        self.running = True
        self.logger.info("Starting blackhole detection thread")
        
        while self.running:
            try:
                # Get features from queue
                features = self.feature_queue.get(timeout=1.0)
                
                # Process features
                detection = self.detect_blackhole(features)
                
                if detection:
                    self.detection_queue.put(detection)
                    self.detection_count += 1
                    
            except Exception as e:
                if self.running:  # Only log if we're still supposed to be running
                    self.logger.error(f"Error in detection loop: {str(e)}")
                time.sleep(0.1)  # Prevent tight loop on errors
                
    def stop(self):
        """Stop the detection thread"""
        if self.running:
            self.running = False
            if self.is_alive():
                self.join(timeout=1.0)
            self.logger.info(f"Detection stopped. Total detections: {self.detection_count}")

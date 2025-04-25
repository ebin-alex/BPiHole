import logging
import threading
import numpy as np
import time
from datetime import datetime
import joblib
import os
import queue
import pandas as pd
import lightgbm as lgb

class ModelInference(threading.Thread):
    def __init__(self, feature_queue, detection_queue):
        super().__init__()
        self.feature_queue = feature_queue
        self.detection_queue = detection_queue
        self.running = False
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.scaler = None
        self.threshold = 0.7  # Increased threshold to reduce false positives
        self.feature_ranges = {
            'transmission_rate_per_1000_ms': (0, 1000),
            'reception_rate_per_1000_ms': (0, 1000),
            'transmission_count_per_sec': (0, 1000),
            'reception_count_per_sec': (0, 1000),
            'transmission_total_duration_per_sec': (0, 3600),
            'reception_total_duration_per_sec': (0, 3600),
            'dao': (0, 1),
            'dis': (0, 1),
            'dio': (0, 1),
            'length': (0, 1500)
        }
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
        
    def load_model(self):
        """Load the trained model and scaler"""
        try:
            model_path = os.path.join('Model', 'blackhole_detector.txt')
            scaler_path = os.path.join('Model', 'scaler.joblib')
            
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                self.logger.error("Model or scaler file not found")
                return False
                
            self.model = lgb.Booster(model_file=model_path)
            self.scaler = joblib.load(scaler_path)
            self.logger.info("Model and scaler loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return False
            
    def validate_features(self, features):
        """Validate feature values are within expected ranges"""
        try:
            for name, value in features.items():
                if name in self.feature_ranges:
                    min_val, max_val = self.feature_ranges[name]
                    if not (min_val <= value <= max_val):
                        self.logger.warning(f"Feature {name} value {value} outside expected range [{min_val}, {max_val}]")
                        # Clip value to range
                        features[name] = max(min_val, min(value, max_val))
            return True
        except Exception as e:
            self.logger.error(f"Error validating features: {str(e)}")
            return False
            
    def calibrate_confidence(self, raw_confidence):
        """Calibrate confidence score to better reflect true probabilities"""
        try:
            # Apply sigmoid calibration
            calibrated = 1 / (1 + np.exp(-(raw_confidence - self.threshold)))
            return float(calibrated)
        except Exception as e:
            self.logger.error(f"Error calibrating confidence: {str(e)}")
            return raw_confidence
            
    def adjust_threshold(self, precision_weight=0.5, recall_weight=0.5):
        """Dynamically adjust the detection threshold based on desired precision/recall trade-off"""
        try:
            # Load threshold evaluation results if available
            threshold_file = 'Model/threshold_evaluation.csv'
            if os.path.exists(threshold_file):
                df = pd.read_csv(threshold_file)
                # Calculate weighted score for each threshold
                df['score'] = (precision_weight * df['precision'] + 
                             recall_weight * df['recall'])
                # Select threshold with highest score
                best_threshold = df.loc[df['score'].idxmax(), 'threshold']
                self.threshold = float(best_threshold)
                self.logger.info(f"Adjusted threshold to {self.threshold}")
        except Exception as e:
            self.logger.error(f"Error adjusting threshold: {str(e)}")
            
    def preprocess_features(self, features):
        """Preprocess features for model input"""
        try:
            # Get base features
            length = features.get('length', 0)
            transmission_rate = features.get('transmission_rate_per_1000_ms', 0)
            reception_rate = features.get('reception_rate_per_1000_ms', 0)
            transmission_count = features.get('transmission_count_per_sec', 0)
            reception_count = features.get('reception_count_per_sec', 0)
            transmission_duration = features.get('transmission_total_duration_per_sec', 0)
            reception_duration = features.get('reception_total_duration_per_sec', 0)
            
            # Calculate averages
            transmission_average = transmission_duration / transmission_count if transmission_count != 0 else 0
            reception_average = reception_duration / reception_count if reception_count != 0 else 0
            
            # Calculate ratios with protection against division by zero
            tx_rx_rate_ratio = transmission_rate / reception_rate if reception_rate != 0 else 0
            tx_rx_count_ratio = transmission_count / reception_count if reception_count != 0 else 0
            tx_rx_duration_ratio = transmission_duration / reception_duration if reception_duration != 0 else 0
            
            # Create feature vector in correct order
            feature_vector = [
                length,
                transmission_rate,
                reception_rate,
                transmission_count,
                reception_count,
                transmission_duration,
                reception_duration,
                tx_rx_rate_ratio,
                tx_rx_count_ratio,
                tx_rx_duration_ratio,
                features.get('dao', 0),
                features.get('dis', 0),
                features.get('dio', 0)
            ]
            
            # Create DataFrame with feature names
            feature_df = pd.DataFrame([feature_vector], columns=self.feature_names)
            
            # Scale features if scaler is available
            if self.scaler is not None:
                feature_df = pd.DataFrame(
                    self.scaler.transform(feature_df),
                    columns=self.feature_names
                )
            
            return feature_df.values
            
        except Exception as e:
            self.logger.error(f"Error preprocessing features: {str(e)}")
            return None
        
    def detect_blackhole(self, features):
        """Detect blackhole behavior using the model"""
        try:
            if self.model is None:
                self.logger.error("Model not loaded")
                return False, 0.0
                
            # Preprocess features
            X = self.preprocess_features(features)
            if X is None:
                return False, 0.0
                
            # Make prediction using LightGBM's predict method
            raw_prediction = self.model.predict(X)
            confidence = float(raw_prediction[0])
            
            # Calibrate confidence
            calibrated_confidence = self.calibrate_confidence(confidence)
            
            # Determine if blackhole based on threshold
            is_blackhole = calibrated_confidence >= self.threshold
            
            return bool(is_blackhole), float(calibrated_confidence)
            
        except Exception as e:
            self.logger.error(f"Error in blackhole detection: {str(e)}")
            return False, 0.0
        
    def run(self):
        """Main processing loop"""
        if not self.load_model():
            self.logger.error("Failed to load model, stopping inference thread")
            return
            
        self.running = True
        self.logger.info("Model inference started")
        
        while self.running:
            try:
                if not self.feature_queue.empty():
                    features = self.feature_queue.get()
                    
                    # Detect blackhole
                    is_blackhole, confidence = self.detect_blackhole(features)
                    
                    # Create detection result
                    detection = {
                        'timestamp': datetime.now().isoformat(),
                        'source_ip': features.get('source_ip', 'unknown'),
                        'destination_ip': features.get('destination_ip', 'unknown'),
                        'is_blackhole': is_blackhole,
                        'confidence': confidence,
                        'features': {k: v for k, v in features.items() if k in self.feature_names}
                    }
                    
                    # Send detection to queue
                    self.detection_queue.put(detection)
                    
                    # Log detection
                    if is_blackhole:
                        self.logger.warning(
                            f"Blackhole detected! Source: {detection['source_ip']}, "
                            f"Dest: {detection['destination_ip']}, Confidence: {confidence:.2f}"
                        )
                    
            except Exception as e:
                self.logger.error(f"Error in inference loop: {str(e)}")
                
    def stop(self):
        """Stop the inference thread"""
        if self.running:
            self.running = False
            if self.is_alive():
                self.join(timeout=1.0)
            self.logger.info("Model inference stopped")

    def start_processing(self):
        """Start the inference thread"""
        self.logger.info("Starting model inference...")
        super().start()

    def process_packet(self, packet):
        """Process a single packet through the model"""
        try:
            # Extract features
            features = self.feature_extractor.extract_features(packet)
            if not features:
                self.logger.error("Failed to extract features from packet")
                return
            
            # Convert features to DataFrame
            feature_df = pd.DataFrame([features])
            
            # Make prediction
            prediction = self.model.predict(feature_df)
            probability = self.model.predict_proba(feature_df)[0][1]  # Probability of being malicious
            
            # Create detection if probability exceeds threshold
            if probability > self.threshold:
                detection = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'source_ip': features.get('source_ip', 'unknown'),
                    'destination_ip': features.get('destination_ip', 'unknown'),
                    'confidence': float(probability),
                    'is_blackhole': probability > self.threshold,
                    'features': features
                }
                
                self.logger.info(f"Detection created: {detection}")
                self.detections.append(detection)
                
                # Notify dashboard
                if self.dashboard:
                    self.dashboard.add_detection(detection)
            
        except Exception as e:
            self.logger.error(f"Error processing packet: {str(e)}") 
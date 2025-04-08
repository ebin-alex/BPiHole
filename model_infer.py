import logging
import threading
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
from datetime import datetime
import os

class MAMLModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MAMLModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        x = self.layer1(inputs)
        x = self.relu(x)
        x = self.layer2(x)
        return x

class ModelInfer(threading.Thread):
    def __init__(self, feature_queue, detection_queue):
        super().__init__()
        self.feature_queue = feature_queue
        self.detection_queue = detection_queue
        self.running = False
        self.logger = logging.getLogger(__name__)
        self.detection_count = 0
        
        # More sensitive thresholds
        self.packet_rate_threshold = 5  # packets per second (reduced from 10)
        self.duration_threshold = 3  # seconds (reduced from 5)
        
    def detect_blackhole(self, features):
        """Detect blackhole behavior using simple threshold-based rules"""
        is_blackhole = False
        confidence = 0.0
        
        self.logger.debug(f"Analyzing features for blackhole detection: {features}")
        
        # Check if packet rate suddenly drops
        if features['duration'] > self.duration_threshold:
            if features['packet_rate'] < self.packet_rate_threshold:
                is_blackhole = True
                # Calculate confidence based on how far below threshold
                confidence = max(0.3, min(0.9, 1 - (features['packet_rate'] / self.packet_rate_threshold)))
                self.logger.debug(f"Blackhole detected: rate={features['packet_rate']}, threshold={self.packet_rate_threshold}")
                
        detection = {
            'source_ip': features['source_ip'],
            'dest_ip': features['dest_ip'],
            'is_blackhole': is_blackhole,
            'confidence': confidence,
            'timestamp': features['timestamp'],
            'packet_rate': features['packet_rate'],
            'duration': features['duration']
        }
        
        self.logger.debug(f"Generated detection: {detection}")
        return detection
        
    def run(self):
        """Main processing loop"""
        self.running = True
        self.logger.info("Model inference started")
        
        while self.running:
            try:
                if not self.feature_queue.empty():
                    features = self.feature_queue.get()
                    self.detection_count += 1
                    
                    self.logger.debug(f"Processing features: {features}")
                    detection = self.detect_blackhole(features)
                    
                    # Send detection to the dashboard
                    self.detection_queue.put(detection)
                    
                    # Log progress
                    if self.detection_count % 10 == 0:
                        self.logger.info(f"Processed {self.detection_count} feature sets")
                else:
                    time.sleep(0.1)  # Prevent CPU overload
                    
            except Exception as e:
                self.logger.error(f"Error in model inference: {str(e)}")
                
    def stop(self):
        """Stop the model inference thread"""
        if self.running:
            self.running = False
            if self.is_alive():
                self.join(timeout=1.0)
            self.logger.info(f"Model inference stopped. Total detections: {self.detection_count}")

    def preprocess_features(self, features):
        """Preprocess features for model input"""
        # Convert to tensor and add batch dimension
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        return features_tensor
        
    def infer(self, feature_data):
        """Perform inference on extracted features"""
        try:
            # Preprocess features
            features = self.preprocess_features(feature_data['features'])
            
            # Perform inference
            with torch.no_grad():
                output = self.model(features)
                prediction = torch.sigmoid(output).item()
                
            # Create detection result
            detection = {
                'source_ip': feature_data['source_ip'],
                'timestamp': feature_data['timestamp'],
                'is_blackhole': prediction > 0.5,  # Threshold at 0.5
                'confidence': prediction
            }
            
            return detection
            
        except Exception as e:
            self.logger.error(f"Error during inference: {str(e)}")
            return None 
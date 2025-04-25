import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def load_data(file_path):
    """Load and preprocess the dataset"""
    logger.info(f"Loading dataset from {file_path}")
    try:
        # Load the dataset
        df = pd.read_csv(file_path)
        logger.info(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")
        
        # Check for missing values
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            logger.warning(f"Dataset contains {missing_values} missing values")
            # Fill missing values with median
            df = df.fillna(df.median())
        
        # Separate features and target
        X = df.drop(['category', 'label', 'time', 'source', 'destination', 'info'], axis=1)
        y = df['label']
        
        logger.info(f"Features: {X.columns.tolist()}")
        logger.info(f"Class distribution: {y.value_counts().to_dict()}")
        
        return X, y
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

def train_models(X, y):
    """Train multiple models and select the best one"""
    logger.info("Starting model training")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    logger.info(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    joblib.dump(scaler, 'Model/scaler.joblib')
    logger.info("Scaler saved to Model/scaler.joblib")
    
    # Define models to try
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    # Train and evaluate each model
    best_model = None
    best_score = 0
    results = {}
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        start_time = time.time()
        
        # For RandomForest and GradientBoosting, use a smaller subset for faster training
        if name in ['RandomForest', 'GradientBoosting']:
            # Use 20% of data for training to speed up the process
            X_train_subset, _, y_train_subset, _ = train_test_split(
                X_train_scaled, y_train, train_size=0.2, random_state=42, stratify=y_train
            )
            model.fit(X_train_subset, y_train_subset)
        else:
            model.fit(X_train_scaled, y_train)
        
        # Evaluate on test set
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        training_time = time.time() - start_time
        logger.info(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Time: {training_time:.2f}s")
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'training_time': training_time
        }
        
        # Update best model
        if f1 > best_score:
            best_score = f1
            best_model = (name, model)
    
    logger.info(f"Best model: {best_model[0]} with F1 score: {best_score:.4f}")
    
    # Save the best model
    os.makedirs('Model', exist_ok=True)
    joblib.dump(best_model[1], 'Model/blackhole_detector.joblib')
    logger.info("Best model saved to Model/blackhole_detector.joblib")
    
    # Save model metadata
    metadata = {
        'best_model': best_model[0],
        'feature_names': X.columns.tolist(),
        'results': {name: {k: v for k, v in result.items() if k != 'model'} for name, result in results.items()}
    }
    joblib.dump(metadata, 'Model/model_metadata.joblib')
    logger.info("Model metadata saved to Model/model_metadata.joblib")
    
    return best_model[1], scaler, X_test_scaled, y_test

def create_inference_script():
    """Create a lightweight inference script for Raspberry Pi"""
    logger.info("Creating lightweight inference script for Raspberry Pi")
    
    with open('lightweight_model_infer.py', 'w') as f:
        f.write('''import logging
import threading
import numpy as np
import time
import joblib
import os

class LightweightModelInfer(threading.Thread):
    def __init__(self, feature_queue, detection_queue):
        super().__init__()
        self.feature_queue = feature_queue
        self.detection_queue = detection_queue
        self.running = False
        self.logger = logging.getLogger(__name__)
        self.detection_count = 0
        
        # Load model and scaler
        self.model = None
        self.scaler = None
        self.load_model()
        
    def load_model(self):
        """Load the trained model and scaler"""
        try:
            model_path = os.path.join('Model', 'blackhole_detector.joblib')
            scaler_path = os.path.join('Model', 'scaler.joblib')
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")
            
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler not found at {scaler_path}")
            
            self.logger.info(f"Loading model from {model_path}")
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.logger.info("Model and scaler loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
        
    def preprocess_features(self, features):
        """Convert features to model input format"""
        try:
            # Extract numerical features
            feature_names = [
                'length',
                'transmission_rate_per_1000_ms',
                'reception_rate_per_1000_ms',
                'transmission_average_per_sec',
                'reception_average_per_sec',
                'transmission_count_per_sec',
                'reception_count_per_sec',
                'transmission_total_duration_per_sec',
                'reception_total_duration_per_sec',
                'dao',
                'dis',
                'dio'
            ]
            
            feature_vector = np.array([features.get(name, 0) for name in feature_names], dtype=np.float32)
            
            # Scale features
            feature_vector = self.scaler.transform(feature_vector.reshape(1, -1))[0]
            
            return feature_vector
            
        except Exception as e:
            self.logger.error(f"Error preprocessing features: {str(e)}")
            return None
        
    def detect_blackhole(self, features):
        """Detect blackhole behavior using the lightweight model"""
        try:
            # Preprocess features
            x = self.preprocess_features(features)
            if x is None:
                return None
                
            # Get model prediction
            confidence = self.model.predict_proba(x.reshape(1, -1))[0][1]
            
            # Determine if it's a blackhole
            is_blackhole = confidence > 0.5  # Threshold can be adjusted
            
            # Create detection result
            detection = {
                'timestamp': int(time.time()),
                'source_ip': features.get('source', 'unknown'),
                'destination_ip': features.get('destination', 'unknown'),
                'is_blackhole': is_blackhole,
                'confidence': float(confidence),
                'packet_rate': features.get('transmission_rate_per_1000_ms', 0),
                'duration': features.get('transmission_total_duration_per_sec', 0),
                'total_packets': features.get('transmission_count_per_sec', 0),
                'rate_change': features.get('reception_rate_per_1000_ms', 0)
            }
            
            self.logger.debug(f"Detection result: {detection}")
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
''')
    
    logger.info("Lightweight inference script created: lightweight_model_infer.py")

if __name__ == "__main__":
    try:
        # Load the dataset
        X, y = load_data('dataset/blackhole.csv')
        
        # Train models and select the best one
        best_model, scaler, X_test, y_test = train_models(X, y)
        
        # Create the lightweight inference script
        create_inference_script()
        
        logger.info("Model training completed successfully")
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}") 
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('retrain.log'),
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

def preprocess_features(X):
    """Preprocess features to match inference requirements"""
    logger.info("Preprocessing features...")
    
    # Calculate ratio features
    X['transmission_reception_ratio'] = (
        X['transmission_rate_per_1000_ms'] / 
        X['reception_rate_per_1000_ms'].replace(0, 1)  # Avoid division by zero
    )
    X['count_ratio'] = (
        X['transmission_count_per_sec'] / 
        X['reception_count_per_sec'].replace(0, 1)  # Avoid division by zero
    )
    X['duration_ratio'] = (
        X['transmission_total_duration_per_sec'] / 
        X['reception_total_duration_per_sec'].replace(0, 1)  # Avoid division by zero
    )
    
    # Select and order features to match inference
    feature_names = [
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
    
    X = X[feature_names]
    logger.info(f"Selected features: {feature_names}")
    
    return X

def train_model(X, y):
    """Train LightGBM model"""
    logger.info("Starting model training")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    logger.info(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    os.makedirs('Model', exist_ok=True)
    joblib.dump(scaler, 'Model/scaler.joblib')
    logger.info("Scaler saved to Model/scaler.joblib")
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train_scaled, label=y_train)
    test_data = lgb.Dataset(X_test_scaled, label=y_test, reference=train_data)
    
    # Define model parameters
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    # Train the model
    logger.info("Training LightGBM model...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[test_data],
        callbacks=[lgb.early_stopping(stopping_rounds=10)]
    )
    
    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)
    
    logger.info(f"Model performance:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    
    # Save the model
    model.save_model('Model/blackhole_detector.txt')
    logger.info("Model saved to Model/blackhole_detector.txt")
    
    # Save model metadata
    metadata = {
        'feature_names': X.columns.tolist(),
        'performance': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    }
    joblib.dump(metadata, 'Model/model_metadata.joblib')
    logger.info("Model metadata saved to Model/model_metadata.joblib")
    
    return model, scaler

if __name__ == "__main__":
    try:
        # Load the dataset
        X, y = load_data('dataset/blackhole.csv')
        
        # Preprocess features
        X = preprocess_features(X)
        
        # Train the model
        model, scaler = train_model(X, y)
        
        logger.info("Model training completed successfully")
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise 
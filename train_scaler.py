import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import argparse
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def train_scaler(dataset_path, output_path='feature_scaler.pkl'):
    """
    Train a StandardScaler on the dataset and save it
    
    Args:
        dataset_path: Path to the dataset CSV file
        output_path: Path to save the trained scaler
    """
    try:
        # Check if dataset exists
        if not os.path.exists(dataset_path):
            logging.error(f"Dataset file '{dataset_path}' not found")
            return False
            
        # Load the dataset
        logging.info(f"Loading dataset from '{dataset_path}'")
        dataset = pd.read_csv(dataset_path)
        
        # Drop columns that shouldn't be used for scaling
        columns_to_drop = ['category', 'label']
        if 'transmission_average_per_sec' in dataset.columns:
            columns_to_drop.append('transmission_average_per_sec')
            
        features = dataset.drop(columns=columns_to_drop)
        
        # Train the scaler
        logging.info("Training StandardScaler")
        scaler = StandardScaler()
        scaler.fit(features)
        
        # Save the scaler
        with open(output_path, 'wb') as f:
            pickle.dump(scaler, f)
            
        logging.info(f"Scaler saved to '{output_path}'")
        
        # Print feature names and their statistics
        logging.info("Feature statistics:")
        for i, feature_name in enumerate(features.columns):
            mean = scaler.mean_[i]
            scale = scaler.scale_[i]
            logging.info(f"{feature_name}: mean={mean:.4f}, scale={scale:.4f}")
            
        return True
        
    except Exception as e:
        logging.error(f"Error training scaler: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Train a StandardScaler on the dataset and save it')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset CSV file')
    parser.add_argument('--output', type=str, default='feature_scaler.pkl', help='Path to save the trained scaler')
    
    args = parser.parse_args()
    
    success = train_scaler(args.dataset, args.output)
    
    if success:
        print("Scaler training completed successfully")
    else:
        print("Scaler training failed")

if __name__ == "__main__":
    main() 
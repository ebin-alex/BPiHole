import os
import pandas as pd
import numpy as np
from tflite_runtime.interpreter import Interpreter
from sklearn.preprocessing import StandardScaler
import joblib

def main():
    try:
        print("Loading dataset...")
        dataset = pd.read_csv('dataset/blackhole.csv')
        print(f"Dataset loaded with shape: {dataset.shape}")
        
        # Prepare features and labels
        print("Preparing features and labels...")
        features = dataset.drop(columns=['category', 'label'])
        labels = dataset['label']
        
        # Outlier detection and removal using IQR
        print("Removing outliers...")
        Q1 = features.quantile(0.25)
        Q3 = features.quantile(0.75)
        IQR = Q3 - Q1
        mask = ~((features < (Q1 - 1.5 * IQR)) | (features > (Q3 + 1.5 * IQR))).any(axis=1)
        filtered_data = features[mask].copy()
        filtered_labels = labels[mask].copy()
        print(f"Removed {len(features) - len(filtered_data)} outliers")
        
        # Select features in the correct order
        selected_features = [
            'time', 'source', 'destination', 'length', 'info',
            'transmission_rate_per_1000_ms', 'reception_rate_per_1000_ms',
            'reception_average_per_sec', 'transmission_count_per_sec',
            'reception_count_per_sec', 'transmission_total_duration_per_sec',
            'reception_total_duration_per_sec', 'dao', 'dis', 'dio'
        ]
        filtered_data = filtered_data[selected_features]
        
        print(f"Selected features: {selected_features}")
        print(f"Features shape: {filtered_data.shape}, Labels shape: {filtered_labels.shape}")
        
        # Load and apply scaler
        print("Loading scaler...")
        scaler = joblib.load('models/scaler.save')
        normalized_features = scaler.transform(filtered_data)
        print("Features normalized")
        
        # Load TFLite model
        print("Loading TFLite model...")
        interpreter = Interpreter(model_path='models/relation_net_traced.pt')
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Make predictions
        print("Making predictions...")
        predictions = []
        for i in range(len(normalized_features)):
            # Set input tensor
            interpreter.set_tensor(input_details[0]['index'], [normalized_features[i]])
            
            # Run inference
            interpreter.invoke()
            
            # Get prediction
            output = interpreter.get_tensor(output_details[0]['index'])
            predictions.append(1 if output[0][0] > 0.5 else 0)
        
        # Calculate accuracy
        accuracy = np.mean(predictions == filtered_labels)
        print(f"\nAccuracy: {accuracy:.4f}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 
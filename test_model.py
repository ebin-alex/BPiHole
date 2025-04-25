import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class RelationNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.relation_layer = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def embed(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

    def forward(self, x1, x2):
        combined = torch.cat((x1, x2), dim=-1)
        relation_score = self.relation_layer(combined)
        return relation_score

def main():
    try:
        print("Loading dataset...")
        dataset = pd.read_csv('dataset/blackhole.csv')
        print(f"Dataset loaded with shape: {dataset.shape}")
        
        # Prepare features and labels
        print("Preparing features and labels...")
        features = dataset.drop(columns=['category', 'label', 'source', 'destination', 'info', 'time'])
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
        
        # Add engineered features
        print("Adding engineered features...")
        filtered_data.loc[:, 'transmission_reception_ratio'] = (
            filtered_data['transmission_rate_per_1000_ms'] / 
            filtered_data['reception_rate_per_1000_ms'].replace(0, 1)
        )
        filtered_data.loc[:, 'count_ratio'] = (
            filtered_data['transmission_count_per_sec'] / 
            filtered_data['reception_count_per_sec'].replace(0, 1)
        )
        filtered_data.loc[:, 'duration_ratio'] = (
            filtered_data['transmission_total_duration_per_sec'] / 
            filtered_data['reception_total_duration_per_sec'].replace(0, 1)
        )
        
        # Select important features
        selected_features = [
            'transmission_rate_per_1000_ms', 'reception_rate_per_1000_ms',
            'transmission_count_per_sec', 'reception_count_per_sec',
            'transmission_total_duration_per_sec', 'reception_total_duration_per_sec',
            'transmission_reception_ratio', 'count_ratio', 'duration_ratio',
            'dao', 'dis', 'dio', 'length'
        ]
        filtered_data = filtered_data[selected_features]
        
        print(f"Selected features: {selected_features}")
        print(f"Features shape: {filtered_data.shape}, Labels shape: {filtered_labels.shape}")
        
        # Load and apply scaler
        print("Loading scaler...")
        scaler = joblib.load('models/scaler.save')
        normalized_features = scaler.transform(filtered_data)
        print("Features normalized")
        
        # Convert to tensors
        features_tensor = torch.FloatTensor(normalized_features)
        labels_tensor = torch.LongTensor(filtered_labels.values)
        
        # Load model
        print("Loading model...")
        input_size = len(selected_features)
        hidden_size = 64
        model = RelationNetwork(input_size, hidden_size)
        model.load_state_dict(torch.load('models/relation_net.pt'))
        model.eval()
        print("Model loaded successfully")
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        features_tensor = features_tensor.to(device)
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = model.embed(features_tensor)
        
        # Calculate class prototypes
        print("Calculating class prototypes...")
        prototypes = []
        for label in torch.unique(labels_tensor):
            class_embeddings = embeddings[labels_tensor == label]
            prototypes.append(class_embeddings.mean(dim=0))
        prototypes = torch.stack(prototypes)
        
        # Make predictions
        print("Making predictions...")
        predictions = []
        with torch.no_grad():
            for embedding in embeddings:
                relation_scores = []
                for prototype in prototypes:
                    relation_score = model(embedding.unsqueeze(0), prototype.unsqueeze(0))
                    relation_scores.append(relation_score)
                relation_scores = torch.cat(relation_scores).view(-1)
                predicted_label = torch.argmax(relation_scores).item()
                predictions.append(predicted_label)
        
        # Convert to numpy arrays for evaluation
        predictions = np.array(predictions)
        true_labels = filtered_labels.values
        
        # Evaluate
        print("\nModel Evaluation:")
        print("Confusion Matrix:")
        print(confusion_matrix(true_labels, predictions))
        print("\nClassification Report:")
        print(classification_report(true_labels, predictions))
        print("\nAccuracy Score:")
        print(accuracy_score(true_labels, predictions))
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 
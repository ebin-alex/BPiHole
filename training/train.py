import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch.utils.data import Dataset, DataLoader
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

class BlackholeDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels.values if hasattr(labels, 'values') else labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    best_val_loss = float('inf')
    best_model = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            # Forward pass to generate embeddings
            embeddings = model.embed(features)
            
            # Calculate relation scores between each embedding and class prototypes
            prototypes = []
            for label in torch.unique(labels):
                class_embeddings = embeddings[labels == label]
                prototypes.append(class_embeddings.mean(dim=0))
            prototypes = torch.stack(prototypes)
            
            relation_scores = []
            targets = []
            for idx, embedding in enumerate(embeddings):
                for proto_idx, prototype in enumerate(prototypes):
                    relation_score = model(embedding.unsqueeze(0), prototype.unsqueeze(0))
                    relation_scores.append(relation_score)
                    targets.append(1.0 if labels[idx] == proto_idx else 0.0)
            
            relation_scores = torch.cat(relation_scores).view(-1)
            targets = torch.tensor(targets, dtype=torch.float32).to(device)
            
            # Calculate loss
            loss = criterion(relation_scores, targets)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (relation_scores > 0.5).float()
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                embeddings = model.embed(features)
                
                # Calculate class prototypes
                prototypes = []
                for label in torch.unique(labels):
                    class_embeddings = embeddings[labels == label]
                    prototypes.append(class_embeddings.mean(dim=0))
                prototypes = torch.stack(prototypes)
                
                relation_scores = []
                targets = []
                for idx, embedding in enumerate(embeddings):
                    for proto_idx, prototype in enumerate(prototypes):
                        relation_score = model(embedding.unsqueeze(0), prototype.unsqueeze(0))
                        relation_scores.append(relation_score)
                        targets.append(1.0 if labels[idx] == proto_idx else 0.0)
                
                relation_scores = torch.cat(relation_scores).view(-1)
                targets = torch.tensor(targets, dtype=torch.float32).to(device)
                
                loss = criterion(relation_scores, targets)
                val_loss += loss.item()
                
                predicted = (relation_scores > 0.5).float()
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        val_loss /= len(val_loader)
        val_acc = 100 * correct / total
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
    
    return best_model

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
        filtered_data = features[mask].copy()  # Create a copy to avoid SettingWithCopyWarning
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
        
        # Normalize features
        print("Normalizing features...")
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(filtered_data)
        print("Features normalized")
        
        # Save scaler for inference
        print("Saving scaler...")
        os.makedirs('models', exist_ok=True)
        joblib.dump(scaler, 'models/scaler.save')
        print("Scaler saved")
        
        # Split data with stratification
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            normalized_features, filtered_labels, test_size=0.3, random_state=42, stratify=filtered_labels
        )
        print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
        
        print("Creating datasets...")
        train_dataset = BlackholeDataset(X_train, y_train)
        test_dataset = BlackholeDataset(X_test, y_test)
        
        print("Creating data loaders...")
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16)
        print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
        
        print("Initializing model...")
        input_size = len(selected_features)
        hidden_size = 64
        model = RelationNetwork(input_size, hidden_size)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        print(f"Model initialized with input size: {input_size}, hidden size: {hidden_size}")
        
        print("Starting training...")
        best_model = train_model(model, train_loader, test_loader, criterion, optimizer)
        
        print("Saving model...")
        model.load_state_dict(best_model)
        torch.save(model.state_dict(), 'models/relation_net.pt')
        
        # Export to TorchScript for deployment
        print("Exporting model to TorchScript...")
        model.eval()
        example_input = torch.randn(1, input_size)
        traced_script_module = torch.jit.trace(model, (example_input, example_input))
        traced_script_module.save("models/relation_net_traced.pt")
        
        print("Training complete! Models saved in 'models' directory")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 
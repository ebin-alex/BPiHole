#!/bin/bash

# Install required packages
echo "Installing required packages..."
sudo apt-get update
sudo apt-get install -y python3 python3-pip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip3 install pandas numpy scikit-learn joblib

# Create necessary directories
echo "Creating directories..."
mkdir -p models dataset

# Test the model
echo "Testing the model..."
python3 test_model.py 
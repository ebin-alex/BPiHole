#!/bin/bash

# Define the Raspberry Pi hostname
PI_HOST="ebinalex@ebinalex.local"

# Define the destination directory on the Raspberry Pi
PI_DEST_DIR="/home/ebinalex/BPiHole"

# Create the destination directory if it doesn't exist
ssh $PI_HOST "mkdir -p $PI_DEST_DIR/Model"

# Copy core Python files
echo "Copying core Python files..."
scp main.py $PI_HOST:$PI_DEST_DIR/
scp model_infer.py $PI_HOST:$PI_DEST_DIR/
scp feature_extraction.py $PI_HOST:$PI_DEST_DIR/
scp packet_sniffer.py $PI_HOST:$PI_DEST_DIR/
scp hardware_controllers.py $PI_HOST:$PI_DEST_DIR/
scp dashboard.py $PI_HOST:$PI_DEST_DIR/
scp notifier.py $PI_HOST:$PI_DEST_DIR/
scp requirements.txt $PI_HOST:$PI_DEST_DIR/

# Copy model files
echo "Copying model files..."
scp Model/blackhole_detector.joblib $PI_HOST:$PI_DEST_DIR/Model/
scp Model/scaler.joblib $PI_HOST:$PI_DEST_DIR/Model/

echo "All files copied successfully!" 
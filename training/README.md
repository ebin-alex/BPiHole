# Blackhole Detection Model Training

This directory contains the code for training and deploying a TensorFlow Lite model for blackhole attack detection.

## Directory Structure
```
training/
├── train.py           # Training script
├── inference.py       # Inference script for Raspberry Pi
├── requirements.txt   # Python dependencies
└── models/           # Directory for saved models (created during training)
    ├── relation_net.tflite  # Converted TFLite model
    └── scaler.save         # Saved feature scaler
```

## Training the Model

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run training:
```bash
python train.py
```

The training script will:
- Load and preprocess the dataset
- Train the RelationNetwork model
- Convert the model to TensorFlow Lite format
- Save the model and scaler in the `models/` directory

## Deploying to Raspberry Pi

1. Install TensorFlow Lite Runtime on Raspberry Pi:
```bash
pip3 install tflite-runtime numpy joblib
```

2. Copy the following files to your Raspberry Pi:
- `models/relation_net.tflite`
- `models/scaler.save`
- `inference.py`

3. Use the model in your application by importing the `BlackholeDetector` class:
```python
from inference import BlackholeDetector

detector = BlackholeDetector()
result = detector.detect_blackhole(features)
```

## Model Details

- Architecture: RelationNetwork with simplified structure
- Input: Network traffic features
- Output: Binary classification (0: Normal, 1: Blackhole)
- Optimization: TensorFlow Lite with float16 quantization
- Performance: Optimized for Raspberry Pi deployment 
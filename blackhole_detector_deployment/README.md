# Blackhole Detection System

This is a Python-based system for detecting blackhole attacks in network traffic. The system uses machine learning to analyze network packets and identify potential blackhole attacks.

## Directory Structure

```
blackhole_detector_deployment/
├── src/                    # Source code files
├── models/                 # Model and scaler files
├── config/                 # Configuration files
├── logs/                   # Log files
└── tests/                  # Test files
```

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the detection system:
   ```bash
   python src/test_main.py
   ```

2. Access the dashboard:
   - Open a web browser
   - Navigate to `http://localhost:5000`

## Files Description

### Source Files
- `test_main.py`: Main entry point for the detection system
- `dashboard.py`: Web dashboard for visualizing detections
- `model_infer.py`: Model inference and prediction logic
- `feature_extractor.py`: Network packet feature extraction
- `mock_packet_sniffer.py`: Mock packet capture for testing
- `mock_notifier.py`: Mock notification system

### Model Files
- `blackhole_detector.txt`: Trained LightGBM model
- `scaler.joblib`: Feature scaler
- `model_metadata.joblib`: Model metadata

## Logs

Log files are stored in the `logs` directory. The main log file is `blackhole_detector.log`.

## Support

For any issues or questions, please refer to the documentation or contact the system administrator. 
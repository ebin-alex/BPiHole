# Raspberry Pi Blackhole Detection System

A lightweight intrusion detection system for IoT networks using Few-Shot Learning (Relation Network) deployed on a Raspberry Pi 3 Model B.

## Features

- Real-time network traffic monitoring
- Feature extraction from network packets
- Blackhole detection using pre-trained Relation Network model
- Multiple alert mechanisms:
  - LED alerts (via MCP23017 GPIO expander)
  - Buzzer alerts
  - OLED display notifications
  - MQTT alerts
  - Web dashboard
- Lightweight router-style dashboard interface
- Efficient resource usage optimized for Raspberry Pi 3

## Hardware Requirements

- Raspberry Pi 3 Model B v1.2
- MCP23017 I2C GPIO expander
- SSD1306 OLED display
- Optional: Buzzer
- Network connectivity

## Software Requirements

- Raspberry Pi OS Lite (headless)
- Python 3.7+
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/blackhole-detector.git
   cd blackhole-detector
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   Create a `.env` file with the following:
   ```
   MQTT_BROKER=your_mqtt_broker
   MQTT_PORT=1883
   MQTT_TOPIC=blackhole/alerts
   ```

4. Prepare your trained model:
   - If you have a model trained using the provided training script, convert it to the format expected by this system:
     ```bash
     python convert_model.py --input /path/to/your/maml_model.pth --output relation_net.pth
     ```
   - If you have a dataset, train a StandardScaler for feature normalization:
     ```bash
     python train_scaler.py --dataset /path/to/your/blackhole.csv
     ```

## Usage

1. Start the system:
   ```bash
   python main.py
   ```

2. Access the dashboard:
   - Open a web browser and navigate to `http://<raspberry_pi_ip>:5000`
   - The dashboard will show real-time system status and alerts

3. Monitor alerts:
   - Physical alerts: LED will blink, buzzer will sound, OLED will display alerts
   - Web dashboard: Shows detection history and system status
   - MQTT: Alerts are published to the configured MQTT topic

## System Architecture

The system consists of several components:

1. **Packet Sniffer**: Captures network traffic using scapy
2. **Feature Extractor**: Processes packets and extracts relevant features
3. **Model Inference**: Runs the Relation Network model for detection
4. **Notifier**: Handles all alert mechanisms
5. **Dashboard**: Provides web interface for monitoring

## Feature Extraction

The system extracts the following features from network packets:

1. **Basic Packet Statistics**:
   - Packet count
   - Average packet size
   - Standard deviation of packet size

2. **Protocol Distribution**:
   - Protocol entropy (measure of protocol diversity)

3. **TTL Statistics**:
   - Average TTL
   - Standard deviation of TTL

4. **Inter-arrival Time Statistics**:
   - Average inter-arrival time
   - Standard deviation of inter-arrival time

5. **Time-based Features**:
   - Time since last packet

These features are normalized using a StandardScaler trained on your dataset before being fed into the model.

## Development

### Adding New Features

1. Create new Python modules in the project root
2. Update the main orchestration in `main.py`
3. Add new routes to `dashboard.py` if needed
4. Update the dashboard template in `templates/index.html`

### Testing

1. Use sample pcap files for testing packet capture
2. Simulate network traffic for testing detection
3. Test hardware components individually using provided scripts

## Troubleshooting

1. **Permission Issues**:
   - Ensure proper permissions for network capture
   - Run with sudo if needed: `sudo python main.py`

2. **Hardware Issues**:
   - Check I2C connections
   - Verify GPIO connections
   - Test individual components using provided scripts

3. **Performance Issues**:
   - Monitor CPU and memory usage
   - Adjust feature extraction window size
   - Optimize model inference if needed

4. **Model Issues**:
   - Ensure the model architecture matches the one used during training
   - Verify that the features extracted match those used during training
   - Check that the StandardScaler is properly trained on your dataset

## License

MIT License - See LICENSE file for details

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request 
import logging
import threading
import time
from datetime import datetime
import paho.mqtt.client as mqtt
from dotenv import load_dotenv
import os

class MockNotifier:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize MQTT client
        self.mqtt_client = None
        self.mqtt_connected = False
        self._setup_mqtt()
        
        # Alert history
        self.alert_history = []
        
        # Mock hardware status
        self.led_status = "OFF"
        self.buzzer_status = "OFF"
        self.display_message = "No alerts"
        
    def _setup_mqtt(self):
        """Setup MQTT client"""
        try:
            mqtt_broker = os.getenv('MQTT_BROKER', 'localhost')
            mqtt_port = int(os.getenv('MQTT_PORT', 1883))
            mqtt_topic = os.getenv('MQTT_TOPIC', 'blackhole/alerts')
            
            self.mqtt_client = mqtt.Client()
            self.mqtt_client.on_connect = self._on_mqtt_connect
            self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
            
            # Connect to MQTT broker
            self.mqtt_client.connect(mqtt_broker, mqtt_port)
            self.mqtt_client.loop_start()
            
        except Exception as e:
            logging.error(f"Error setting up MQTT: {str(e)}")
            
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            self.mqtt_connected = True
            logging.info("Connected to MQTT broker")
        else:
            logging.error(f"Failed to connect to MQTT broker with code: {rc}")
            
    def _on_mqtt_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback"""
        self.mqtt_connected = False
        logging.warning("Disconnected from MQTT broker")
        
    def trigger_alerts(self, detection):
        """Trigger all alert mechanisms"""
        try:
            # Create alert message
            alert_msg = f"Blackhole detected from {detection['source_ip']} (confidence: {detection['confidence']:.2f})"
            timestamp = datetime.fromtimestamp(detection['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            
            # Update alert history
            self.alert_history.append({
                'timestamp': timestamp,
                'message': alert_msg,
                'source_ip': detection['source_ip'],
                'confidence': detection['confidence']
            })
            
            # Keep only last 100 alerts
            if len(self.alert_history) > 100:
                self.alert_history = self.alert_history[-100:]
            
            # Simulate LED alert
            self.led_status = "BLINKING"
            logging.info("LED would blink")
            
            # Simulate buzzer alert
            self.buzzer_status = "ON"
            logging.info("Buzzer would sound")
            
            # Update OLED display
            self.display_message = alert_msg
            logging.info(f"OLED would display: {alert_msg}")
            
            # Send MQTT alert
            if self.mqtt_connected:
                mqtt_topic = os.getenv('MQTT_TOPIC', 'blackhole/alerts')
                self.mqtt_client.publish(mqtt_topic, alert_msg)
                logging.info(f"MQTT alert sent to topic: {mqtt_topic}")
            
            logging.info(f"Alerts triggered: {alert_msg}")
            
        except Exception as e:
            logging.error(f"Error triggering alerts: {str(e)}")
            
    def get_alert_history(self):
        """Get the alert history"""
        return self.alert_history
        
    def get_hardware_status(self):
        """Get the status of mock hardware"""
        return {
            'led': self.led_status,
            'buzzer': self.buzzer_status,
            'display': self.display_message
        }
        
    def cleanup(self):
        """Cleanup resources"""
        try:
            # Stop MQTT client
            if self.mqtt_client:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
                
        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")

class MockLEDController:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def blink_alert(self):
        self.logger.info("LED Alert: Blinking LED")
        
    def cleanup(self):
        self.logger.info("LED Controller: Cleanup")

class MockBuzzerController:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def trigger_alert(self):
        self.logger.info("Buzzer Alert: Triggering buzzer")
        
    def cleanup(self):
        self.logger.info("Buzzer Controller: Cleanup")

class MockDisplayController:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def show_alert(self, message):
        self.logger.info(f"Display Alert: {message}")
        
    def cleanup(self):
        self.logger.info("Display Controller: Cleanup") 
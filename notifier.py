import logging
import threading
import time
from datetime import datetime
import os
from hardware_controllers import LEDController, BuzzerController, DisplayController

class Notifier:
    def __init__(self, use_hardware=True):
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.use_hardware = use_hardware
        
        try:
            # Initialize hardware controllers
            if use_hardware:
                self.led = LEDController()
                self.buzzer = BuzzerController()
                self.display = DisplayController()
                self.logger.info("Notifier initialized with hardware controllers")
            else:
                from mock_notifier import MockLEDController, MockBuzzerController, MockDisplayController
                self.led = MockLEDController()
                self.buzzer = MockBuzzerController()
                self.display = MockDisplayController()
                self.logger.info("Notifier initialized with mock controllers")
        except Exception as e:
            self.logger.error(f"Error initializing hardware controllers: {str(e)}")
            self.logger.warning("Falling back to mock controllers")
            from mock_notifier import MockLEDController, MockBuzzerController, MockDisplayController
            self.led = MockLEDController()
            self.buzzer = MockBuzzerController()
            self.display = MockDisplayController()
        
        # Alert history
        self.alert_history = []
        
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
            
            # Trigger LED alert
            self.led.blink_alert()
            
            # Trigger buzzer alert
            self.buzzer.trigger_alert()
            
            # Update OLED display
            self.display.show_alert(alert_msg)
                
            self.logger.info(f"Alerts triggered: {alert_msg}")
            
        except Exception as e:
            self.logger.error(f"Error triggering alerts: {str(e)}")
            
    def get_alert_history(self):
        """Get the alert history"""
        return self.alert_history
        
    def cleanup(self):
        """Cleanup resources"""
        try:
            # Cleanup hardware controllers
            self.led.cleanup()
            self.buzzer.cleanup()
            self.display.cleanup()
            self.logger.info("Notifier cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}") 
import logging
import threading
import time
from datetime import datetime
import paho.mqtt.client as mqtt
from dotenv import load_dotenv
import os

class MockNotifier:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def notify_blackhole(self, source_ip, confidence):
        """Log blackhole detection instead of using hardware"""
        self.logger.warning(f"BLACKHOLE DETECTED! Source IP: {source_ip}, Confidence: {confidence:.2f}")
        
    def notify_normal(self, source_ip):
        """Log normal traffic instead of using hardware"""
        self.logger.info(f"Normal traffic from {source_ip}")
        
    def notify_error(self, message):
        """Log error instead of using hardware"""
        self.logger.error(f"Error: {message}")

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
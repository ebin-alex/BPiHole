import logging
import threading
import time
from datetime import datetime
import os
from hardware_controllers import LEDController, BuzzerController, DisplayController
import sys
import busio
import board
import digitalio
from board import SCL, SDA
from PIL import Image, ImageDraw, ImageFont
import adafruit_ssd1306
from adafruit_mcp230xx.mcp23017 import MCP23017

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
        
    def notify_detection(self, detection):
        """Notify about a blackhole detection"""
        try:
            # Add to history
            self.alert_history.append(detection)
            
            # Update display with detection info
            self.display.show_detection(detection)
            
            # If it's a blackhole, trigger alerts
            if detection.get('is_blackhole', False):
                # Flash LEDs
                self.led.blink_alert()
                
                # Sound buzzer
                self.buzzer.sound_alert(duration=1.0)
                
                self.logger.info(f"Blackhole detected: {detection}")
            else:
                self.logger.debug(f"Normal traffic: {detection}")
                
        except Exception as e:
            self.logger.error(f"Error in notification: {str(e)}")
            
    def cleanup(self):
        """Clean up resources"""
        try:
            self.led.cleanup()
            self.buzzer.cleanup()
            self.display.cleanup()
            self.logger.info("Notifier cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}") 
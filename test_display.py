#!/usr/bin/env python3
# Test script for OLED display

import sys
import time
import logging
from hardware_controllers import DisplayController

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting OLED display test")
    
    # Initialize display controller
    display = DisplayController()
    
    # Create a test detection
    test_detection = {
        'is_blackhole': True,
        'source_ip': '192.168.1.100',
        'destination_ip': '192.168.1.1',
        'confidence': 0.85
    }
    
    # Show the test detection
    logger.info("Displaying test detection")
    display.show_detection(test_detection)
    
    # Wait for 5 seconds
    logger.info("Waiting for 5 seconds")
    time.sleep(5)
    
    # Show normal status
    logger.info("Displaying normal status")
    normal_detection = {
        'is_blackhole': False,
        'source_ip': '192.168.1.101',
        'destination_ip': '192.168.1.2',
        'confidence': 0.15
    }
    display.show_detection(normal_detection)
    
    # Wait for 5 seconds
    logger.info("Waiting for 5 seconds")
    time.sleep(5)
    
    # Clean up
    logger.info("Cleaning up")
    display.cleanup()
    
    logger.info("Test completed")

if __name__ == "__main__":
    main() 
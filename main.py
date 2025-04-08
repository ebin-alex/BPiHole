import logging
import time
from datetime import datetime
import threading
from queue import Queue
import os
import signal
import sys

from packet_sniffer import PacketSniffer
from feature_extractor import FeatureExtractor
from model_infer import ModelInfer
from notifier import Notifier
from dashboard import Dashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('blackhole_detector.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class BlackholeDetector:
    def __init__(self):
        self.packet_queue = Queue()
        self.feature_queue = Queue()
        self.detection_queue = Queue()
        self.running = False
        
        # Initialize components
        self.sniffer = PacketSniffer(self.packet_queue)
        self.extractor = FeatureExtractor(self.packet_queue, self.feature_queue)
        self.model = ModelInfer(self.feature_queue, self.detection_queue)
        self.notifier = Notifier()
        self.dashboard = Dashboard()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("Received shutdown signal")
        self.running = False
        
    def start(self):
        try:
            logger.info("Starting Blackhole Detection System...")
            self.running = True
            
            # Start the dashboard
            dashboard_thread = threading.Thread(target=self.dashboard.run)
            dashboard_thread.daemon = True
            dashboard_thread.start()
            
            # Start packet capture
            self.sniffer.start()
            
            # Start feature extraction
            self.extractor.start_processing()
            
            # Start model inference
            self.model.start()
            
            # Main detection loop
            detection_count = 0
            while self.running:
                if not self.detection_queue.empty():
                    detection = self.detection_queue.get()
                    detection_count += 1
                    
                    # Send detection to dashboard
                    self.dashboard.add_detection(detection)
                    
                    if detection['is_blackhole']:
                        logger.warning(f"Blackhole activity detected from node {detection['source_ip']}")
                        self.notifier.trigger_alerts(detection)
                    
                    # Log progress
                    if detection_count % 10 == 0:
                        logger.info(f"Processed {detection_count} detections")
                
                time.sleep(0.1)  # Prevent CPU overload
                
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            self.cleanup()
            
    def cleanup(self):
        logger.info("Cleaning up resources...")
        self.running = False
        self.sniffer.stop()
        self.extractor.stop()
        self.model.stop()
        self.notifier.cleanup()
        self.dashboard.stop()
        
        # Force exit after a short delay to ensure cleanup
        threading.Timer(1.0, lambda: os._exit(0)).start()

if __name__ == "__main__":
    detector = BlackholeDetector()
    detector.start() 
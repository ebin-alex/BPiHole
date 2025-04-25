import logging
import time
from datetime import datetime
import threading
from queue import Queue
import os
import signal
import sys

from mock_packet_sniffer import MockPacketSniffer as PacketSniffer
from feature_extractor import FeatureExtractor
from model_infer import ModelInference
from mock_notifier import MockNotifier as Notifier
from dashboard import Dashboard

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class BlackholeDetector:
    def __init__(self):
        self.packet_queue = Queue()
        self.feature_queue = Queue()
        self.detection_queue = Queue()
        self.running = False
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.sniffer = PacketSniffer(self.packet_queue)
        self.extractor = FeatureExtractor(self.packet_queue, self.feature_queue)
        self.model = ModelInference(self.feature_queue, self.detection_queue)
        self.notifier = Notifier()
        self.dashboard = Dashboard(host='0.0.0.0', port=5001)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info("Received shutdown signal")
        self.running = False
        
    def start(self):
        """Start the blackhole detection system"""
        try:
            self.running = True
            self.logger.info("Starting Blackhole Detection System...")
            
            # Start dashboard
            self.dashboard.start()
            self.logger.info("Dashboard started")
            
            # Start packet capture
            self.sniffer.start()
            self.logger.info("Packet sniffer started")
            
            # Start feature extraction
            self.extractor.start()
            self.logger.info("Feature extractor started")
            
            # Start model inference
            self.model.start()
            self.logger.info("Model inference started")
            
            # Main loop
            while self.running:
                try:
                    if not self.detection_queue.empty():
                        detection = self.detection_queue.get()
                        
                        # Update dashboard stats for all traffic
                        self.dashboard.stats['total_packets'] += 1
                        self.dashboard.stats['last_update'] = int(time.time())
                        
                        if detection and detection.get('is_blackhole', False):
                            self.notifier.notify_blackhole(
                                detection['source_ip'],
                                detection['confidence']
                            )
                            # Update dashboard with the blackhole detection
                            self.dashboard.add_detection({
                                'timestamp': int(time.time()),
                                'source_ip': detection['source_ip'],
                                'destination_ip': detection.get('destination_ip', 'Unknown'),
                                'confidence': detection['confidence'],
                                'is_blackhole': True
                            })
                            # Update blackhole stats
                            self.dashboard.stats['blackhole_detections'] += 1
                        else:
                            # Update normal traffic stats
                            self.dashboard.stats['normal_packets'] += 1
                            # Add normal traffic to dashboard
                            self.dashboard.add_detection({
                                'timestamp': int(time.time()),
                                'source_ip': detection['source_ip'],
                                'destination_ip': detection.get('destination_ip', 'Unknown'),
                                'confidence': detection['confidence'],
                                'is_blackhole': False
                            })
                            
                    time.sleep(0.1)
                except Exception as e:
                    self.logger.error(f"Error in main loop: {str(e)}")
                    
        except Exception as e:
            self.logger.error(f"Error starting system: {str(e)}")
        finally:
            self.stop()
            
    def stop(self):
        """Stop the blackhole detection system"""
        self.running = False
        self.logger.info("Stopping system...")
        
        # Stop components
        if hasattr(self, 'sniffer'):
            self.sniffer.stop()
        if hasattr(self, 'extractor'):
            self.extractor.stop()
        if hasattr(self, 'model'):
            self.model.stop()
        if hasattr(self, 'dashboard'):
            self.dashboard.stop()
            
        self.logger.info("System stopped")

if __name__ == "__main__":
    detector = BlackholeDetector()
    try:
        detector.start()
    except KeyboardInterrupt:
        detector.stop() 
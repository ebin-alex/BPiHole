import logging
import time
from datetime import datetime
import threading
from queue import Queue
import os
import argparse
from scapy.all import rdpcap, IP

from feature_extractor import FeatureExtractor
from model_infer import ModelInfer
from mock_notifier import MockNotifier
from dashboard import Dashboard

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG level
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('blackhole_detector_test.log'),
        logging.StreamHandler()
    ]
)

class BlackholeDetectorTester:
    def __init__(self, pcap_file):
        self.pcap_file = pcap_file
        self.packet_queue = Queue()
        self.feature_queue = Queue()
        self.detection_queue = Queue()
        
        # Initialize components
        self.extractor = FeatureExtractor(self.packet_queue, self.feature_queue)
        self.model = ModelInfer(self.feature_queue, self.detection_queue)
        
        try:
            self.notifier = MockNotifier()
        except Exception as e:
            logging.warning(f"Failed to initialize MQTT notifier: {str(e)}. Notifications will be disabled.")
            self.notifier = None
            
        self.dashboard = Dashboard()
        
        # Start dashboard in a separate thread
        self.dashboard_thread = threading.Thread(target=self.dashboard.run)
        self.dashboard_thread.daemon = True
        
    def load_pcap(self):
        """Load packets from PCAP file"""
        try:
            logging.info(f"Loading PCAP file: {self.pcap_file}")
            packets = rdpcap(self.pcap_file)
            logging.info(f"Loaded {len(packets)} packets from PCAP file")
            return packets
        except Exception as e:
            logging.error(f"Error loading PCAP file: {str(e)}")
            return []
            
    def process_packets(self, packets):
        """Process packets from PCAP file"""
        logging.debug("Starting packet processing...")
        for i, packet in enumerate(packets):
            if IP in packet:
                packet_data = {
                    'timestamp': time.time() + i * 0.001,  # Simulate time progression
                    'source_ip': packet[IP].src,
                    'dest_ip': packet[IP].dst,
                    'protocol': packet[IP].proto,
                    'length': len(packet),
                    'ttl': packet[IP].ttl,
                    'raw_packet': bytes(packet)
                }
                self.packet_queue.put(packet_data)
                logging.debug(f"Queued packet {i+1}/{len(packets)} from {packet_data['source_ip']} to {packet_data['dest_ip']}")
                
                # Simulate real-time processing
                time.sleep(0.001)  # 1ms delay between packets
                
    def start(self):
        try:
            logging.info("Starting Blackhole Detection System Test...")
            
            # Start the dashboard
            self.dashboard_thread.start()
            logging.info("Dashboard started")
            
            # Start feature extraction and model inference threads
            threading.Thread(target=self.extractor.run).start()
            threading.Thread(target=self.model.run).start()
            logging.info("Feature extractor and model inference threads started")
            
            # Load and process PCAP file
            packets = self.load_pcap()
            if packets:
                self.process_packets(packets)
                
            # Main detection loop
            while True:
                if not self.detection_queue.empty():
                    detection = self.detection_queue.get()
                    logging.debug(f"Received detection: {detection}")
                    
                    # Modified detection logic to be more sensitive
                    if detection.get('is_blackhole', False) or detection.get('confidence', 0) > 0.3:
                        logging.warning(f"Blackhole activity detected from node {detection['source_ip']}")
                        if self.notifier:
                            self.notifier.trigger_alerts(detection)
                        self.dashboard.add_detection(detection)
                        logging.info(f"Added detection to dashboard: {detection}")
                    else:
                        logging.debug(f"No blackhole detected: {detection}")
                
                time.sleep(0.1)  # Prevent CPU overload
                
        except KeyboardInterrupt:
            logging.info("Shutting down Blackhole Detection System Test...")
            self.cleanup()
        except Exception as e:
            logging.error(f"Error in main loop: {str(e)}")
            self.cleanup()
            
    def cleanup(self):
        """Cleanup resources"""
        self.extractor.stop()
        self.model.stop()
        if self.notifier:
            self.notifier.cleanup()

def main():
    parser = argparse.ArgumentParser(description='Test the Blackhole Detection System with a PCAP file')
    parser.add_argument('--pcap', type=str, required=True, help='Path to the PCAP file')
    args = parser.parse_args()
    
    tester = BlackholeDetectorTester(args.pcap)
    tester.start()

if __name__ == "__main__":
    main() 
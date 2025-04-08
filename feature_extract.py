import logging
import threading
import time
from collections import defaultdict
from datetime import datetime

class FeatureExtractor(threading.Thread):
    def __init__(self, packet_queue, feature_queue):
        super().__init__()
        self.packet_queue = packet_queue
        self.feature_queue = feature_queue
        self.running = False
        
        # Track packet counts and timestamps
        self.flow_stats = defaultdict(lambda: {
            'packet_count': 0,
            'start_time': None,
            'last_time': None
        })
        
    def extract_features(self, flow_key, stats):
        """Extract features from flow statistics"""
        if stats['start_time'] is None or stats['last_time'] is None:
            return None
            
        duration = (stats['last_time'] - stats['start_time']).total_seconds()
        if duration == 0:
            return None
            
        packet_rate = stats['packet_count'] / duration
        
        source_ip, dest_ip = flow_key
        
        features = {
            'source_ip': source_ip,
            'dest_ip': dest_ip,
            'packet_rate': packet_rate,
            'duration': duration,
            'timestamp': stats['last_time']
        }
        
        logging.debug(f"Extracted features for flow {flow_key}: {features}")
        return features
        
    def run(self):
        """Main processing loop"""
        self.running = True
        logging.info("Feature extraction started")
        
        while self.running:
            try:
                if not self.packet_queue.empty():
                    packet = self.packet_queue.get()
                    logging.debug(f"Processing packet: {packet}")
                    
                    # Extract flow key (source IP, destination IP)
                    flow_key = (packet['source_ip'], packet['dest_ip'])
                    
                    # Update flow statistics
                    stats = self.flow_stats[flow_key]
                    if stats['start_time'] is None:
                        stats['start_time'] = packet['timestamp']
                    stats['last_time'] = packet['timestamp']
                    stats['packet_count'] += 1
                    
                    # Extract features if we have enough data
                    if stats['packet_count'] >= 10:  # Minimum packets for feature extraction
                        features = self.extract_features(flow_key, stats)
                        if features:
                            self.feature_queue.put(features)
                            logging.debug(f"Queued features: {features}")
                            
                else:
                    time.sleep(0.1)  # Prevent CPU overload
                    
            except Exception as e:
                logging.error(f"Error in feature extraction: {str(e)}")
                
    def stop(self):
        """Stop the feature extraction thread"""
        if self.running:
            self.running = False
            if self.is_alive():
                self.join(timeout=1.0)
            logging.info("Feature extraction stopped") 
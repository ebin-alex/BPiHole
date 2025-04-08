import logging
import threading
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import os
import pickle
import pandas as pd
import time

class FeatureExtractor(threading.Thread):
    def __init__(self, packet_queue, feature_queue):
        super().__init__()
        self.packet_queue = packet_queue
        self.feature_queue = feature_queue
        self.running = False
        self.flow_stats = {}
        self.logger = logging.getLogger(__name__)
        self.packet_count = 0
        
    def extract_features(self, packet_data):
        """Extract features from a packet for blackhole detection"""
        source_ip = packet_data['source_ip']
        dest_ip = packet_data['dest_ip']
        flow_key = f"{source_ip}->{dest_ip}"
        
        # Initialize flow stats if not exists
        if flow_key not in self.flow_stats:
            self.flow_stats[flow_key] = {
                'packet_count': 0,
                'byte_count': 0,
                'start_time': packet_data['timestamp'],
                'last_time': packet_data['timestamp'],
                'ttl_values': [],
                'packet_sizes': [],
                'protocols': set()
            }
            
        stats = self.flow_stats[flow_key]
        stats['packet_count'] += 1
        stats['byte_count'] += packet_data['length']
        stats['last_time'] = packet_data['timestamp']
        stats['ttl_values'].append(packet_data['ttl'])
        stats['packet_sizes'].append(packet_data['length'])
        stats['protocols'].add(packet_data['protocol'])
        
        # Calculate features
        duration = stats['last_time'] - stats['start_time']
        if duration > 0:
            packet_rate = stats['packet_count'] / duration
            byte_rate = stats['byte_count'] / duration
        else:
            packet_rate = 0
            byte_rate = 0
            
        features = {
            'source_ip': source_ip,
            'dest_ip': dest_ip,
            'packet_count': stats['packet_count'],
            'byte_count': stats['byte_count'],
            'duration': duration,
            'packet_rate': packet_rate,
            'byte_rate': byte_rate,
            'avg_ttl': np.mean(stats['ttl_values']),
            'std_ttl': np.std(stats['ttl_values']) if len(stats['ttl_values']) > 1 else 0,
            'avg_size': np.mean(stats['packet_sizes']),
            'std_size': np.std(stats['packet_sizes']) if len(stats['packet_sizes']) > 1 else 0,
            'protocol_count': len(stats['protocols']),
            'timestamp': packet_data['timestamp']
        }
        
        return features
        
    def run(self):
        """Main processing loop"""
        self.running = True
        self.logger.info("Feature extractor started")
        
        while self.running:
            try:
                if not self.packet_queue.empty():
                    packet_data = self.packet_queue.get()
                    self.packet_count += 1
                    
                    # Extract features
                    features = self.extract_features(packet_data)
                    
                    # Send features to the model
                    self.feature_queue.put(features)
                    
                    # Log progress
                    if self.packet_count % 10 == 0:
                        self.logger.info(f"Processed {self.packet_count} packets")
                else:
                    time.sleep(0.1)  # Prevent CPU overload
                    
            except Exception as e:
                self.logger.error(f"Error processing packet: {str(e)}")
                
    def stop(self):
        """Stop the feature extraction thread"""
        if self.running:
            self.running = False
            if self.is_alive():
                self.join(timeout=1.0)
            self.logger.info(f"Feature extractor stopped. Total packets processed: {self.packet_count}")
            
    def start_processing(self):
        """Start the feature extraction thread"""
        self.logger.info("Starting feature extractor...")
        super().start()

    def _load_or_create_scaler(self):
        """Load existing scaler or create a new one"""
        scaler_path = 'feature_scaler.pkl'
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logging.error(f"Error loading scaler: {str(e)}")
        
        # Create a new scaler if none exists
        return StandardScaler()
        
    def process_packet(self, packet_data):
        """Process a single packet and update feature storage"""
        source_ip = packet_data['source_ip']
        current_time = packet_data['timestamp']
        
        # Update feature storage
        self.packet_counts[source_ip] += 1
        self.packet_sizes[source_ip].append(packet_data['length'])
        
        # Update protocol counts
        self.protocol_counts[source_ip][packet_data['protocol']] += 1
        
        # Update TTL values
        self.ttl_values[source_ip].append(packet_data['ttl'])
        
        # Update inter-arrival times
        if self.last_packet_time[source_ip] > 0:
            inter_arrival = current_time - self.last_packet_time[source_ip]
            self.inter_arrival_times[source_ip].append(inter_arrival)
        self.last_packet_time[source_ip] = current_time
        
        # Update last seen time
        self.last_seen[source_ip] = current_time
        
        # Extract features if enough packets are collected
        if self.packet_counts[source_ip] >= self.min_packets:
            features = self.extract_features(packet_data)
            self.feature_queue.put(features)
            
    def _run_extractor(self):
        """Run the feature extractor"""
        while self.running:
            try:
                if not self.packet_queue.empty():
                    packet_data = self.packet_queue.get()
                    self.process_packet(packet_data)
            except Exception as e:
                logging.error(f"Error in feature extractor: {str(e)}")
                
    def stop(self):
        """Stop the feature extractor"""
        self.running = False
        self.join(timeout=1.0)
        logging.info("Feature extractor stopped") 
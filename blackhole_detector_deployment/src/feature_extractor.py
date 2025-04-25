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
        self.flow_stats = defaultdict(lambda: {
            'packets': [],
            'start_time': None,
            'last_time': None,
            'total_packets': 0,
            'total_bytes': 0,
            'dao_count': 0,
            'dis_count': 0,
            'dio_count': 0
        })
        self.flow_history = {}  # Initialize flow_history
        self.logger = logging.getLogger(__name__)
        self.packet_count = 0
        
        # Define feature names for consistency
        self.feature_names = [
            'transmission_rate_per_1000_ms',
            'reception_rate_per_1000_ms',
            'transmission_count_per_sec',
            'reception_count_per_sec',
            'transmission_total_duration_per_sec',
            'reception_total_duration_per_sec',
            'transmission_reception_ratio',
            'count_ratio',
            'duration_ratio',
            'dao',
            'dis',
            'dio',
            'tx_rx_rate_ratio',
            'tx_rx_count_ratio',
            'tx_rx_duration_ratio'
        ]
        
    def _get_flow_key(self, packet):
        """Generate a unique key for the flow"""
        try:
            src_ip = packet.get('source_ip', 'unknown')
            dst_ip = packet.get('destination_ip', 'unknown')
            return f"{src_ip}_{dst_ip}"
        except Exception as e:
            self.logger.error(f"Error generating flow key: {str(e)}")
            return None
            
    def _update_flow_stats(self, flow_key, packet):
        """Update statistics for a flow"""
        try:
            stats = self.flow_stats[flow_key]
            current_time = datetime.now()
            
            # Initialize start time if not set
            if stats['start_time'] is None:
                stats['start_time'] = current_time
                
            # Update packet counts and sizes
            stats['total_packets'] += 1
            stats['total_bytes'] += len(str(packet))
            stats['last_time'] = current_time
            
            # Update control message counts
            if packet.get('type') == 'DAO':
                stats['dao_count'] += 1
            elif packet.get('type') == 'DIS':
                stats['dis_count'] += 1
            elif packet.get('type') == 'DIO':
                stats['dio_count'] += 1
                
            # Store packet timestamp
            stats['packets'].append(current_time)
            
        except Exception as e:
            self.logger.error(f"Error updating flow stats: {str(e)}")
            
    def _calculate_features(self, flow_key):
        """Calculate features for a flow"""
        try:
            stats = self.flow_stats[flow_key]
            if not stats['packets']:
                return None
                
            # Calculate time window
            current_time = datetime.now()
            window_start = current_time - timedelta(seconds=1)
            
            # Filter packets in the time window
            window_packets = [p for p in stats['packets'] if p['timestamp'] >= window_start]
            
            if not window_packets:
                return None

            # Calculate basic features
            features = {
                'length': stats['total_bytes'] / stats['total_packets'] if stats['total_packets'] > 0 else 0,
                'transmission_rate_per_1000_ms': len([p for p in window_packets if p['direction'] == 'tx']) * 1000,
                'reception_rate_per_1000_ms': len([p for p in window_packets if p['direction'] == 'rx']) * 1000,
                'transmission_average_per_sec': np.mean([p['length'] for p in window_packets if p['direction'] == 'tx']) if any(p['direction'] == 'tx' for p in window_packets) else 0,
                'reception_average_per_sec': np.mean([p['length'] for p in window_packets if p['direction'] == 'rx']) if any(p['direction'] == 'rx' for p in window_packets) else 0,
                'transmission_count_per_sec': len([p for p in window_packets if p['direction'] == 'tx']),
                'reception_count_per_sec': len([p for p in window_packets if p['direction'] == 'rx']),
                'transmission_total_duration_per_sec': sum(p['duration'] for p in window_packets if p['direction'] == 'tx'),
                'reception_total_duration_per_sec': sum(p['duration'] for p in window_packets if p['direction'] == 'rx'),
                'dao': stats['dao_count'],
                'dis': stats['dis_count'],
                'dio': stats['dio_count']
            }
            
            # Calculate ratio features
            tx_rate = features['transmission_rate_per_1000_ms']
            rx_rate = features['reception_rate_per_1000_ms']
            tx_count = features['transmission_count_per_sec']
            rx_count = features['reception_count_per_sec']
            tx_duration = features['transmission_total_duration_per_sec']
            rx_duration = features['reception_total_duration_per_sec']

            # Add ratio features with protection against division by zero
            features['tx_rx_rate_ratio'] = tx_rate / rx_rate if rx_rate != 0 else 0
            features['tx_rx_count_ratio'] = tx_count / rx_count if rx_count != 0 else 0
            features['tx_rx_duration_ratio'] = tx_duration / rx_duration if rx_duration != 0 else 0
            
            # Ensure all feature names are present
            for name in self.feature_names:
                if name not in features:
                    features[name] = 0
                    
            return features
            
        except Exception as e:
            self.logger.error(f"Error calculating features: {str(e)}")
            return None
            
    def extract_features(self, packet):
        """Extract features from a packet"""
        try:
            # Get source and destination IPs
            source_ip = packet.get('source_ip', '')
            destination_ip = packet.get('destination_ip', '')
            
            # Initialize flow history if not exists
            if (source_ip, destination_ip) not in self.flow_history:
                self.flow_history[(source_ip, destination_ip)] = {
                    'transmission_rate_per_1000_ms': 0,
                    'reception_rate_per_1000_ms': 0,
                    'transmission_count_per_sec': 0,
                    'reception_count_per_sec': 0,
                    'transmission_total_duration_per_sec': 0,
                    'reception_total_duration_per_sec': 0,
                    'dao': 0,
                    'dis': 0,
                    'dio': 0,
                    'length': 0
                }
            
            # Update flow statistics
            flow_stats = self.flow_history[(source_ip, destination_ip)]
            if packet.get('direction') == 'transmission':
                flow_stats['transmission_rate_per_1000_ms'] += 1
                flow_stats['transmission_count_per_sec'] += 1
                flow_stats['transmission_total_duration_per_sec'] += packet.get('duration', 0)
            else:
                flow_stats['reception_rate_per_1000_ms'] += 1
                flow_stats['reception_count_per_sec'] += 1
                flow_stats['reception_total_duration_per_sec'] += packet.get('duration', 0)
            
            # Update RPL control message counts
            if packet.get('protocol') == 'DAO':
                flow_stats['dao'] += 1
            elif packet.get('protocol') == 'DIS':
                flow_stats['dis'] += 1
            elif packet.get('protocol') == 'DIO':
                flow_stats['dio'] += 1
            
            # Update packet length
            flow_stats['length'] = packet.get('length', 0)
            
            # Calculate ratio features
            features = {
                'source_ip': source_ip,
                'destination_ip': destination_ip,
                'transmission_rate_per_1000_ms': flow_stats['transmission_rate_per_1000_ms'],
                'reception_rate_per_1000_ms': flow_stats['reception_rate_per_1000_ms'],
                'transmission_count_per_sec': flow_stats['transmission_count_per_sec'],
                'reception_count_per_sec': flow_stats['reception_count_per_sec'],
                'transmission_total_duration_per_sec': flow_stats['transmission_total_duration_per_sec'],
                'reception_total_duration_per_sec': flow_stats['reception_total_duration_per_sec'],
                'dao': flow_stats['dao'],
                'dis': flow_stats['dis'],
                'dio': flow_stats['dio'],
                'length': flow_stats['length']
            }
            
            # Calculate ratio features
            features['transmission_reception_ratio'] = (
                features['transmission_rate_per_1000_ms'] / 
                max(features['reception_rate_per_1000_ms'], 1)  # Avoid division by zero
            )
            features['count_ratio'] = (
                features['transmission_count_per_sec'] / 
                max(features['reception_count_per_sec'], 1)  # Avoid division by zero
            )
            features['duration_ratio'] = (
                features['transmission_total_duration_per_sec'] / 
                max(features['reception_total_duration_per_sec'], 1)  # Avoid division by zero
            )
            
            self.logger.debug(f"Extracted features: {features}")
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {str(e)}")
            return None
            
    def clear_flow_stats(self, flow_key=None):
        """Clear statistics for a flow or all flows"""
        try:
            if flow_key:
                if flow_key in self.flow_stats:
                    del self.flow_stats[flow_key]
            else:
                self.flow_stats.clear()
        except Exception as e:
            self.logger.error(f"Error clearing flow stats: {str(e)}")
        
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
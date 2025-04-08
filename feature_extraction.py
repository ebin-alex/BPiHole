import logging
import numpy as np
from datetime import datetime, timedelta

class FeatureExtractor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.packet_history = {}  # Store packet history for each source IP
        
    def extract_features(self, packet):
        """Extract features from a packet"""
        try:
            src_ip = packet['src_ip']
            timestamp = packet['timestamp']
            
            # Initialize packet history for new source IP
            if src_ip not in self.packet_history:
                self.packet_history[src_ip] = []
                
            # Add packet to history
            self.packet_history[src_ip].append({
                'timestamp': timestamp,
                'length': packet['length'],
                'protocol': packet['protocol']
            })
            
            # Clean old packets (older than 1 hour)
            self._clean_old_packets(src_ip)
            
            # Extract features
            features = {
                'packet_count': len(self.packet_history[src_ip]),
                'avg_packet_length': self._calculate_avg_length(src_ip),
                'protocol_ratio': self._calculate_protocol_ratio(src_ip),
                'packet_rate': self._calculate_packet_rate(src_ip),
                'burst_count': self._calculate_burst_count(src_ip)
            }
            
            self.logger.debug(f"Extracted features for {src_ip}: {features}")
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {str(e)}")
            return None
            
    def _clean_old_packets(self, src_ip):
        """Remove packets older than 1 hour"""
        current_time = datetime.now()
        self.packet_history[src_ip] = [
            p for p in self.packet_history[src_ip]
            if current_time - p['timestamp'] < timedelta(hours=1)
        ]
        
    def _calculate_avg_length(self, src_ip):
        """Calculate average packet length"""
        packets = self.packet_history[src_ip]
        if not packets:
            return 0
        return sum(p['length'] for p in packets) / len(packets)
        
    def _calculate_protocol_ratio(self, src_ip):
        """Calculate ratio of TCP to UDP packets"""
        packets = self.packet_history[src_ip]
        if not packets:
            return 0
        tcp_count = sum(1 for p in packets if p['protocol'] == 'TCP')
        return tcp_count / len(packets)
        
    def _calculate_packet_rate(self, src_ip):
        """Calculate packets per second"""
        packets = self.packet_history[src_ip]
        if not packets:
            return 0
        time_span = (packets[-1]['timestamp'] - packets[0]['timestamp']).total_seconds()
        if time_span == 0:
            return 0
        return len(packets) / time_span
        
    def _calculate_burst_count(self, src_ip):
        """Calculate number of packet bursts"""
        packets = self.packet_history[src_ip]
        if not packets:
            return 0
            
        burst_count = 0
        in_burst = False
        burst_threshold = 0.1  # 100ms threshold for burst
        
        for i in range(1, len(packets)):
            time_diff = (packets[i]['timestamp'] - packets[i-1]['timestamp']).total_seconds()
            if time_diff < burst_threshold:
                if not in_burst:
                    burst_count += 1
                    in_burst = True
            else:
                in_burst = False
                
        return burst_count 
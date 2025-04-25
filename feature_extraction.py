import logging
import numpy as np
from datetime import datetime, timedelta
import time

class FeatureExtractor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.flow_history = {}  # Store packet history for each src-dst pair
        self.cleanup_interval = 3600  # Clean up flows older than 1 hour
        self.last_cleanup = time.time()
        
    def _get_flow_key(self, src_ip, dst_ip):
        """Create a unique key for each flow"""
        return f"{src_ip}->{dst_ip}"
        
    def extract_features(self, packet):
        """Extract features from a packet"""
        try:
            src_ip = packet['source_ip']
            dst_ip = packet['dest_ip']
            flow_key = self._get_flow_key(src_ip, dst_ip)
            current_time = time.time()
            
            # Periodic cleanup of old flows
            if current_time - self.last_cleanup > 300:  # Every 5 minutes
                self._cleanup_old_flows()
                self.last_cleanup = current_time
            
            # Initialize flow history
            if flow_key not in self.flow_history:
                self.flow_history[flow_key] = {
                    'packets': [],
                    'start_time': current_time,
                    'dao_count': 0,
                    'dis_count': 0,
                    'dio_count': 0
                }
                
            # Add packet to history and update RPL counters
            packet_info = {
                'timestamp': current_time,
                'length': packet['length'],
                'protocol': packet['protocol']
            }
            
            # Update RPL counters if present
            if packet.get('icmpv6_type') == 155:  # RPL Control Message
                if packet.get('rpl_type') == 1:  # DAO
                    self.flow_history[flow_key]['dao_count'] += 1
                elif packet.get('rpl_type') == 0:  # DIS
                    self.flow_history[flow_key]['dis_count'] += 1
                elif packet.get('rpl_type') == 2:  # DIO
                    self.flow_history[flow_key]['dio_count'] += 1
                    
            self.flow_history[flow_key]['packets'].append(packet_info)
            
            # Extract features
            features = {
                'packet_count': len(self.flow_history[flow_key]['packets']),
                'avg_packet_length': self._calculate_avg_length(flow_key),
                'protocol_ratio': self._calculate_protocol_ratio(flow_key),
                'packet_rate': self._calculate_packet_rate(flow_key),
                'burst_count': self._calculate_burst_count(flow_key),
                'flow_duration': self._calculate_flow_duration(flow_key),
                'bytes_per_second': self._calculate_bytes_per_second(flow_key),
                'packet_size_std': self._calculate_packet_size_std(flow_key),
                'inter_arrival_time_mean': self._calculate_iat_mean(flow_key),
                'dao_count': self.flow_history[flow_key]['dao_count'],
                'dis_count': self.flow_history[flow_key]['dis_count'],
                'dio_count': self.flow_history[flow_key]['dio_count']
            }
            
            self.logger.debug(f"Extracted features for flow {flow_key}: {features}")
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {str(e)}")
            return None
            
    def _cleanup_old_flows(self):
        """Remove flows older than cleanup_interval"""
        current_time = time.time()
        old_flows = []
        
        for flow_key, flow_data in self.flow_history.items():
            if current_time - flow_data['start_time'] > self.cleanup_interval:
                old_flows.append(flow_key)
                
        for flow_key in old_flows:
            del self.flow_history[flow_key]
            
    def _calculate_avg_length(self, flow_key):
        """Calculate average packet length"""
        packets = self.flow_history[flow_key]['packets']
        if not packets:
            return 0
        return sum(p['length'] for p in packets) / len(packets)
        
    def _calculate_protocol_ratio(self, flow_key):
        """Calculate ratio of TCP to UDP packets"""
        packets = self.flow_history[flow_key]['packets']
        if not packets:
            return 0
        tcp_count = sum(1 for p in packets if p['protocol'] == 'TCP')
        return tcp_count / len(packets)
        
    def _calculate_packet_rate(self, flow_key):
        """Calculate packets per second"""
        flow_data = self.flow_history[flow_key]
        packets = flow_data['packets']
        if not packets:
            return 0
        duration = self._calculate_flow_duration(flow_key)
        return len(packets) / max(duration, 1)  # Avoid division by zero
        
    def _calculate_burst_count(self, flow_key):
        """Calculate number of packet bursts using dynamic thresholding"""
        packets = self.flow_history[flow_key]['packets']
        if len(packets) < 3:
            return 0
            
        # Calculate inter-arrival times
        iats = [packets[i]['timestamp'] - packets[i-1]['timestamp'] 
                for i in range(1, len(packets))]
        
        # Dynamic burst threshold based on mean and std of IATs
        iat_mean = np.mean(iats)
        iat_std = np.std(iats)
        burst_threshold = max(0.1, iat_mean - iat_std)  # At least 100ms
        
        burst_count = 0
        in_burst = False
        
        for iat in iats:
            if iat < burst_threshold:
                if not in_burst:
                    burst_count += 1
                    in_burst = True
            else:
                in_burst = False
                
        return burst_count
        
    def _calculate_flow_duration(self, flow_key):
        """Calculate flow duration in seconds"""
        packets = self.flow_history[flow_key]['packets']
        if len(packets) < 2:
            return 0
        return packets[-1]['timestamp'] - packets[0]['timestamp']
        
    def _calculate_bytes_per_second(self, flow_key):
        """Calculate bytes per second"""
        flow_data = self.flow_history[flow_key]
        packets = flow_data['packets']
        if not packets:
            return 0
        total_bytes = sum(p['length'] for p in packets)
        duration = self._calculate_flow_duration(flow_key)
        return total_bytes / max(duration, 1)  # Avoid division by zero
        
    def _calculate_packet_size_std(self, flow_key):
        """Calculate standard deviation of packet sizes"""
        packets = self.flow_history[flow_key]['packets']
        if len(packets) < 2:
            return 0
        sizes = [p['length'] for p in packets]
        return np.std(sizes)
        
    def _calculate_iat_mean(self, flow_key):
        """Calculate mean inter-arrival time"""
        packets = self.flow_history[flow_key]['packets']
        if len(packets) < 2:
            return 0
        iats = [packets[i]['timestamp'] - packets[i-1]['timestamp'] 
                for i in range(1, len(packets))]
        return np.mean(iats) 
import logging
import threading
import time
from scapy.all import sniff, IP
from datetime import datetime

class PacketCapture(threading.Thread):
    def __init__(self, packet_queue, interface=None):
        super().__init__()
        self.packet_queue = packet_queue
        self.interface = interface
        self.running = False
        self.logger = logging.getLogger(__name__)
        self.flow_stats = {}
        
    def process_packet(self, packet_data):
        """Process a captured packet"""
        try:
            # Validate packet data
            if not all(key in packet_data for key in ['source_ip', 'dest_ip', 'timestamp']):
                self.logger.warning(f"Invalid packet data: missing required fields")
                return
                
            src_ip = packet_data['source_ip']
            dst_ip = packet_data['dest_ip']
            
            # Log packet details for debugging
            self.logger.debug(f"Processing packet: {src_ip} -> {dst_ip}")
            
            # Update flow statistics
            flow_key = (src_ip, dst_ip)
            if flow_key not in self.flow_stats:
                self.flow_stats[flow_key] = {
                    'packet_count': 0,
                    'start_time': packet_data['timestamp'],
                    'last_update': packet_data['timestamp']
                }
            
            stats = self.flow_stats[flow_key]
            stats['packet_count'] += 1
            stats['last_update'] = packet_data['timestamp']
            
            # Check for blackhole conditions
            self.check_blackhole_condition(flow_key, stats)
            
        except Exception as e:
            self.logger.error(f"Error processing packet: {str(e)}")
            
    def run(self):
        """Main capture loop"""
        self.running = True
        logging.info(f"Starting packet capture on interface: {self.interface or 'default'}")
        
        try:
            # Start sniffing packets
            sniff(
                iface=self.interface,
                prn=self.process_packet,
                store=False,
                stop_filter=lambda x: not self.running
            )
        except Exception as e:
            logging.error(f"Error in packet capture: {str(e)}")
            
    def stop(self):
        """Stop the packet capture thread"""
        if self.running:
            self.running = False
            if self.is_alive():
                self.join(timeout=1.0)
            logging.info("Packet capture stopped") 
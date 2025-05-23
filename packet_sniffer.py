import logging
import threading
from scapy.all import sniff, IP
from datetime import datetime
import time
import os

class PacketSniffer:
    def __init__(self, packet_queue):
        self.packet_queue = packet_queue
        self.is_running = False
        self.sniffer_thread = None
        self.packet_count = 0
        self.logger = logging.getLogger(__name__)
        
    def packet_callback(self, packet):
        """Process each captured packet"""
        try:
            if IP in packet:
                self.packet_count += 1
                # Ensure we have valid source and destination IPs
                src_ip = packet[IP].src
                dst_ip = packet[IP].dst
                
                if not src_ip or not dst_ip:
                    self.logger.warning(f"Invalid IP addresses in packet: src={src_ip}, dst={dst_ip}")
                    return
                    
                packet_data = {
                    'timestamp': datetime.now().timestamp(),
                    'source_ip': src_ip,
                    'dest_ip': dst_ip,
                    'protocol': packet[IP].proto,
                    'length': len(packet),
                    'ttl': packet[IP].ttl,
                    'raw_packet': bytes(packet)
                }
                self.packet_queue.put(packet_data)
                if self.packet_count % 10 == 0:  # Log every 10 packets
                    self.logger.info(f"Captured {self.packet_count} packets")
        except Exception as e:
            self.logger.error(f"Error processing packet: {str(e)}")
            
    def start(self):
        """Start packet capture in a separate thread"""
        self.is_running = True
        self.sniffer_thread = threading.Thread(target=self._run_sniffer)
        self.sniffer_thread.daemon = True
        self.sniffer_thread.start()
        self.logger.info("Packet sniffer started")
        
    def _run_sniffer(self):
        """Run the packet sniffer"""
        try:
            # Check if running with root privileges
            if os.geteuid() != 0:
                self.logger.error("Packet sniffer must be run with root privileges (sudo)")
                return
                
            # Capture packets on all interfaces
            self.logger.info("Starting packet capture...")
            # Use a filter to capture only IP packets
            sniff(prn=self.packet_callback, store=0, filter="ip")
        except Exception as e:
            self.logger.error(f"Error in packet sniffer: {str(e)}")
            self.is_running = False
            
    def stop(self):
        """Stop the packet sniffer"""
        self.is_running = False
        if self.sniffer_thread:
            self.sniffer_thread.join(timeout=1.0)
        self.logger.info(f"Packet sniffer stopped. Total packets captured: {self.packet_count}") 
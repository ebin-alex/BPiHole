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
        
    def process_packet(self, packet):
        """Process a captured packet"""
        try:
            if IP in packet:
                # Extract packet information
                packet_info = {
                    'source_ip': packet[IP].src,
                    'dest_ip': packet[IP].dst,
                    'timestamp': datetime.now(),
                    'length': len(packet),
                    'protocol': packet[IP].proto
                }
                
                logging.debug(f"Captured packet: {packet_info}")
                self.packet_queue.put(packet_info)
                
        except Exception as e:
            logging.error(f"Error processing packet: {str(e)}")
            
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
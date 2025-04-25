import logging
import threading
import time
import random
from datetime import datetime

class MockPacketSniffer:
    def __init__(self, packet_queue):
        self.packet_queue = packet_queue
        self.running = False
        self.logger = logging.getLogger(__name__)
        
    def start(self):
        """Start the mock packet sniffer"""
        self.running = True
        self.thread = threading.Thread(target=self._generate_packets)
        self.thread.daemon = True
        self.thread.start()
        self.logger.info("Mock packet sniffer started")
        
    def stop(self):
        """Stop the mock packet sniffer"""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        self.logger.info("Mock packet sniffer stopped")
        
    def _generate_packets(self):
        """Generate mock packets"""
        while self.running:
            try:
                # Generate random packet data
                packet = {
                    'source_ip': f"192.168.1.{random.randint(1, 254)}",
                    'destination_ip': f"192.168.1.{random.randint(1, 254)}",
                    'length': random.randint(64, 1500),
                    'protocol': random.choice(['TCP', 'UDP', 'ICMP']),
                    'timestamp': datetime.now(),
                    'direction': random.choice(['transmission', 'reception']),
                    'duration': random.uniform(0.001, 0.1)
                }
                
                # Randomly add RPL control messages
                if random.random() < 0.1:  # 10% chance
                    packet['protocol'] = random.choice(['DAO', 'DIS', 'DIO'])
                
                self.packet_queue.put(packet)
                time.sleep(random.uniform(0.01, 0.1))  # Random delay between packets
                
            except Exception as e:
                self.logger.error(f"Error generating mock packet: {str(e)}")
                time.sleep(1)  # Wait before retrying 
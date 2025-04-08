from scapy.all import *
import random
import time
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pcap_generator.log'),
        logging.StreamHandler()
    ]
)

class PCAPGenerator:
    def __init__(self, output_file, duration=60, blackhole_prob=0.3):
        self.output_file = output_file
        self.duration = duration  # Duration in seconds
        self.blackhole_prob = blackhole_prob
        self.packets = []
        
    def generate_normal_traffic(self, source_ip, dest_ip, num_packets):
        """Generate normal network traffic"""
        for _ in range(num_packets):
            # Random packet size between 64 and 1500 bytes
            size = random.randint(64, 1500)
            
            # Create IP packet
            packet = IP(
                src=source_ip,
                dst=dest_ip,
                ttl=random.randint(32, 64),
                proto=random.choice([6, 17])  # TCP or UDP
            )
            
            # Add random payload
            payload = Raw(load=os.urandom(size - len(packet)))
            packet = packet/payload
            
            self.packets.append(packet)
            
    def generate_blackhole_traffic(self, source_ip, dest_ip, num_packets):
        """Generate traffic pattern indicative of blackhole behavior"""
        # First generate normal traffic
        self.generate_normal_traffic(source_ip, dest_ip, num_packets // 2)
        
        # Then abruptly stop sending packets
        time.sleep(1)  # Simulate blackhole formation
        
        # Generate some retransmission attempts
        for _ in range(num_packets // 4):
            packet = IP(
                src=source_ip,
                dst=dest_ip,
                ttl=random.randint(32, 64),
                proto=6  # TCP for retransmissions
            )
            payload = Raw(load=os.urandom(64))  # Small retransmission packets
            packet = packet/payload
            self.packets.append(packet)
            
    def generate(self):
        """Generate the PCAP file with mixed normal and blackhole traffic"""
        try:
            logging.info(f"Generating PCAP file: {self.output_file}")
            
            # Generate traffic for multiple source-destination pairs
            num_pairs = 5
            for i in range(num_pairs):
                source_ip = f"192.168.1.{10+i}"
                dest_ip = f"192.168.1.{100+i}"
                
                # Determine if this pair will exhibit blackhole behavior
                is_blackhole = random.random() < self.blackhole_prob
                
                # Generate appropriate traffic pattern
                num_packets = random.randint(100, 500)
                if is_blackhole:
                    logging.info(f"Generating blackhole traffic for {source_ip} -> {dest_ip}")
                    self.generate_blackhole_traffic(source_ip, dest_ip, num_packets)
                else:
                    logging.info(f"Generating normal traffic for {source_ip} -> {dest_ip}")
                    self.generate_normal_traffic(source_ip, dest_ip, num_packets)
                    
            # Write packets to PCAP file
            wrpcap(self.output_file, self.packets)
            logging.info(f"Successfully generated PCAP file with {len(self.packets)} packets")
            
        except Exception as e:
            logging.error(f"Error generating PCAP file: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Generate test PCAP files for blackhole detection')
    parser.add_argument('--output', type=str, required=True, help='Output PCAP file path')
    parser.add_argument('--duration', type=int, default=60, help='Duration in seconds')
    parser.add_argument('--blackhole-prob', type=float, default=0.3, help='Probability of blackhole behavior')
    
    args = parser.parse_args()
    
    generator = PCAPGenerator(
        output_file=args.output,
        duration=args.duration,
        blackhole_prob=args.blackhole_prob
    )
    generator.generate()

if __name__ == "__main__":
    main() 
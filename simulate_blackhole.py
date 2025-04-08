import scapy.all as scapy
import time
import random
import argparse
import logging
from scapy.layers.inet import IP, ICMP, TCP, UDP
from scapy.layers.l2 import Ether

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BlackholeSimulator:
    def __init__(self, interface, drop_rate=0.5, target_ip=None):
        """
        Initialize the blackhole simulator
        
        Args:
            interface (str): Network interface to use
            drop_rate (float): Probability of dropping packets (0.0 to 1.0)
            target_ip (str): Specific IP to target for blackhole behavior
        """
        self.interface = interface
        self.drop_rate = drop_rate
        self.target_ip = target_ip
        self.running = False
        
    def start(self):
        """Start the blackhole simulation"""
        logger.info(f"Starting blackhole simulation on interface {self.interface}")
        logger.info(f"Drop rate: {self.drop_rate}, Target IP: {self.target_ip}")
        
        self.running = True
        try:
            # Start packet sniffing
            scapy.sniff(
                iface=self.interface,
                prn=self._process_packet,
                store=0,
                filter=f"ip and not src {self.target_ip}" if self.target_ip else "ip"
            )
        except KeyboardInterrupt:
            logger.info("Stopping blackhole simulation...")
            self.running = False
        except Exception as e:
            logger.error(f"Error in blackhole simulation: {e}")
            self.running = False
            
    def _process_packet(self, packet):
        """Process incoming packets and simulate blackhole behavior"""
        if not self.running:
            return
            
        try:
            # Check if packet should be dropped based on drop rate
            if random.random() < self.drop_rate:
                # Instead of dropping, we'll send a fake response
                if IP in packet:
                    # Create a fake response packet
                    if TCP in packet:
                        fake_response = IP(dst=packet[IP].src)/TCP(
                            sport=packet[TCP].dport,
                            dport=packet[TCP].sport,
                            seq=packet[TCP].ack,
                            ack=packet[TCP].seq + 1,
                            flags='R'  # RST flag to indicate connection reset
                        )
                    elif UDP in packet:
                        fake_response = IP(dst=packet[IP].src)/UDP(
                            sport=packet[UDP].dport,
                            dport=packet[UDP].sport
                        )
                    else:
                        fake_response = IP(dst=packet[IP].src)/ICMP(
                            type=3,  # Destination Unreachable
                            code=1   # Host Unreachable
                        )
                    
                    # Send the fake response
                    scapy.send(fake_response, iface=self.interface, verbose=False)
                    logger.debug(f"Dropped packet from {packet[IP].src} to {packet[IP].dst}")
                    
        except Exception as e:
            logger.error(f"Error processing packet: {e}")

def main():
    parser = argparse.ArgumentParser(description='Simulate a blackhole node in the network')
    parser.add_argument('--interface', required=True, help='Network interface to use')
    parser.add_argument('--drop-rate', type=float, default=0.5, help='Probability of dropping packets (0.0 to 1.0)')
    parser.add_argument('--target-ip', help='Specific IP to target for blackhole behavior')
    
    args = parser.parse_args()
    
    simulator = BlackholeSimulator(
        interface=args.interface,
        drop_rate=args.drop_rate,
        target_ip=args.target_ip
    )
    
    simulator.start()

if __name__ == "__main__":
    main() 
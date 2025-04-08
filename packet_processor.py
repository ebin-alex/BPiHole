import logging
from scapy.all import rdpcap, IP, TCP, UDP
from datetime import datetime

class PacketProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def process_pcap(self, pcap_file):
        """Process packets from a PCAP file"""
        try:
            self.logger.info(f"Processing PCAP file: {pcap_file}")
            packets = rdpcap(pcap_file)
            self.logger.info(f"Loaded {len(packets)} packets from PCAP file")
            
            processed_packets = []
            for packet in packets:
                if IP in packet:
                    processed_packet = self._process_packet(packet)
                    if processed_packet:
                        processed_packets.append(processed_packet)
                        
            self.logger.info(f"Successfully processed {len(processed_packets)} IP packets")
            return processed_packets
            
        except Exception as e:
            self.logger.error(f"Error processing PCAP file: {str(e)}")
            return []
            
    def _process_packet(self, packet):
        """Extract relevant information from a packet"""
        try:
            ip = packet[IP]
            
            # Get basic IP information
            processed = {
                'src_ip': ip.src,
                'dst_ip': ip.dst,
                'timestamp': datetime.fromtimestamp(packet.time),
                'length': len(packet),
                'protocol': 'unknown'
            }
            
            # Get protocol-specific information
            if TCP in packet:
                processed['protocol'] = 'TCP'
                processed['src_port'] = packet[TCP].sport
                processed['dst_port'] = packet[TCP].dport
            elif UDP in packet:
                processed['protocol'] = 'UDP'
                processed['src_port'] = packet[UDP].sport
                processed['dst_port'] = packet[UDP].dport
                
            self.logger.debug(f"Processed packet: {processed}")
            return processed
            
        except Exception as e:
            self.logger.error(f"Error processing packet: {str(e)}")
            return None 
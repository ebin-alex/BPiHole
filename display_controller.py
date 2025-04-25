# Display detection details
self.display.text(f"Source: {detection['source_ip']}", 0, 20)
self.display.text(f"Dest: {detection['destination_ip']}", 0, 30)
self.display.text(f"Conf: {detection['confidence']:.2f}", 0, 40) 
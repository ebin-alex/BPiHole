import sys
sys.path.append('/home/pi/Adafruit-Raspberry-Pi-Python-Code-legacy/Adafruit_CircuitPython_MCP230xx')
import time
import board
import busio
import digitalio
from adafruit_mcp230xx.mcp23017 import MCP23017

# Initialize the I2C bus
i2c = busio.I2C(board.SCL, board.SDA)

# Initialize the MCP23017 chip on the bonnet
mcp = MCP23017(i2c)

# Make a list of all the port A pins (Refer User Manual page no 14)
PortA = []
for pin in range(0, 8):
    PortA.append(mcp.get_pin(pin))

# Make a list of all the port B pins (Refer User Manual page no 14)
PortB = []
for pin in range(8, 16):
    PortB.append(mcp.get_pin(pin - 8))

# Clear all Port A pins
for pin in range(0, 8):
    PortA[pin].value = False

# Clear all Port B pins
for pin in range(0, 8):
    PortB[pin].value = False

# Set all Port A and Port B pins as output pins
for pin in range(0, 8):
    PortA[pin].direction = digitalio.Direction.OUTPUT
for pin in range(0, 8):
    PortB[pin].direction = digitalio.Direction.OUTPUT

# Function to create an LED chase pattern
def led_chase():
    while True:
        # LED 1 sequence
        PortA[0].value = True  # Red
        time.sleep(0.5)
        PortA[0].value = False
        
        PortA[1].value = True  # Green
        time.sleep(0.5)
        PortA[1].value = False
        
        PortA[2].value = True  # Blue
        time.sleep(0.5)
        PortA[2].value = False
        
        PortA[3].value = True  # White
        time.sleep(0.5)
        PortA[3].value = False
        
        # LED 2 sequence
        PortA[4].value = True  # Red
        time.sleep(0.5)
        PortA[4].value = False
        
        PortA[5].value = True  # Green
        time.sleep(0.5)
        PortA[5].value = False
        
        PortA[6].value = True  # Blue
        time.sleep(0.5)
        PortA[6].value = False
        
        PortA[7].value = True  # White
        time.sleep(0.5)
        PortA[7].value = False
        
        # LED 3 sequence
        PortB[0].value = True  # Red
        time.sleep(0.5)
        PortB[0].value = False
        
        PortB[1].value = True  # Green
        time.sleep(0.5)
        PortB[1].value = False
        
        PortB[2].value = True  # Blue
        time.sleep(0.5)
        PortB[2].value = False
        
        PortB[3].value = True  # White
        time.sleep(0.5)
        PortB[3].value = False

# Run the LED chase pattern
try:
    print("Running LED chase pattern")
    led_chase()

except KeyboardInterrupt:
    # Clear all port pins when the script is interrupted
    for pin in range(0, 8):
        PortA[pin].value = False
    for pin in range(0, 8):
        PortB[pin].value = False
    print("Script interrupted and GPIO cleaned up")

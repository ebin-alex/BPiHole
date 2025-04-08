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

# Set the first three pins on Port A as output pins for the LEDs
for pin in range(0, 8):
    PortA[pin].direction = digitalio.Direction.OUTPUT

# Set PortB[0] as output pin
PortB[0].direction = digitalio.Direction.OUTPUT

# Function to turn on all LEDs with different colors
def turn_on_all_leds():
    PortA[0].value = True  # led 1 Red LED
    PortA[5].value = True  # led 2 blue LED
    PortA[7].value = True  # led 3 green LED

# Turn on all LEDs with different colors
try:
    print("Turning on all LEDs with different colors")
    turn_on_all_leds()
    # Keep LEDs on for 5 seconds
    time.sleep(5)
    
    # Turn off all LEDs
    for pin in range(0, 8):
        PortA[pin].value = False
    for pin in range(0, 8):
        PortB[pin].value = False
    print("All LEDs turned off")

except KeyboardInterrupt:
    # Clear all port pins when the script is interrupted
    for pin in range(0, 8):
        PortA[pin].value = False
    for pin in range(0, 8):
        PortB[pin].value = False
    print("Script interrupted and GPIO cleaned up")

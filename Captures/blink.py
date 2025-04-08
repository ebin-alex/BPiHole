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
    PortB.append(mcp.get_pin(pin))

# Clear all Port A pins
for pin in range(0, 8):
    PortA[pin].value = False

# Clear all Port B pins
for pin in range(0, 8):
    PortB[pin].value = False

# Set the first three pins on Port A as output pins for the LEDs
PortA[0].direction = digitalio.Direction.OUTPUT
PortA[1].direction = digitalio.Direction.OUTPUT
PortA[2].direction = digitalio.Direction.OUTPUT
PortA[3].direction = digitalio.Direction.OUTPUT
PortA[4].direction = digitalio.Direction.OUTPUT
PortA[5].direction = digitalio.Direction.OUTPUT
PortA[6].direction = digitalio.Direction.OUTPUT
PortA[7].direction = digitalio.Direction.OUTPUT

PortB[0].direction = digitalio.Direction.OUTPUT

# Function to blink LED in a standard pattern
def blink_standard(led_pin, delay):
    led_pin.value = True
    time.sleep(delay)
    led_pin.value = False
    time.sleep(delay)

# Function to blink LED in a fast pattern
def blink_fast(led_pin, delay):
    led_pin.value = True
    time.sleep(delay / 2)
    led_pin.value = False
    time.sleep(delay / 2)

# Function to blink LED in a slow pattern
def blink_slow(led_pin, delay):
    led_pin.value = True
    time.sleep(delay * 2)
    led_pin.value = False
    time.sleep(delay * 2)

# Blink the LEDs in different styles
try:
    while True:
        # Blink the first LED in a standard pattern
        print("Standard Blink LED 0")
        blink_standard(PortA[0], 1)

        # Blink the second LED in a fast pattern
        print("Fast Blink LED 1")
        blink_fast(PortA[3], 1)

        # Blink the third LED in a slow pattern
        print("Slow Blink LED 2")
        blink_slow(PortA[6], 1)

except KeyboardInterrupt:
    # Clear all port pins when the script is interrupted
    for pin in range(0, 3):
        PortA[pin].value = False
    print("Script interrupted and GPIO cleaned up")

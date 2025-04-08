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
mcp = MCP23017(i2c, address=0x20)

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

# Configuring Switch Pin as Input with pull-up resistor and Buzzer Pin as Output
PortB[2].direction = digitalio.Direction.INPUT  # SWITCH S1
PortB[2].pull = digitalio.Pull.UP  # Enable pull-up resistor for the switch

PortB[3].direction = digitalio.Direction.OUTPUT  # BUZZER

# Function to control the buzzer based on button press
def control_buzzer():
    while True:
        if not PortB[2].value:  # If button is pressed (logic inverted)
            PortB[3].value = True  # Turn on the buzzer
            print("Buzzer ON")
        else:
            PortB[3].value = False  # Turn off the buzzer
            print("Buzzer OFF")
        time.sleep(0.1)  # Small delay to debounce the button

# Run the control buzzer function
try:
    print("Press the button to turn on the buzzer")
    control_buzzer()

except KeyboardInterrupt:
    # Clear all port pins when the script is interrupted
    for pin in range(0, 8):
        PortA[pin].value = False
    for pin in range(0, 8):
        PortB[pin].value = False
    print("Script interrupted and GPIO cleaned up")

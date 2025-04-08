# This program will display a simple message on the OLED screen.
# This OLED (SSD1306) has 128x64 resolution and 1-bit color (i.e., black and white).
# Refer Experiment No.4 in the manual for more details.

import sys
sys.path.append('/home/pi/Adafruit-Raspberry-Pi-Python-Code-legacy/Adafruit_CircuitPython_SSD1306')
import busio
import board
import time
import digitalio
from board import SCL, SDA
from digitalio import Direction, Pull
from PIL import Image, ImageDraw, ImageFont
import adafruit_ssd1306

# Define the reset pin
RESET_PIN = digitalio.DigitalInOut(board.D4)

# Initialize the I2C bus
i2c = board.I2C()

# Initialize the OLED display
oled = adafruit_ssd1306.SSD1306_I2C(128, 64, i2c, addr=0x3C, reset=RESET_PIN)

# Clear the display
oled.fill(0)
oled.show()

# Create a blank image for drawing
image = Image.new("1", (oled.width, oled.height))
draw = ImageDraw.Draw(image)

# Load a font in the same size
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
font2 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)

# Draw text on the image
draw.text((0, 0), "***Hello***", font=font, fill=255)  # Line 1
draw.text((0, 20), "Ebin Alex", font=font2, fill=255)  # Line 2
draw.text((0, 40), "***********", font=font2, fill=255)  # Line 3

# Display the image on the OLED
oled.image(image)
oled.show()

# Print messages to the console
print("Hi")
print("EIS IoT KIT")

# Pause for 10 seconds
time.sleep(10)

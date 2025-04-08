import sys
import time
import board
import busio
import digitalio
from PIL import Image, ImageDraw, ImageFont
import adafruit_ssd1306
from adafruit_mcp230xx.mcp23017 import MCP23017
import logging

class LEDController:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            # Initialize I2C bus
            self.i2c = busio.I2C(board.SCL, board.SDA)
            
            # Initialize MCP23017
            self.mcp = MCP23017(self.i2c)
            
            # Initialize Port A pins for LEDs
            self.leds = []
            for pin in range(8):
                led = self.mcp.get_pin(pin)
                led.direction = digitalio.Direction.OUTPUT
                led.value = False
                self.leds.append(led)
                
            self.logger.info("LED Controller initialized")
        except Exception as e:
            self.logger.error(f"Error initializing LED Controller: {str(e)}")
            
    def blink_alert(self):
        """Blink LEDs in an alert pattern"""
        try:
            # Blink all LEDs in sequence
            for led in self.leds:
                led.value = True
                time.sleep(0.1)
                led.value = False
                
            # Flash all LEDs together
            for _ in range(3):
                for led in self.leds:
                    led.value = True
                time.sleep(0.2)
                for led in self.leds:
                    led.value = False
                time.sleep(0.2)
                
            self.logger.info("LED alert triggered")
        except Exception as e:
            self.logger.error(f"Error in LED alert: {str(e)}")
            
    def cleanup(self):
        """Clean up LED pins"""
        try:
            for led in self.leds:
                led.value = False
            self.logger.info("LED Controller cleanup completed")
        except Exception as e:
            self.logger.error(f"Error in LED cleanup: {str(e)}")

class BuzzerController:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            # Initialize I2C bus
            self.i2c = busio.I2C(board.SCL, board.SDA)
            
            # Initialize MCP23017
            self.mcp = MCP23017(self.i2c, address=0x20)
            
            # Initialize buzzer pin (Port B, pin 3)
            self.buzzer = self.mcp.get_pin(11)  # 8 + 3 for Port B
            self.buzzer.direction = digitalio.Direction.OUTPUT
            self.buzzer.value = False
            
            self.logger.info("Buzzer Controller initialized")
        except Exception as e:
            self.logger.error(f"Error initializing Buzzer Controller: {str(e)}")
            
    def trigger_alert(self):
        """Trigger buzzer alert pattern"""
        try:
            # Short beeps for alert
            for _ in range(3):
                self.buzzer.value = True
                time.sleep(0.1)
                self.buzzer.value = False
                time.sleep(0.1)
                
            self.logger.info("Buzzer alert triggered")
        except Exception as e:
            self.logger.error(f"Error in buzzer alert: {str(e)}")
            
    def cleanup(self):
        """Clean up buzzer pin"""
        try:
            self.buzzer.value = False
            self.logger.info("Buzzer Controller cleanup completed")
        except Exception as e:
            self.logger.error(f"Error in buzzer cleanup: {str(e)}")

class DisplayController:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            # Initialize I2C bus
            self.i2c = board.I2C()
            
            # Initialize reset pin
            self.reset_pin = digitalio.DigitalInOut(board.D4)
            
            # Initialize OLED display
            self.oled = adafruit_ssd1306.SSD1306_I2C(128, 64, self.i2c, addr=0x3C, reset=self.reset_pin)
            
            # Clear the display
            self.oled.fill(0)
            self.oled.show()
            
            # Initialize image and drawing objects
            self.image = Image.new("1", (self.oled.width, self.oled.height))
            self.draw = ImageDraw.Draw(self.image)
            
            # Load fonts
            self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
            self.font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
            
            self.logger.info("Display Controller initialized")
        except Exception as e:
            self.logger.error(f"Error initializing Display Controller: {str(e)}")
            
    def show_alert(self, message):
        """Display alert message on OLED"""
        try:
            # Clear the image
            self.draw.rectangle((0, 0, self.oled.width, self.oled.height), fill=0)
            
            # Draw alert header
            self.draw.text((0, 0), "! ALERT !", font=self.font_bold, fill=255)
            
            # Draw message (word wrap for long messages)
            words = message.split()
            lines = []
            current_line = []
            
            for word in words:
                current_line.append(word)
                test_line = ' '.join(current_line)
                bbox = self.draw.textbbox((0, 0), test_line, font=self.font)
                if bbox[2] > self.oled.width:
                    current_line.pop()
                    lines.append(' '.join(current_line))
                    current_line = [word]
            
            if current_line:
                lines.append(' '.join(current_line))
                
            # Draw lines
            y_position = 20
            for line in lines:
                self.draw.text((0, y_position), line, font=self.font, fill=255)
                y_position += 15
                
            # Show the image
            self.oled.image(self.image)
            self.oled.show()
            
            self.logger.info("Display alert shown")
        except Exception as e:
            self.logger.error(f"Error showing display alert: {str(e)}")
            
    def cleanup(self):
        """Clean up display"""
        try:
            self.oled.fill(0)
            self.oled.show()
            self.logger.info("Display Controller cleanup completed")
        except Exception as e:
            self.logger.error(f"Error in display cleanup: {str(e)}") 
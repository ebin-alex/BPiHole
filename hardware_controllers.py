import sys
import time
import board
import busio
import digitalio
from PIL import Image, ImageDraw, ImageFont
import adafruit_ssd1306
from adafruit_mcp230xx.mcp23017 import MCP23017
import logging

try:
    import board
    import busio
    import digitalio
    from adafruit_mcp230xx.mcp23017 import MCP23017
    from adafruit_ssd1306 import SSD1306_I2C
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False
    print("Hardware libraries not available. Using mock implementations.")

# Define I2C pins for Raspberry Pi
I2C_SCL_PIN = board.SCL
I2C_SDA_PIN = board.SDA

# Global I2C bus instance
_i2c_bus = None

def get_i2c_bus():
    """Get or create I2C bus instance with proper error handling"""
    global _i2c_bus
    if _i2c_bus is None:
        try:
            _i2c_bus = board.I2C()  # Use board.I2C() instead of busio.I2C()
            logging.info("I2C bus initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize I2C bus: {str(e)}")
            _i2c_bus = None
    return _i2c_bus

class LEDController:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        if HARDWARE_AVAILABLE:
            try:
                # Get I2C bus instance
                self.i2c = get_i2c_bus()
                if self.i2c is None:
                    raise Exception("I2C bus initialization failed")
                
                # Initialize MCP23017
                self.mcp = MCP23017(self.i2c)
                
                # Initialize Port A pins for LEDs
                self.leds = [self.mcp.get_pin(i) for i in range(8)]
                for led in self.leds:
                    led.direction = digitalio.Direction.OUTPUT
                    led.value = False
                
                self.logger.info("LED Controller initialized")
            except Exception as e:
                self.logger.error(f"Error initializing LED Controller: {str(e)}")
                self.leds = [MockLED() for _ in range(8)]
        else:
            self.leds = [MockLED() for _ in range(8)]
            
    def set_led(self, index, state):
        if 0 <= index < len(self.leds):
            try:
                self.leds[index].value = state
            except Exception as e:
                self.logger.error(f"Error setting LED {index}: {str(e)}")

    def blink_led(self, index):
        if 0 <= index < len(self.leds):
            try:
                self.leds[index].value = True
                time.sleep(0.5)
                self.leds[index].value = False
            except Exception as e:
                self.logger.error(f"Error blinking LED {index}: {str(e)}")

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
        if HARDWARE_AVAILABLE:
            try:
                # Get I2C bus instance
                self.i2c = get_i2c_bus()
                if self.i2c is None:
                    raise Exception("I2C bus initialization failed")
                
                # Initialize MCP23017
                self.mcp = MCP23017(self.i2c)
                
                # Initialize buzzer pin (Port B, pin 3)
                self.buzzer = self.mcp.get_pin(11)  # 8 + 3 for Port B
                self.buzzer.direction = digitalio.Direction.OUTPUT
                self.buzzer.value = False
                
                self.logger.info("Buzzer Controller initialized")
            except Exception as e:
                self.logger.error(f"Error initializing Buzzer Controller: {str(e)}")
                self.buzzer = MockBuzzer()
        else:
            self.buzzer = MockBuzzer()
            
    def sound_alert(self, duration=1.0):
        try:
            self.buzzer.value = True
            time.sleep(duration)
            self.buzzer.value = False
        except Exception as e:
            self.logger.error(f"Error in sound alert: {str(e)}")

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
        if HARDWARE_AVAILABLE:
            try:
                # Initialize I2C bus directly
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
                
                # Try to load a system font, fall back to default if not available
                try:
                    self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
                except IOError:
                    self.logger.warning("System font not found, using default font")
                    self.font = ImageFont.load_default()
                
                # Initialize display state
                self.status = "NORMAL"
                self.start_time = time.time()
                self.last_detection = None
                
                self.logger.info("Display Controller initialized")
            except Exception as e:
                self.logger.error(f"Error initializing Display Controller: {str(e)}")
                self.oled = None
        else:
            self.oled = None
            
    def show_detection(self, detection):
        """Display blackhole detection information"""
        try:
            if not HARDWARE_AVAILABLE or self.oled is None:
                print(f"Mock Display: {detection}")
                return
                
            # Update status and last detection
            self.status = "BLACKHOLE" if detection.get('is_blackhole', False) else "NORMAL"
            self.last_detection = detection
            
            # Clear the display
            self.oled.fill(0)
            self.draw.rectangle((0, 0, self.oled.width, self.oled.height), fill=0)
            
            # Calculate uptime
            uptime_seconds = int(time.time() - self.start_time)
            hours = uptime_seconds // 3600
            minutes = (uptime_seconds % 3600) // 60
            seconds = uptime_seconds % 60
            uptime_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            # Get current time
            current_time = time.strftime("%H:%M:%S")
            
            # Get source and destination IPs
            source_ip = detection.get('source_ip', 'N/A')
            dest_ip = detection.get('destination_ip', 'N/A')
            
            # Get confidence and create progress bar
            confidence = detection.get('confidence', 0)
            progress_bar = self._create_progress_bar(confidence)
            
            # Draw text on the display
            y_pos = 0
            line_height = 10
            
            # Line 1: Status
            self.draw.text((0, y_pos), f"BPiHole: {self.status}", font=self.font, fill=255)
            y_pos += line_height
            
            # Line 2: Time
            self.draw.text((0, y_pos), f"Time: {current_time}", font=self.font, fill=255)
            y_pos += line_height
            
            # Line 3: Uptime
            self.draw.text((0, y_pos), f"Uptime: {uptime_str}", font=self.font, fill=255)
            y_pos += line_height
            
            # Line 4: Source IP
            self.draw.text((0, y_pos), f"SRC: {source_ip}", font=self.font, fill=255)
            y_pos += line_height
            
            # Line 5: Destination IP
            self.draw.text((0, y_pos), f"DST: {dest_ip}", font=self.font, fill=255)
            y_pos += line_height
            
            # Line 6: Confidence with progress bar
            self.draw.text((0, y_pos), f"Conf: {progress_bar} {int(confidence*100)}%", font=self.font, fill=255)
            
            # Show the image
            self.oled.image(self.image)
            self.oled.show()
            
            self.logger.info("Display updated with detection information")
        except Exception as e:
            self.logger.error(f"Error showing detection: {str(e)}")
            
    def _create_progress_bar(self, value, width=10):
        """Create a text-based progress bar"""
        filled = int(value * width)
        return "█" * filled + "░" * (width - filled)
            
    def cleanup(self):
        """Clean up display"""
        try:
            if HARDWARE_AVAILABLE and self.oled is not None:
                self.oled.fill(0)
                self.oled.show()
            self.logger.info("Display Controller cleanup completed")
        except Exception as e:
            self.logger.error(f"Error in display cleanup: {str(e)}")

# Mock classes for when hardware is not available
class MockLED:
    def __init__(self):
        self.value = False

class MockBuzzer:
    def __init__(self):
        self.value = False

class MockDisplay:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        
    def fill(self, color):
        pass
        
    def text(self, text, x, y, color):
        pass
        
    def show(self):
        pass 
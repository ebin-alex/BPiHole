import logging
from hardware_controllers import LEDController, DisplayController

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_hardware():
    try:
        # Test LED Controller
        logger.info("Testing LED Controller...")
        led_controller = LEDController()
        led_controller.test_leds()
        
        # Test Display Controller
        logger.info("Testing Display Controller...")
        display_controller = DisplayController()
        display_controller.test_display()
        
        logger.info("All hardware tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Hardware test failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    test_hardware() 
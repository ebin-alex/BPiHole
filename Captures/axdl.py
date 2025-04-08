# This program will display the acceleration in the X, Y, and Z directions
# Sensor Library: sudo pip3 install adafruit-circuitpython-bmi160
# Reference Link: https://learn.adafruit.com/adafruit-bmi160-6-dof-accelerometer-gyroscope

import sys
sys.path.append('/home/pi/Adafruit-Raspberry-Pi-Python-Code-legacy/Adafruit_CircuitPython_BMI160')
import time
import board
import busio
import bmi160

# Initialize the I2C bus
i2c = board.I2C()

# Initialize the BMI160 sensor
sensor = bmi160.BMI160(i2c, address=0x68)

# Read acceleration values in a loop
while True:
    x, y, z = sensor.acceleration
    print("X-axis={:.2f} m/s^2 Y-axis={:.2f} m/s^2 Z-axis={:.2f} m/s^2".format(x, y, z))
    time.sleep(0.5)

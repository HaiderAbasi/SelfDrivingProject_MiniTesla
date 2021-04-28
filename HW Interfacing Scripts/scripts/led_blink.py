import RPi.GPIO as GPIO          
from time import sleep
GPIO.setmode(GPIO.BCM)
led_pin = 18

GPIO.setup(led_pin,GPIO.OUT)
 
while True:
    GPIO.output(led_pin,GPIO.HIGH)
    sleep(2)
    GPIO.output(led_pin,GPIO.LOW)
    sleep(0.5)
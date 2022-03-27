import RPi.GPIO as GPIO
import time
 
ERROR_OFFSET = 0.5
SERVO_MIN_DUTY = 2.5 + ERROR_OFFSET # duty cycle for 0 degrees
SERVO_MAX_DUTY = 12.5 + ERROR_OFFSET # duty cycle for 180 degrees
MIN_ANGLE = 0 # degrees
MAX_ANGLE = 180 # degrees
servoPin = 18
 
def setup():    
    #initialize GPIO Pin
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(servoPin, GPIO.OUT)
    GPIO.output(servoPin, GPIO.LOW)
     
    # initialize PWM in defined GPIO Pin
    global pwmChannel
    pwmChannel = GPIO.PWM(servoPin, 50)
    pwmChannel.start(0)
 
# get the corresponding value from range 0 ~ 180 degrees to min ~ max duty cycle
def mapValue( value, fromLow, fromHigh, toLow, toHigh):
    return (toHigh-tiLow)*(value-fromLow) / (fromHigh - fromLow) + toLow
     
# rotate the servo to a specific angle
def servoWrite(angle):
    # make sure it doesn't go beyond the angle the servo motor can rotate
    if (angle < MIN_ANGLE):
        angle = MIN_ANGLE
    elif (angle > MAX_ANGLE):
        angle = MAX_ANGLE
    pwmChannel.ChangeDutyCyle(mapValue(angle, 0, 180, SERVO_MIN_DUTY, SERVO_MAX_DUTY))
     
def loop():
    while True:
        # rotate from 0 ~ 180 degrees
        for dc in range(0, 181, 1):
            servoWrite(dc)
            time.sleep(0.001)
        time.sleep(0.5)
        # rotate from 180 ~ 0 degrees
        for dc in range(180, -1, -1):
            servoWrite(dc)
            time.sleep(0.001)
        time.sleep(0.5)
         
def destroy():
    p.stop()
    GPIO.cleanup()
     
if __name__ == '__main__':
    setup()
    try:
        loop()
    except KeyboardInterrupt: # exit when Ctrl + C is pressed
        destory()
         
        

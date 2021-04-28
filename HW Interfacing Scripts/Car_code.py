#!/usr/bin/enable_motorv python3

# This Code controls Dc motor and servo motor through pwm
# We defined all gpio pins in BCM mode and create a switch statment to test all
#DC motor pwm -> 0-100
# servo motor angle -> 15-110 ( 75 is straight , 110 is full right , 15 is full left)
import sys
from time import sleep
import RPi.GPIO as GPIO

def signal_handler(sig, frame):
    GPIO.cleanup()
    sys.exit(0)
#dc_pwm,servo_pwm,servo_motor=0,0,18
#Motors Pins
motor_a = 20 
motor_b = 21
enable_motor = 16
servo_motor=12 
GPIO.setmode(GPIO.BCM)
    #Motors Setup
GPIO.setup(motor_a,GPIO.OUT)
GPIO.setup(motor_b,GPIO.OUT)
GPIO.setup(enable_motor,GPIO.OUT)
GPIO.setup(servo_motor, GPIO.OUT)
    #Pwm setup
dc_pwm=GPIO.PWM(enable_motor,1000)
dc_pwm.start(0)
servo_pwm=GPIO.PWM(servo_motor, 50)
servo_pwm.start(0)


def setServoAngle(angle):
    duty = angle / 18 + 2
    GPIO.output(servo_motor, True)
    servo_pwm.ChangeDutyCycle(duty)
    sleep(1)
    GPIO.output(servo_motor, False)
    servo_pwm.ChangeDutyCycle(duty)
#looping function
    
def loop():

    while True:

        #setting x= o to not recursively act on conditions
        x=input("Which motor \nf-forward\nb-backwar\ns-stop\nz-speed\ne-exit\nt-servo\n\n")
        if  x=='f':
            print("\nForward\n")
            GPIO.output(motor_a,GPIO.HIGH)
            GPIO.output(motor_b,GPIO.LOW)
            x='o' 
            
            
        elif x=='b':
            print("\nBackward\n")
            GPIO.output(motor_a,GPIO.LOW)
            GPIO.output(motor_b,GPIO.HIGH)
            x='o'
            
        
        elif x=='s':
            print("\nstop\n")
            GPIO.output(motor_a,GPIO.LOW)
            GPIO.output(motor_b,GPIO.LOW)
            x='o'
        
        elif x=='z': ## changing speed by providing duty cycle
            x=int(input("Input Speed 0 - 100\n"))
            dc_pwm.ChangeDutyCycle(x)
            x='o'
            
        elif x=='e':
            GPIO.cleanup()
            dc_pwm.stop()
            servo_pwm.stop()
            break
        elif x=='t':
            x=int(input("\nInput angle\n"))
            setServoAngle(x)
            x='o'
        else :
            print("Wrong Input !\n\n")
        
if __name__ == '__main__':
    loop()
    




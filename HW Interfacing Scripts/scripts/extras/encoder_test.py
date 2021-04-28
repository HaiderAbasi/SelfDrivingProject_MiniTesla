#!/usr/bin/env python3
import signal
from time import sleep
import sys
import RPi.GPIO as GPIO
int_enc_pin_a = 7
int_enc_pin_b = 8
count =0
motor_a = 24 
motor_b = 23
enable_motor = 25
GPIO.setmode(GPIO.BCM)
GPIO.setup(motor_a,GPIO.OUT)
GPIO.setup(motor_b,GPIO.OUT)
GPIO.setup(enable_motor,GPIO.OUT)
dc_pwm=GPIO.PWM(enable_motor,1000)
dc_pwm.start(0)

def signal_handler(sig, frame):
    GPIO.cleanup()
    sys.exit(0)
    
def callback_a(channel):
    global count
    if GPIO.input(int_enc_pin_b):
        count=count+1
    else :
        count=count-1
def set_forward():
    GPIO.output(motor_a,GPIO.HIGH)
    GPIO.output(motor_b,GPIO.LOW)
def set_stop():
    GPIO.output(motor_a,GPIO.LOW)
    GPIO.output(motor_b,GPIO.LOW)
        
def loop():
    global count
    while(1):
        y=int(input("\npwm value = "))
        if(y==9): ## stop robot and clea gpios
            GPIO.cleanup()
            dc_pwm.stop()
            break
        else:
            dc_pwm.ChangeDutyCycle(y)
            x=int(input("\nType ticks = "))
            set_forward()
            while(count<x-2):{}
            set_stop()
            sleep(1)
            print(count)
            count=0

        
if __name__ == '__main__':
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(int_enc_pin_b, GPIO.IN)
    GPIO.setup(int_enc_pin_a, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.add_event_detect(int_enc_pin_a, GPIO.FALLING,callback=callback_a, bouncetime=100)
    loop()
    signal.signal(signal.SIGINT, signal_handler)
    signal.pause()


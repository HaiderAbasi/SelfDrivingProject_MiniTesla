#!/usr/bin/env python3
import signal
import sys
import RPi.GPIO as GPIO
int_enc_pin_a = 7
int_enc_pin_b = 8
count =0

def signal_handler(sig, frame):
    GPIO.cleanup()
    sys.exit(0)
def callback_a(channel):
    global count
    if GPIO.input(int_enc_pin_b):
        count=count+1
    else :
        count=count-1
    print(count)

        
if __name__ == '__main__':
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(int_enc_pin_b, GPIO.IN)
    GPIO.setup(int_enc_pin_a, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.add_event_detect(int_enc_pin_a, GPIO.FALLING,callback=callback_a, bouncetime=100)
    signal.signal(signal.SIGINT, signal_handler)
    signal.pause()

#!/usr/bin/enable_motorv python3

# This Code controls Dc motor and servo motor through pwm
# We defined all gpio pins in BCM mode and create a switch statment to test all
#DC motor pwm -> 0-100 (direct duty cycle)
# servo motor angle -> 0-90 ( 35 is straight , 65 is full right , 0 is full left)

import sys
from numpy import interp
from time import sleep
import RPi.GPIO as GPIO


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
car_speed=65
dc_pwm.ChangeDutyCycle(car_speed)
## function names are self representing 
def setServoAngle(angle):
    duty = angle / 18 + 2
    GPIO.output(servo_motor, True)
    servo_pwm.ChangeDutyCycle(duty)
    #sleep(1)
    GPIO.output(servo_motor, False)
    servo_pwm.ChangeDutyCycle(duty)
def forward():
    GPIO.output(motor_a,GPIO.HIGH)
    GPIO.output(motor_b,GPIO.LOW)
    print("forward main hun")
def backward():
    GPIO.output(motor_a,GPIO.LOW)
    GPIO.output(motor_b,GPIO.HIGH)
def stop():
    GPIO.output(motor_a,GPIO.LOW)
    GPIO.output(motor_b,GPIO.LOW)
def changePwm(x):
    dc_pwm.ChangeDutyCycle(x)
def turnOfCar():
    GPIO.cleanup()
    dc_pwm.stop()
    servo_pwm.stop()

<<<<<<< HEAD:Control/Motors_control.py
=======
# def beInLane_(Max_Sane_dist,distance,curvature):
#     Max_turn_angle = 90
#     Max_turn_angle_neg = -90
# 
#     CarTurn_angle = 0
# 
#     if( (distance > Max_Sane_dist) or (distance < (-1 * Max_Sane_dist) ) ):
#         # Max sane distance reached ---> Max penalize (Max turn Tires)
#         if(distance > Max_Sane_dist):
#             #Car offseted left --> Turn full wheels right
#             CarTurn_angle = Max_turn_angle + curvature
#         else:
#             #Car Offseted right--> Turn full wheels left
#             CarTurn_angle = Max_turn_angle_neg + curvature
#     else:
#         # Within allowed distance limits for car and lane
#         # Interpolate distance to Angle Range
#         Turn_angle_interpolated = interp(distance,[-Max_Sane_dist,Max_Sane_dist],[-90,90])
#         print("Turn_angle_interpolated = ", Turn_angle_interpolated)
#         CarTurn_angle = Turn_angle_interpolated + curvature
# 
#     # Handle Max Limit [if (greater then either limits) --> set to max limit]
#     if( (CarTurn_angle > (Max_turn_angle-30)) or (CarTurn_angle < ( (-1 *Max_turn_angle)+30 ) ) ):
#         if(CarTurn_angle > (Max_turn_angle-30)):
#             CarTurn_angle = Max_turn_angle
#         else:
#             CarTurn_angle = -Max_turn_angle
# 
#     angle = interp(CarTurn_angle,[-Max_turn_angle,Max_turn_angle],[0,65])
#     if(angle>55):
#         dc_pwm.ChangeDutyCycle(80)
#     else:
#         dc_pwm.ChangeDutyCycle(car_speed)
#     setServoAngle(int(angle))
#     return angle   

>>>>>>> 6a33bcddb63554774ab6a57dbc5a195f82ff7703:LaneDetection/Motors_control.py
def beInLane(Max_Sane_dist,distance,curvature):
    IncreaseTireSpeedInTurns = True
    
    Max_turn_angle = 90
    Max_turn_angle_neg = -90

    CarTurn_angle = 0

    if( (distance > Max_Sane_dist) or (distance < (-1 * Max_Sane_dist) ) ):
        # Max sane distance reached ---> Max penalize (Max turn Tires)
        if(distance > Max_Sane_dist):
            #Car offseted left --> Turn full wheels right
            CarTurn_angle = Max_turn_angle + curvature
        else:
            #Car Offseted right--> Turn full wheels left
            CarTurn_angle = Max_turn_angle_neg + curvature
    else:
        # Within allowed distance limits for car and lane
        # Interpolate distance to Angle Range
        Turn_angle_interpolated = interp(distance,[-Max_Sane_dist,Max_Sane_dist],[-90,90])
        print("Turn_angle_interpolated = ", Turn_angle_interpolated)
        CarTurn_angle = Turn_angle_interpolated + curvature

    # Handle Max Limit [if (greater then either limits) --> set to max limit]
    if( (CarTurn_angle > Max_turn_angle) or (CarTurn_angle < (-1 *Max_turn_angle) ) ):
        if(CarTurn_angle > Max_turn_angle):
            CarTurn_angle = Max_turn_angle
        else:
            CarTurn_angle = -Max_turn_angle

    angle = interp(CarTurn_angle,[-Max_turn_angle,Max_turn_angle],[30,120])

    curr_speed = car_speed.copy()
    if IncreaseTireSpeedInTurns:
<<<<<<< HEAD:Control/Motors_control.py
        if(angle>95):
            car_speed_turn = interp(angle,[95,120],[80,100])
            dc_pwm.ChangeDutyCycle(car_speed_turn)
            curr_speed = car_speed_turn
        elif(angle<55):
            car_speed_turn = interp(angle,[30,55],[100,80])
            dc_pwm.ChangeDutyCycle(car_speed_turn)
            curr_speed = car_speed_turn
=======
        if(95<angle and angle<110):
            dc_pwm.ChangeDutyCycle(80)
        elif(angle>110):
            dc_pwm.ChangeDutyCycle(100)
>>>>>>> 6a33bcddb63554774ab6a57dbc5a195f82ff7703:LaneDetection/Motors_control.py
        else:
            dc_pwm.ChangeDutyCycle(car_speed)
    
    setServoAngle(int(angle))
    return angle , curr_speed
#turnOfCar() // to disconnect all channels of pwm

    
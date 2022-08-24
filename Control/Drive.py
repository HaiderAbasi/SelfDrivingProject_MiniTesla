from Control.Motors_control import forward,backward,setServoAngle,stop,turnOfCar,changePwm,beInLane
import config
import cv2

def Steer(Distance,Curvature,frame , Mode , Tracked_class):    
    if config.Testing:
        if(Distance != -1000 | Curvature != -1000):
            if (config.debugging==False):		
                angle_of_car , current_speed = beInLane(int(frame.shape[1]/4), Distance, Curvature ,Mode ,Tracked_class )
                angle_speed_str = "[ Angle ,Speed ] = [ " + str(int(angle_of_car)) + " , " + str(int(current_speed)) + " ] "
                #cv2.putText(frame_disp,str(angle_of_car),(frame.shape[1]-400,50),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,255),2)
                cv2.putText(frame,str(angle_speed_str),(20,20),cv2.FONT_HERSHEY_DUPLEX,0.4,(0,0,255),1)

    else:
        if(Distance != -1000 | Curvature != -1000):
            if (config.debugging==False):
                beInLane(int(frame.shape[1]/4), Distance,Curvature  , Mode , Tracked_class)       

def Drive_Car(Current_State):
    """Act on extracted information based on the SDC control mechanism

    Args:
        Current_State (List): information extracted from SDC surroundings 
                              E.g. (Information regarding the lane boundaries for lane assist + 
                                    Information regarding the traffic signs for cruise control)
    """    
    [distance, Curvature, frame_disp, Mode, Tracked_class] = Current_State
    Steer(distance, Curvature,frame_disp, Mode, Tracked_class)
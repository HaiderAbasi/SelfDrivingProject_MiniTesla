import cv2
import numpy as np
import os
import config

# ****************************************************  DETECTION ****************************************************

# ****************************************************    LANES   ****************************************************
from Detection.Lanes.Lane_Detection import Detect_Lane

# ****************************************************    SIGNS   ****************************************************
from Detection.Signs.SignDetectionApi import detect_Signs

# ****************************************************   CONTROL  *****************************************************

from Control.special import Drive_Car

# >>>>>>>>>>>>>>>>>> OPTIMIZING CODE # 3 [Threading] (PyImageSearch) <<<<<<<<<<<<<<<<<<<<<<
# Threading Controls
Use_Threading = False
Live_Testing = False
if Use_Threading:
    from imutils.video.pivideostream import PiVideoStream



def main():

    if Use_Threading:
        # start the file video stream thread and allow the buffer to fill
        PI_vs = PiVideoStream().start()

    else:
        if Live_Testing:
            cap = cv2.VideoCapture(0)
        else:
            #cap = cv2.VideoCapture(os.path.abspath("SW Modular Scripts/data/vids/Lane_vid.avi"))
            cap = cv2.VideoCapture(os.path.abspath("SW Modular Scripts/data/vids/signs_forward.mp4"))
    
    waitTime = 100

    while(1):
        if Use_Threading:
            img = PI_vs.read().copy()
        else:
            ret,img = cap.read()
            if not ret:
                break
        # >>>>>>>>>>>>>>>>>>>>>>>> Optimization No 1 [RESIZING]<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        if not Use_Threading:
            img = cv2.resize(img,(320,240))
        #CropHeight = 130
        #minArea = 250
        #CropHeight = 260
        #minArea = 500
        
        img_orig = img.copy()# Keep it for

        # ****************************************************  DETECTION [LANES] ****************************************************

        # 1. Extracting Information that best defines relation between lanes and Car
        distance, Curvature = Detect_Lane(img)

        # ****************************************************  DETECTION  [LANES] ****************************************************

        # ****************************************************  DETECTION  [SIGNS] ****************************************************
        Disp_img = img.copy()

        # 2. Detecting and extracting information from the signs
        Mode , Tracked_class = detect_Signs(img_orig,img)

        # ****************************************************  DETECTION  [SIGNS] ****************************************************
        
        # 3. Drive Car On Basis of Current State Info provided by Detection module
        Current_State = [distance, Curvature , img , Mode , Tracked_class]
        Drive_Car(Current_State)
        
        
        config.loopCount = config.loopCount+1
        if(config.loopCount==50000):
            break

        cv2.imshow("Frame",img)
        k = cv2.waitKey(waitTime)
        if k==27:
            break


    if Use_Threading:
        PI_vs.stop() 


if __name__ == '__main__':
	main()
#git push from raspberry pi
#Control Variables for 3c_threaded_Mod4
import os
import cv2

detect = 1 # Set to 1 for Lane detection

Testing = True# Set to True --> if want to see what the car is seeing
Profiling = False # Set to True --> If you want to profile code

debugging = True # Set to True --> If you want to debug code
clr_segmentation_tuning = True # Set to True --> If you want to tune color segmentation parameters

Detect_lane_N_Draw = True

vid_path = os.path.abspath("Detection/Lanes/Inputs/signs_forward.mp4")
#vid_path = os.path.abspath("Detection/Lanes/Inputs/in_16_2.avi")
loopCount=0


Resized_width = 320#320#240#640#320 # Control Parameter
Resized_height = 240#240#180#480#240


if debugging:
    waitTime = 1
else:
    waitTime = 1

#============================================ Paramters for Lane Detection =======================================
Ref_imgWidth = 1920
Ref_imgHeight = 1080
Frame_pixels = Ref_imgWidth * Ref_imgHeight

Resize_Framepixels = Resized_width * Resized_height

Lane_Extraction_minArea_per = 700 / Frame_pixels
minArea_resized = int(Resize_Framepixels * Lane_Extraction_minArea_per)
#600
BWContourOpen_speed_MaxDist_per = 400 / Ref_imgHeight
MaxDist_resized = int(Resized_height * BWContourOpen_speed_MaxDist_per)

CropHeight = 750
CropHeight_resized_crop = int( (CropHeight / Ref_imgHeight ) * Resized_height )
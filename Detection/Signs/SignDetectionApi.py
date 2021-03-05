import tensorflow as tf # tensorflow imported to check installed tf version
from tensorflow.keras.models import load_model # import load_model function to load trained CNN model for Sign classification
import timeit
import os # for getting absolute filepath to mitigate cross platform inconsistensies
import cv2
import time
import numpy as np
import config

detected_img = 0 #Set this to current dataset images size so that new images number starts from there and dont overwrite
if config.Detect_lane_N_Draw:
    write_data = False
else:
    write_data = True
draw_detected = True
display_images = False
model_loaded = False
model = 0
sign_classes = ["speed_sign_40","speed_sign_70","stop","No_Sign"] # Trained CNN Classes


def image_forKeras(image):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)# Image everywher is in rgb but Opencv does it in BGR convert Back
    image = cv2.resize(image,(30,30)) #Resize to model size requirement
    image = np.expand_dims(image, axis=0) # Dimension of model is [Batch_size, input_row,inp_col , inp_chan]
    return image

def SignDetection(gray,cimg,frame_draw,model):
    NumOfVotesForCircle = 40 #parameter 1 MinVotes needed to be classified as circle
    CannyHighthresh = 200 # High threshold value for applying canny
    mindDistanBtwnCircles = 100 # kept as sign will likely not be overlapping
    max_rad = 150 # smaller circles dont have enough votes so only maxRadius need to be controlled 
                    # As signs are right besides road so they will eventually be in view so ignore circles larger than said limit

    circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,mindDistanBtwnCircles,param1=CannyHighthresh,param2=NumOfVotesForCircle,minRadius=10,maxRadius=max_rad)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            center =(i[0],i[1])
            radius = i[2] + 5
            if (radius !=5):
                global detected_img
                detected_img = detected_img + 1 

                startP = (center[0]-radius,center[1]-radius)
                endP = (center[0]+radius,center[1]+radius)
                
                detected_sign = cimg[startP[1]:endP[1],startP[0]:endP[0]]

                if(detected_sign.shape[1] and detected_sign.shape[0]):
                    sign = sign_classes[np.argmax(model(image_forKeras(detected_sign)))]
                    if(sign != "No_Sign"):
                        cv2.putText(frame_draw,sign,(endP[0]-20,startP[1]+10),cv2.FONT_HERSHEY_PLAIN,0.5,(0,0,255),1)
                    if write_data:
                        if  (sign =="speed_sign_40"):
                            class_id ="0/"
                        elif(sign =="speed_sign_70"):
                            class_id ="1/"
                        elif(sign =="stop"):
                            class_id ="2/"
                        else:
                            class_id ="3/"
                        img_dir = os.path.abspath("Detection/Signs/datasets/") + class_id
                        #img_name = "Detection/Signs/datasets/"+ class_id + str(detected_img)+".png"
                        img_name = img_dir + str(detected_img)+".png"
                        if not os.path.exists(img_dir):
                            os.mkdir(img_dir)
                        cv2.imwrite(img_name , detected_sign)
                    if draw_detected:
                        if(sign != "No_Sign"):
                            # draw the outer circle
                            cv2.circle(frame_draw,(i[0],i[1]),i[2],(0,255,0),1)
                            # draw the center of the circle
                            cv2.circle(frame_draw,(i[0],i[1]),2,(0,0,255),3)
        
        if display_images:
            cimg_str = 'detected circles'
            cv2.imshow(cimg_str,frame_draw)
            cv2.waitKey(1)

def detect_Signs(frame,frame_draw):
    
    global model_loaded
    if not model_loaded:
        print(tf.__version__)#2.4.1
        print("************ LOADING MODEL **************")
        global model
        # load model
        model = load_model(os.path.abspath('Detection/Signs/models/4_signClassification_model_nano.h5'),compile=False)
        # summarize model.
        model.summary()
        model_loaded = True

    # Convert Rgb to colourImg
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # Localizing Potetial Candidates and Classifying them in SignDetection
    start_signDetection = time.time()
    SignDetection(gray.copy(),frame.copy(),frame_draw,model)
    end_signDetection = time.time()

    print("[Profiling] SignDetection took ",end_signDetection - start_signDetection," sec <-->  ",(1/(end_signDetection - start_signDetection)),"  FPS ")

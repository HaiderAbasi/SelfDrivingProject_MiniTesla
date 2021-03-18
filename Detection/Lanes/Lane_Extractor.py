import cv2
import numpy as np
from Detection.Lanes.Morph_op import BwareaOpen,RetLargestContour,RetLargestContour_OuterLane,RetClosestContour2
import time
import config

Hue_Low = 82#74#83 63
#Lit_Low = 175#174#112 Prevopus
Lit_Low = 95#174#112 
Sat_Low = 0

#Hue_Low_Y = 31
#Hue_High_Y = 51
Hue_Low_Y = 30
Hue_High_Y = 49#42#38
#Hue_High_Y = 45
Lit_Low_Y = 0
Sat_Low_Y = 51


def OnHueLowChange(val):
	global Hue_Low
	Hue_Low = val
def OnLitLowChange(val):
	global Lit_Low
	Lit_Low = val
def OnSatLowChange(val):
	global Sat_Low
	Sat_Low = val

def OnHueLowChange_Y(val):
	global Hue_Low_Y
	Hue_Low_Y = val
def OnHueHighChange_Y(val):
	global Hue_High_Y
	Hue_High_Y = val	
def OnLitLowChange_Y(val):
	global Lit_Low_Y
	Lit_Low_Y = val
def OnSatLowChange_Y(val):
	global Sat_Low_Y
	Sat_Low_Y = val


#cv2.namedWindow("HSL",cv2.WINDOW_NORMAL)
#cv2.namedWindow("frame_Lane",cv2.WINDOW_NORMAL)
#cv2.namedWindow("Lane_gray",cv2.WINDOW_NORMAL)
#cv2.namedWindow("Lane_gray_opened",cv2.WINDOW_NORMAL)
#cv2.namedWindow("Lane_gray_Smoothed",cv2.WINDOW_NORMAL)
#cv2.namedWindow("Lane_edge",cv2.WINDOW_NORMAL)
#cv2.namedWindow("Lane_edge_ROI",cv2.WINDOW_NORMAL)
#cv2.namedWindow("Mid_ROI_mask",cv2.WINDOW_NORMAL)


if(config.clr_segmentation_tuning):

    cv2.namedWindow("mask",cv2.WINDOW_NORMAL)
    cv2.namedWindow("mask_Y",cv2.WINDOW_NORMAL)

    cv2.createTrackbar("Hue_L","mask",Hue_Low,255,OnHueLowChange)
    cv2.createTrackbar("Lit_L","mask",Lit_Low,255,OnLitLowChange)
    cv2.createTrackbar("Sat_L","mask",Sat_Low,255,OnSatLowChange)

    cv2.createTrackbar("Hue_L","mask_Y",Hue_Low_Y,255,OnHueLowChange_Y)
    cv2.createTrackbar("Hue_H","mask_Y",Hue_High_Y,255,OnHueHighChange_Y)
    cv2.createTrackbar("Lit_L","mask_Y",Lit_Low_Y,255,OnLitLowChange_Y)
    cv2.createTrackbar("Sat_L","mask_Y",Sat_Low_Y,255,OnSatLowChange_Y)

def clr_segment(HSL,lower_range,upper_range):
    lower = np.array( [lower_range[0],lower_range[1] ,lower_range[2]] )
    upper = np.array( [upper_range[0]    ,255     ,255])
    mask = cv2.inRange(HSL, lower, upper)
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    return mask

def LaneROI(frame,mask,minArea):
    frame_Lane = cv2.bitwise_and(frame,frame,mask=mask)# R & R = R
    Lane_gray = cv2.cvtColor(frame_Lane,cv2.COLOR_BGR2GRAY)
    #start_To = time.time()
    Lane_gray_opened = BwareaOpen(Lane_gray,minArea)#4 msec
    #end_To = time.time()
    #print("**** BWarea Open Loop took ",end_To - start_To," sec <-->  ")
    #Lane_gray = cv2.bitwise_and(Lane_gray,Lane_gray,mask=Lane_gray_opened)# R & R = R
    Lane_gray = cv2.bitwise_and(Lane_gray,Lane_gray_opened)# R & R = R
    Lane_gray_Smoothed = cv2.GaussianBlur(Lane_gray,(11,11),1)
    Lane_edge = cv2.Canny(Lane_gray_Smoothed,50,150, None, 3)#3msec

    #  Selecting Only ROI from Image
    # ROI_mask = np.zeros(Lane_edge.shape, dtype=np.uint8)
    # cv2.rectangle(ROI_mask,(0,CropHeight_resized),(Lane_edge.shape[1],Lane_edge.shape[0]),255,thickness=-1)
    # Lane_edge_ROI = cv2.bitwise_and(Lane_edge,Lane_edge,mask=ROI_mask)
    # Lane_ROI_mask = cv2.bitwise_and(Lane_gray_opened,Lane_gray_opened,mask=ROI_mask)

    #cv2.imshow('ROI_mask',ROI_mask)
    #cv2.imshow('frame_Lane',frame_Lane)
    #cv2.imshow('Lane_gray',Lane_gray)
    #cv2.imshow('Lane_gray_opened',Lane_gray_opened)
    #cv2.imshow('Lane_gray_Smoothed',Lane_gray_Smoothed)
    #cv2.imshow('Lane_edge',Lane_edge)
    #cv2.imshow('Lane_edge_ROI',Lane_edge_ROI)

    #return Lane_edge_ROI,Lane_ROI_mask
    return Lane_edge,Lane_gray_opened

def ROI_extracter(image,mask,strtPnt):
    #  Selecting Only ROI from Image
    ROI_mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.rectangle(ROI_mask,strtPnt,(image.shape[1],image.shape[0]),255,thickness=-1)
    #cv2.imshow('ROI_mask',ROI_mask)
    image_ROI = cv2.bitwise_and(image,image,mask=ROI_mask)
    image_ROI_mask = cv2.bitwise_and(mask,mask,mask=ROI_mask)
    return image_ROI,image_ROI_mask

def OuterLaneROI(frame,mask,minArea):
    Outer_Points_list=[]
    frame_Lane = cv2.bitwise_and(frame,frame,mask=mask)# R & R = R
    Lane_gray = cv2.cvtColor(frame_Lane,cv2.COLOR_BGR2GRAY)
    Lane_gray_opened = BwareaOpen(Lane_gray,minArea)
    #Lane_gray = cv2.bitwise_and(Lane_gray,Lane_gray,mask=Lane_gray_opened)# R & R = R
    Lane_gray = cv2.bitwise_and(Lane_gray,Lane_gray_opened)# R & R = R
    Lane_gray_Smoothed = cv2.GaussianBlur(Lane_gray,(11,11),1)
    Lane_edge = cv2.Canny(Lane_gray_Smoothed,50,150, None, 3)
    #  Selecting Only ROI from Image
    #Lane_edge_ROI,Lane_ROI_mask = ROI_extracter(Lane_edge,Lane_gray_opened,(0,CropHeight_resized))
    #ROI_mask_Largest,Largest_found = RetLargestContour_OuterLane(Lane_ROI_mask,minArea)
    ROI_mask_Largest,Largest_found = RetLargestContour_OuterLane(Lane_gray_opened,minArea)
    if(Largest_found):
        Outer_edge_lane_final = cv2.bitwise_and(Lane_edge,ROI_mask_Largest)
        #cv2.namedWindow("ROI_mask_Largest",cv2.WINDOW_NORMAL)
        #cv2.imshow('ROI_mask_Largest',ROI_mask_Largest)
        Lane_OneSide ,Lane_OneSide_found,Outer_Points_list = RetClosestContour2(ROI_mask_Largest)
        Lane_edge = Outer_edge_lane_final
        #Lane_edge_ROI = Outer_edge_lane_final
    else:
        Lane_OneSide=np.zeros(Lane_gray.shape,Lane_gray.dtype)
    #cv2.imshow('frame_Lane',frame_Lane)
    #cv2.imshow('Lane_gray',Lane_gray)
    #cv2.imshow('Lane_gray_opened',Lane_gray_opened)
    #cv2.imshow('Lane_gray_Smoothed',Lane_gray_Smoothed)
    #cv2.imshow('Lane_edge',Lane_edge)
    #cv2.imshow('Lane_edge_ROI',Lane_edge_ROI)

    #return Lane_edge_ROI,Lane_ROI_mask,Lane_OneSide,Outer_Points_list
    return Lane_edge,Lane_gray_opened,Lane_OneSide,Outer_Points_list

def GetLaneROI(frame,minArea):

    #start_To_clr_segment = time.time()
    HSL = cv2.cvtColor(frame,cv2.COLOR_BGR2HLS)#2 msc
    #end_To_clr_segment = time.time()
    #print("**** CVTCOLOR -> HSV Loop took ",end_To_clr_segment - start_To_clr_segment," sec <-->  ")

    mask   = clr_segment(HSL,(Hue_Low  ,Lit_Low   ,Sat_Low  ),(255       ,255,255))
    mask_Y = clr_segment(HSL,(Hue_Low_Y,Lit_Low_Y ,Sat_Low_Y),(Hue_High_Y,255,255))#Combine 6ms
    
    #start_To = time.time()
    #Outer_edge_ROI,_,OuterLane_OneSide,Outer_Points_list = OuterLaneROI(frame,mask_Y,minArea+500,CropHeight_resized)#27msec
    Outer_edge_ROI,_,OuterLane_OneSide,Outer_Points_list = OuterLaneROI(frame,mask_Y,minArea+500)#27msec
    #end_To = time.time()
    #print("**** OuterLANEROI Loop took ",end_To - start_To," sec <-->  ")
    #start_To2 = time.time()
    #Mid_edge_ROI,Mid_ROI_mask = LaneROI(frame,mask,minArea,CropHeight_resized)#20 msec
    Mid_edge_ROI,Mid_ROI_mask = LaneROI(frame,mask,minArea)#20 msec
    #end_To2 = time.time()
    #print("**** LANEROI Loop took ",end_To2 - start_To2," sec <-->  ")
    #cv2.imshow('HSL',HSL)
    #cv2.imshow('Mid_ROI_mask',Mid_ROI_mask)

    if(config.clr_segmentation_tuning):
        cv2.imshow('mask',mask)
        cv2.imshow('mask_Y',mask_Y)

    return Mid_edge_ROI,Mid_ROI_mask,Outer_edge_ROI,None,OuterLane_OneSide,Outer_Points_list
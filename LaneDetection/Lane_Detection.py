import config
import cv2
import numpy as np 
import matplotlib as plt
import math
from utilities import average_2b,findLineParameter,findlaneCurvature,Distance_
from Lane_Extractor import GetLaneROI
from Morph_op import FindExtremas,BWContourOpen_speed
import time
from sys import platform
#==========Special Imports====================================
if (config.debugging==False):
	from imutils.video.pivideostream import PiVideoStream
	from imutils.video import FPS
	import imutils
	import argparse
	from Motors_control import forward,backward,setServoAngle,stop,turnOfCar,changePwm,beInLane
#=============================================================

if platform == "linux":
    vid_path = "/home/pi/Desktop/SelfDrivingProject_MiniTesla/LaneDetection/Inputs/in2.avi"
    txt_path = "/home/pi/Desktop/SelfDrivingProject_MiniTesla/LaneDetection/Results/LaneDetection_out.txt"
    
else:
    vid_path = "LaneDetection/Inputs/in2.avi"
    txt_path = "LaneDetection/Results/LaneDetection_out.txt"
if (config.debugging):
	cap = cv2.VideoCapture(vid_path)
	cv2.namedWindow('Vid',cv2.WINDOW_NORMAL)
	fps = cap.get(cv2.CAP_PROP_FPS)
	frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	if(frame_count==0):
		frame_count=1127
	duration = int(frame_count / fps)
	print(fps)
	print(frame_count)
	print(duration)
	Video_pos = 35#sec
	#cap = cv2.VideoCapture("/home/pi/Desktop/pivideo.mp4")
else:
	cap = cv2.VideoCapture(0)


def OnVidPosChange(val):
	global Video_pos
	Video_pos = val
	print(Video_pos)
	cap.set(cv2.CAP_PROP_POS_MSEC,Video_pos*1000)

def SegmentLanesAndDisplay_2(copy):
	Lane_trajectory = average_2b(copy)
	return Lane_trajectory


def Cord_Sort(cnts,order):

	cnt=cnts[0]
	cnt=np.reshape(cnt,(cnt.shape[0],cnt.shape[2]))
	order_list=[]
	if(order=="rows"):
		order_list.append((0,1))
	else:
		order_list.append((1,0))
	ind = np.lexsort((cnt[:,order_list[0][0]],cnt[:,order_list[0][1]]))
	Sorted=cnt[ind]
	return Sorted
		

def FindClosestLane(OuterLanes,MidLane,OuterLane_Points):
	#  Fetching the closest outer lane to mid lane is the main goal here
	
	#Container for storing/returning closest Outer Lane
	Outer_Lanes_ret= np.zeros(OuterLanes.shape,OuterLanes.dtype)

	Offset_correction = 0
	Mid_cnts = cv2.findContours(MidLane, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
	
	Ref=(0,0)
	if(Mid_cnts):
		Ref = tuple(Mid_cnts[0][0][0])

	#Condition 1 : if MidLane is present but no Outlane detected
	# Create Outlane on Side that represent the larger Lane as seen by camera
	if(Mid_cnts and (len(OuterLane_Points)==0)):
		# Condition where MidCnts are detected 
		Mid_cnts_Rowsorted = Cord_Sort(Mid_cnts,"rows")
		Mid_Rows = Mid_cnts_Rowsorted.shape[0]

		Mid_lowP = Mid_cnts_Rowsorted[Mid_Rows-1,:]
		Mid_highP = Mid_cnts_Rowsorted[0,:]
		Mid_median_Col = int( (Mid_lowP[0]  + Mid_highP[0] ) / 2 ) 
		if(Mid_median_Col >= int(MidLane.shape[1]/2)):
			low_Col=0
			high_Col=0
			Offset_correction = -20
		else:
			low_Col=(int(MidLane.shape[1])-1)
			high_Col=(int(MidLane.shape[1])-1)
			Offset_correction = 20

		Mid_lowP[1] = MidLane.shape[0]# setting mid_trajectory_lowestPoint_Row to MaxRows of Image
		#LanePoint_lower = ( int( (Mid_lowP[0]  + low_Col ) / 2 ) , int( Mid_lowP[1] ) )
		#LanePoint_top   = ( int( (Mid_highP[0] + high_Col) / 2 ) , int( Mid_highP[1]) )
		LanePoint_lower =  (low_Col , int( Mid_lowP[1] ) )
		LanePoint_top   =  (high_Col, int( Mid_highP[1]) )

		print(" Mid_lower_row = ", Mid_lowP[1])
		print(" Mid_higher_row = ", Mid_highP[1])
		OuterLanes = cv2.line(OuterLanes,LanePoint_lower,LanePoint_top,255,2)		

	Outer_cnts = cv2.findContours(OuterLanes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

	if(len(OuterLane_Points)==2):
		Point_a=OuterLane_Points[0]
		Point_b=OuterLane_Points[1]

		Closest_Index=0
		if(Distance_(Point_a,Ref)<=Distance_(Point_b,Ref)):
			Closest_Index=0
		elif(len(Outer_cnts)>1):
			Closest_Index=1
		#cv2.namedWindow("OuterLanes",cv2.WINDOW_NORMAL)
		#cv2.imshow("OuterLanes",OuterLanes)
		#cv2.waitKey(0)		
		#print("OuterLane_Points ",OuterLane_Points)
		#print("len(Outer_cnts) ",len(Outer_cnts))
		#print("Closest_Index ",Closest_Index)
		Outer_Lanes_ret = cv2.drawContours(Outer_Lanes_ret, Outer_cnts, Closest_Index, 255, 2)
		Outer_cnts_ret = [Outer_cnts[Closest_Index]]
		return Outer_Lanes_ret ,Outer_cnts_ret, Mid_cnts,0
	else:
		return OuterLanes, Outer_cnts, Mid_cnts, Offset_correction


def ExtendLane(Lane,Lane_RowSorted,RefLane_RowSorted):

	Lane_Rows = Lane_RowSorted.shape[0]
	Lane_Cols = Lane_RowSorted.shape[1]
	RefLane_Rows = RefLane_RowSorted.shape[0]
	RefLane_Cols = RefLane_RowSorted.shape[1]

	BottomPoint= Lane_RowSorted[Lane_Rows-1,:]
	RefBottomPoint= RefLane_RowSorted[RefLane_Rows-1,:]

	if(Lane_Rows>20):
		shift=20
	else:
		shift=2
	Last10Points = Lane_RowSorted[Lane_Rows-shift:Lane_Rows-1:2,:]
	#print("Last10Points = ",Last10Points)
	if(len(Last10Points)>1):# Atleast 2 points needed to estimate a line
		#time.sleep(10)
		x = Last10Points[:,0]#cols
		y = Last10Points[:,1]#rows
		parameters = np.polyfit(x, y, 1)
		slope = parameters[0]
		yiCntercept = parameters[1]
		#print("slope",slope)
		#Decreasing slope means Current lane is left lane and by going towards 0 x we touchdown
		if(slope<0):
			LineTouchPoint_col=0
			LineTouchPoint_row=yiCntercept
		else:
			LineTouchPoint_col=Lane.shape[1]-1 # Cols have lenth of ColLength But traversal is from 0 to ColLength-1
			LineTouchPoint_row=slope*LineTouchPoint_col + yiCntercept
		
		TouchPoint=(LineTouchPoint_col,int(LineTouchPoint_row))#(col ,row)
		BottomPoint_tup = tuple(BottomPoint)
		Lane = cv2.line(Lane,TouchPoint,BottomPoint_tup,255)		

		TouchPoint_Ref = (LineTouchPoint_col,RefLane_RowSorted[RefLane_RowSorted.shape[0]-1,1])
		Lane = cv2.line(Lane,TouchPoint,TouchPoint_Ref,255)

		#print("Drawing line form ",BottomPoint_tup," to ",TouchPoint )
		#print("Drawing line form ",TouchPoint," to ", TouchPoint_Ref)
		#time.sleep(10)
	return Lane

def ExtendShortLane(MidLane,Mid_cnts,Outer_cnts,OuterLane):

	NonExtendedLane=0 # 0: None .1: Mid, 2:Outer

	if(Mid_cnts and Outer_cnts):
		Mid_cnts_Rowsorted = Cord_Sort(Mid_cnts,"rows")
		Outer_cnts_Rowsorted = Cord_Sort(Outer_cnts,"rows")
		#Both are not empty so we need to find shorter between the two
		Mid_Rows = Mid_cnts_Rowsorted.shape[0]
		Mid_cnts_lowestRow = Mid_cnts_Rowsorted[Mid_Rows-1,1]
		
		Outer_Rows = Outer_cnts_Rowsorted.shape[0]
		Outer_cnts_lowestRow = Outer_cnts_Rowsorted[Outer_Rows-1,1]

		if(Mid_cnts_lowestRow<=Outer_cnts_lowestRow):
			ExtendLine=0# MidLine
		else:
			ExtendLine=1#OuterLine
		if(ExtendLine==0):
			MidLane = ExtendLane(MidLane,Mid_cnts_Rowsorted,Outer_cnts_Rowsorted)
			NonExtendedLane=1#outer
		else:
			OuterLane = ExtendLane(OuterLane,Outer_cnts_Rowsorted,Mid_cnts_Rowsorted)
			NonExtendedLane=2#mid
	return MidLane,OuterLane,NonExtendedLane

def ExtendBothLane(Lane,OuterLane,Lane_RowSorted,RefLane_RowSorted):

	Image_bottom = Lane.shape[0]
	
	Lane_Rows = Lane_RowSorted.shape[0]
	Lane_Cols = Lane_RowSorted.shape[1]
	BottomPoint = Lane_RowSorted[Lane_Rows-1,:]	

	if (BottomPoint[1] < Image_bottom):
		if(Lane_Rows>20):
			shift=20
		else:
			shift=2
		Last10Points = Lane_RowSorted[Lane_Rows-shift:Lane_Rows-1:2,:]
		#print("Last10Points = ",Last10Points)
		if(len(Last10Points)>1):# Atleast 2 points needed to estimate a line
			#time.sleep(10)
			x = Last10Points[:,0]#cols
			y = Last10Points[:,1]#rows
			parameters = np.polyfit(x, y, 1)
			slope = parameters[0]
			yiCntercept = parameters[1]
			#print("slope",slope)
			#Decreasing slope means Current lane is left lane and by going towards 0 x we touchdown
			if(slope < 0):
				LineTouchPoint_col = 0
				LineTouchPoint_row = yiCntercept
			else:
				LineTouchPoint_col = Lane.shape[1]-1 # Cols have lenth of ColLength But traversal is from 0 to ColLength-1
				LineTouchPoint_row = slope*LineTouchPoint_col + yiCntercept
			
			TouchPoint=(LineTouchPoint_col,int(LineTouchPoint_row))#(col ,row)
			BottomPoint_tup = tuple(BottomPoint)
			Lane = cv2.line(Lane,TouchPoint,BottomPoint_tup,255)

			if(LineTouchPoint_row < Image_bottom):
				TouchPoint_Ref = (LineTouchPoint_col,Image_bottom)
				Lane = cv2.line(Lane,TouchPoint,TouchPoint_Ref,255)

	RefLane_Rows = RefLane_RowSorted.shape[0]
	RefLane_Cols = RefLane_RowSorted.shape[1]
	RefBottomPoint = RefLane_RowSorted[RefLane_Rows-1,:]

	if (RefBottomPoint[1] < Image_bottom):
		if(RefLane_Rows>20):
			shift=20
		else:
			shift=2
		RefLast10Points = RefLane_RowSorted[RefLane_Rows-shift:RefLane_Rows-1:2,:]
		#print("Last10Points = ",Last10Points)
		if(len(RefLast10Points)>1):# Atleast 2 points needed to estimate a line
			#time.sleep(10)
			Ref_x = RefLast10Points[:,0]#cols
			Ref_y = RefLast10Points[:,1]#rows
			Ref_parameters = np.polyfit(Ref_x, Ref_y, 1)
			Ref_slope = Ref_parameters[0]
			Ref_yiCntercept = Ref_parameters[1]
			#print("slope",slope)
			#Decreasing slope means Current lane is left lane and by going towards 0 x we touchdown
			if(Ref_slope < 0):
				Ref_LineTouchPoint_col = 0
				Ref_LineTouchPoint_row = Ref_yiCntercept
			else:
				Ref_LineTouchPoint_col = OuterLane.shape[1]-1 # Cols have lenth of ColLength But traversal is from 0 to ColLength-1
				Ref_LineTouchPoint_row = Ref_slope * Ref_LineTouchPoint_col + Ref_yiCntercept
			
			Ref_TouchPoint = (Ref_LineTouchPoint_col,int(Ref_LineTouchPoint_row))#(col ,row)
			Ref_BottomPoint_tup = tuple(RefBottomPoint)
			OuterLane = cv2.line(OuterLane,Ref_TouchPoint,Ref_BottomPoint_tup,255)

			if(Ref_LineTouchPoint_row < Image_bottom):
				Ref_TouchPoint_Ref = (Ref_LineTouchPoint_col,Image_bottom)
				OuterLane = cv2.line(OuterLane,Ref_TouchPoint,Ref_TouchPoint_Ref,255)
		#print("Drawing line form ",BottomPoint_tup," to ",TouchPoint )
		#print("Drawing line form ",TouchPoint," to ", TouchPoint_Ref)
		#time.sleep(10)

	return Lane , OuterLane

def ExtendShortLane_(MidLane,Mid_cnts,Outer_cnts,OuterLane):

	if(Mid_cnts and Outer_cnts):
		Mid_cnts_Rowsorted = Cord_Sort(Mid_cnts,"rows")
		Outer_cnts_Rowsorted = Cord_Sort(Outer_cnts,"rows")

		MidLane,OuterLane = ExtendBothLane(MidLane,OuterLane,Mid_cnts_Rowsorted,Outer_cnts_Rowsorted)

	return MidLane,OuterLane


def RefineMidEdgeROi(Mid_trajectory,Mid_ROI_mask,Mid_edge_ROI):

	contours_trajectory= cv2.findContours(Mid_trajectory,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[1]
	contours = cv2.findContours(Mid_ROI_mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[1]
	Mid_ROI_mask_refined = np.zeros(Mid_ROI_mask.shape,Mid_ROI_mask.dtype)
	Modified= False
	for index, cnt in enumerate(contours):
		Draw=False
		for i in range(len(contours_trajectory)):
			PointTotest = contours_trajectory[i][0][0]
			#print("PointTotest",PointTotest)
			if(cv2.pointPolygonTest(cnt, tuple(PointTotest), False)>=0):
				Draw = True
				Modified=True 
		if Draw:
			Mid_ROI_mask_refined = cv2.drawContours(Mid_ROI_mask_refined, contours, index,255, 1)
	Mid_edge_ROI = cv2.bitwise_and(Mid_ROI_mask_refined,Mid_edge_ROI)
	return Mid_edge_ROI


def EstimateNonMidMask(MidEdgeROi):
	Mid_Hull_Mask = np.zeros((MidEdgeROi.shape[0], MidEdgeROi.shape[1], 1), dtype=np.uint8)
	contours = cv2.findContours(MidEdgeROi,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[1]
	if contours:
		hull_list = []
		contours = np.concatenate(contours)
		hull = cv2.convexHull(contours)
		hull_list.append(hull)
		# Draw contours + hull results
		Mid_Hull_Mask = cv2.drawContours(Mid_Hull_Mask, hull_list, 0, 255,-1)
		#cv2.namedWindow("Mid_Hull_Mask",cv2.WINDOW_NORMAL)
		#cv2.imshow("Mid_Hull_Mask",Mid_Hull_Mask)
	Non_Mid_Mask=cv2.bitwise_not(Mid_Hull_Mask)
	return Non_Mid_Mask

def LanePoints_(MidLane,OuterLane,Mid_cnts,Outer_cnts,NonExtendedLane,Offset_correction):
	#MODED
	if(NonExtendedLane==1):#1 : Means Mid was extended so need to findContour Mid again
		Mid_cnts = cv2.findContours(MidLane, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
	elif(NonExtendedLane==2):
		Outer_cnts = cv2.findContours(OuterLane, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

	if(Mid_cnts and Outer_cnts):
		Mid_cnts_Rowsorted = Cord_Sort(Mid_cnts,"rows")
		Outer_cnts_Rowsorted = Cord_Sort(Outer_cnts,"rows")
		#print(Mid_cnts_Rowsorted)
		Mid_Rows = Mid_cnts_Rowsorted.shape[0]
		Outer_Rows = Outer_cnts_Rowsorted.shape[0]

		Mid_lowP = Mid_cnts_Rowsorted[Mid_Rows-1,:]
		Mid_highP = Mid_cnts_Rowsorted[0,:]
		Outer_lowP = Outer_cnts_Rowsorted[Outer_Rows-1,:]
		Outer_highP = Outer_cnts_Rowsorted[0,:]

		LanePoint_lower = ( int( (Mid_lowP[0] + Outer_lowP[0]  ) / 2 ) + Offset_correction, int( (Mid_lowP[1]  + Outer_lowP[1] ) / 2 ) )
		LanePoint_top   = ( int( (Mid_highP[0] + Outer_highP[0]) / 2 ) + Offset_correction, int( (Mid_highP[1] + Outer_highP[1]) / 2 ) )

		return LanePoint_lower,LanePoint_top
	else:
		return (0,0),(0,0)

def LanePoints(MidLane,OuterLane,Offset_correction):

	Mid_cnts = cv2.findContours(MidLane, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
	Outer_cnts = cv2.findContours(OuterLane, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

	if(Mid_cnts and Outer_cnts):
		Mid_cnts_Rowsorted = Cord_Sort(Mid_cnts,"rows")
		Outer_cnts_Rowsorted = Cord_Sort(Outer_cnts,"rows")
		#print(Mid_cnts_Rowsorted)
		Mid_Rows = Mid_cnts_Rowsorted.shape[0]
		Outer_Rows = Outer_cnts_Rowsorted.shape[0]

		Mid_lowP = Mid_cnts_Rowsorted[Mid_Rows-1,:]
		Mid_highP = Mid_cnts_Rowsorted[0,:]
		Outer_lowP = Outer_cnts_Rowsorted[Outer_Rows-1,:]
		Outer_highP = Outer_cnts_Rowsorted[0,:]

		LanePoint_lower = ( int( (Mid_lowP[0] + Outer_lowP[0]  ) / 2 ) + Offset_correction, int( (Mid_lowP[1]  + Outer_lowP[1] ) / 2 ) )
		LanePoint_top   = ( int( (Mid_highP[0] + Outer_highP[0]) / 2 ) + Offset_correction, int( (Mid_highP[1] + Outer_highP[1]) / 2 ) )

		return LanePoint_lower,LanePoint_top
	else:
		return (0,0),(0,0)

def DrawProbablePath_(Outer_Lane,Mid_lane,Mid_cnts,Outer_cnts,MidEdgeROi,frame,Offset_correction):
	Lanes_combined = cv2.bitwise_or(Outer_Lane,Mid_lane)
	#cv2.namedWindow("Lanes_combined",cv2.WINDOW_NORMAL)
	if config.Testing:
		cv2.imshow("9.5: Lanes_combined",Lanes_combined)
	#Creating an empty image
	ProjectedLane = np.zeros(Lanes_combined.shape,Lanes_combined.dtype)
	cnts = cv2.findContours(Lanes_combined,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[1]

	drawn=False
	if cnts:
		cnts = np.concatenate(cnts)
		cnts = np.array(cnts)
		cv2.fillConvexPoly(ProjectedLane, cnts, 255)
		#cv2.namedWindow("ProjectedLane",cv2.WINDOW_NORMAL)
	if config.Testing:
		cv2.imshow("9.8: ProjectedLane",ProjectedLane)

		Mid_less_Mask = EstimateNonMidMask(MidEdgeROi)
		ProjectedLane = cv2.bitwise_and(Mid_less_Mask,ProjectedLane)
		# copy where we'll assign the new values
		Lane_drawn_frame = frame
		#Lane_drawn_frame = np.copy(frame)
		# boolean indexing and assignment based on mask
		#Lane_drawn_frame[ProjectedLane==255] = (0,255,0)
		#Lane_drawn_frame[Outer_Lane==255] = (0,0,255)# Outer Lane Coloured Red
		#Lane_drawn_frame[Mid_lane==255] = (255,0,0)# Mid Lane Coloured Blue

		Lane_drawn_frame[ProjectedLane==255] = Lane_drawn_frame[ProjectedLane==255] + (0,100,0)
		Lane_drawn_frame[Outer_Lane==255] = Lane_drawn_frame[Outer_Lane==255] + (0,0,100)# Outer Lane Coloured Red
		Lane_drawn_frame[Mid_lane==255] = Lane_drawn_frame[Mid_lane==255] + (100,0,0)# Mid Lane Coloured Blue
		#cv2.namedWindow("Lane_drawn_frame",cv2.WINDOW_NORMAL)
		#cv2.imshow("Lane_drawn_frame",Lane_drawn_frame)
		#Lane_drawn_frame_w = cv2.addWeighted(Lane_drawn_frame, 0.5, frame, 0.7, 0)
		Lane_drawn_frame_w = Lane_drawn_frame
		#cv2.namedWindow("Lane_drawn_frame_w",cv2.WINDOW_NORMAL)
		#cv2.imshow("Lane_drawn_frame_w",Lane_drawn_frame_w)

		Out_image = Lane_drawn_frame_w
		cv2.line(Out_image,(int(Out_image.shape[1]/2),Out_image.shape[0]),(int(Out_image.shape[1]/2),Out_image.shape[0]-int (Out_image.shape[0]/5)),(0,0,255),2)
		Traj_lowP,Traj_upP = LanePoints(Mid_lane,Outer_Lane,Offset_correction)
		#print("Lines is drawn for = ",Traj_lowP," , ",Traj_upP)
		cv2.line(Out_image,Traj_lowP,Traj_upP,(255,0,0),2)
		PerpDist_ImgCen_CarNose= -1000
		if(Traj_lowP!=(0,0)):
			cv2.line(Out_image,Traj_lowP,(int(Out_image.shape[1]/2),Traj_lowP[1]),(255,255,0),2)# distance of car center with lane path
			PerpDist_ImgCen_CarNose = Traj_lowP[0] - int(Out_image.shape[1]/2)

		curvature = findlaneCurvature(Traj_lowP[0],Traj_lowP[1],Traj_upP[0],Traj_upP[1])
		#texttoPut="Curvature = "+str(curvature)
		texttoPut="Curvature = " + f"{curvature:.2f}"
		texttoPut2="Distance = " + str(PerpDist_ImgCen_CarNose)
		#print(texttoPut)
		cv2.putText(Out_image,texttoPut,(Out_image.shape[1]-400,50),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,255),2)
		cv2.putText(Out_image,texttoPut2,(Out_image.shape[1]-400,80),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,255),2)
		#Out_image = cv2.resize(Out_image,(1280,720))
		drawn=True		
		return Out_image,drawn
	else:
		return Outer_Lane,drawn

def DrawProbablePath(Outer_Lane,Mid_lane,Mid_cnts,Outer_cnts,MidEdgeROi,frame,Offset_correction):

	Out_image = frame
	Lanes_combined = cv2.bitwise_or(Outer_Lane,Mid_lane)
	cnts = cv2.findContours(Lanes_combined,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[1]

	if cnts:
		#MODED# Traj_lowP,Traj_upP = LanePoints(Mid_lane,Outer_Lane,Mid_cnts,Outer_cnts,NonExtendedLane)
		Traj_lowP,Traj_upP = LanePoints(Mid_lane,Outer_Lane,Offset_correction)

		PerpDist_ImgCen_CarNose= -1000
		if(Traj_lowP!=(0,0)):
			PerpDist_ImgCen_CarNose = Traj_lowP[0] - int(Out_image.shape[1]/2)

		curvature = findlaneCurvature(Traj_lowP[0],Traj_lowP[1],Traj_upP[0],Traj_upP[1])


		Distance = PerpDist_ImgCen_CarNose
		Curvature = int(curvature)

		return Distance,Curvature
	else:
		return -1000,-1000


def KeepOutLane(OuterLane,Outer_cnts,Mid_cnts,MaxIntrv,CurrIntrv,SumDistFromMid,AvgDistFromMid,CurrDistFromMid):
	if(Mid_cnts and Outer_cnts):
		Mid_cnts_Rowsorted = Cord_Sort(Mid_cnts,"rows")
		Mid_Rows = Mid_cnts_Rowsorted.shape[0]
		Mid_lastPt = Mid_cnts_Rowsorted[Mid_Rows-1,:]

		Outer_cnts_Rowsorted = Cord_Sort(Outer_cnts,"rows")
		Out_Rows = Outer_cnts_Rowsorted.shape[0]
		Outer_lastPt = Outer_cnts_Rowsorted[Out_Rows-1,:]

		OM_vec = Outer_lastPt - Mid_lastPt

		OM_vec_mag = math.sqrt( ( (OM_vec[0])**2 ) + ( (OM_vec[1])**2 ) )
		OM_Unit_vec = OM_vec / OM_vec_mag

		CurrDistFromMid = math.atan2(OM_Unit_vec[1],OM_Unit_vec[0])*180/math.pi# math.atan2(y(rows),x(cols))
		
		print(CurrDistFromMid)
		if(CurrIntrv == MaxIntrv):
			CurrIntrv = 0 #Reset
			AvgDistFromMid = SumDistFromMid / MaxIntrv
			SumDistFromMid = 0 # Reset sum of angles
			if(abs(CurrDistFromMid - AvgDistFromMid) > 30):
				Outer_cnts = [] # Empty Contours
				OuterLane = np.zeros_like(OuterLane) # Empty the pic
		else:
			print(AvgDistFromMid)
			print(CurrIntrv)
			if( AvgDistFromMid!=0 ):
				if(abs(CurrDistFromMid - AvgDistFromMid) > 30):
					Outer_cnts = [] # Empty Contours
					OuterLane = np.zeros_like(OuterLane) # Empty the pic

			CurrIntrv = CurrIntrv + 1 #Swap to next interval
			SumDistFromMid = SumDistFromMid + CurrDistFromMid
		
	return OuterLane,Outer_cnts,CurrIntrv,SumDistFromMid,AvgDistFromMid,CurrDistFromMid
		

def main():
	#cv2.namedWindow("Mid_trajectory",cv2.WINDOW_NORMAL)
	#cv2.namedWindow("Mid_trajectory_largest",cv2.WINDOW_NORMAL)
	#cv2.namedWindow("OuterLane_trajectory",cv2.WINDOW_NORMAL)
	#cv2.namedWindow("Out_image",cv2.WINDOW_NORMAL)

	
	Ref_imgWidth = 1920
	Ref_imgHeight = 1080
	Frame_pixels = Ref_imgWidth * Ref_imgHeight

	Resized_width = 640 # Control Parameter
	Resized_height = 480
	Resize_Framepixels = Resized_width * Resized_height

	Lane_Extraction_minArea_per = 1500 / Frame_pixels
	minArea_resized = int(Resize_Framepixels * Lane_Extraction_minArea_per)

	#BWContourOpen_speed_MaxDist_per = 500 / cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
	BWContourOpen_speed_MaxDist_per = 500 / Ref_imgHeight
	MaxDist_resized = int(Resized_height * BWContourOpen_speed_MaxDist_per)
	
	CropHeight = 700
	#CropHeight_resized_crop = int( (CropHeight / cap.get(cv2.CAP_PROP_FRAME_HEIGHT) ) * Resized_height )
	CropHeight_resized_crop = int( (CropHeight / Ref_imgHeight ) * Resized_height )

	if(config.write):
		#in_q = cv2.VideoWriter('LaneDetection/Results/in2.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (Resized_width,Resized_height))
		out = cv2.VideoWriter('LaneDetection/Results/out.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (Resized_width,Resized_height))	

	loopCount=0

	if (config.debugging==False):
		vs = PiVideoStream((640,480),30).start()
		time.sleep(2.0)
		waitTime = 1
	else:
		cv2.createTrackbar('Video_pos','Vid',Video_pos,duration,OnVidPosChange)
		LaneDetection_results = open(txt_path,"w")
		result_txt =  "Detected_Frame -> [ Distance , Curvature ] [avg_Dist_4]\n" 
		LaneDetection_results.write(result_txt)		
		detected_frame_count = 0
		waitTime = 0
		# Averaging Distance Control Parameters
		avg_Dist_4 = 0
		temp_dist = 0
		num_Dist = 4

		# Idea is to stick to side yellow lane that you were following for some intervals
		# For M intervals track on which side of mid lane did yellow lane exist
		# Do this by distance (midlane,YellowLane) for M intervals 
		# Once Mth interval come the average of yellow lane describe if we look at left side or right side
		# The other side gets oblierated before extendingLane And rest follows as before
		YelloLane_maxIntrv = 20
		YelloLane_CurrIntrv = 0
		YelloLane_SumDistFromMid = 0
		YelloLane_AvgDistFromMid = 0
		YelloLane_CurrDistFromMid = 0
	while (1):
        
		start = time.time()
		if(config.debugging):
			ret, frame = cap.read()# 6 ms
			frame = cv2.resize(frame,(Resized_width,Resized_height))
		else:
			frame = vs.read().copy()
		
		frame_cropped = frame[CropHeight_resized_crop:,:]

		#if(config.write):
			#in_q.write(frame)

		if 1:
			if (config.debugging):
				cv2.imshow('Vid',frame)
			start_getlanes = time.time()
			# Extracting Outer and Middle lanes using colour Segmentation in HSL mode
			Mid_edge_ROI,Mid_ROI_mask,Outer_edge_ROI,_,OuterLane_TwoSide,OuterLane_Points = GetLaneROI(frame_cropped,minArea_resized)#64 ms
			end_getlanes = time.time()
			print("--> GetLane Loop took ",end_getlanes - start_getlanes," sec <-->  ",(1/(end_getlanes - start_getlanes)),"  FPS ")

			# Using polyfit to estimate the trajectory of outer lane
			Mid_trajectory = SegmentLanesAndDisplay_2(Mid_edge_ROI)#13ms

			# Keeping only the Estimated trajectory upto the ROI we decided
			Mid_trajectory = cv2.bitwise_and(Mid_trajectory,Mid_ROI_mask)# 0.1ms
			
			# Remove small contours and keep only largest
			Mid_trajectory_largest = BWContourOpen_speed(Mid_trajectory,MaxDist_resized)#13msec

			#cv2.imshow('Mid_trajectory',Mid_trajectory)
			#cv2.imshow('Mid_trajectory_largest',Mid_trajectory_largest)
			

			# Once we have the mid and outer lanes Only keep the closest outer lane to mid lane
			OuterLane_OneSide,Outer_cnts_oneSide,Mid_cnts,Offset_correction = FindClosestLane(OuterLane_TwoSide,Mid_trajectory_largest,OuterLane_Points)#3ms

			# Algo to Remove/Deselect YellowLine that doesnt support the general trend
			#OuterLane_OneSide,Outer_cnts_oneSide,YelloLane_CurrIntrv,YelloLane_SumDistFromMid,YelloLane_AvgDistFromMid,YelloLane_CurrDistFromMid = KeepOutLane(OuterLane_OneSide,Outer_cnts_oneSide,Mid_cnts,YelloLane_maxIntrv,YelloLane_CurrIntrv,YelloLane_SumDistFromMid,YelloLane_AvgDistFromMid,YelloLane_CurrDistFromMid)
			#if (config.debugging):
			#	cv2.imshow('OuterLane_OneSide_Sanitized',OuterLane_OneSide)
	
			# Extend Short Lane (Either Mid or Outer)So that both has same location of foot
			#Mid_trajectory_largest,OuterLane_OneSide, NonExtendedLane = ExtendShortLane(Mid_trajectory_largest,Mid_cnts,Outer_cnts_oneSide,OuterLane_OneSide)#3ms
			Mid_trajectory_largest,OuterLane_OneSide = ExtendShortLane_(Mid_trajectory_largest,Mid_cnts,Outer_cnts_oneSide,OuterLane_OneSide)

			Mid_trajectory = cv2.bitwise_and(Mid_trajectory,Mid_trajectory_largest)
			Mid_edge_ROI = RefineMidEdgeROi(Mid_trajectory,Mid_ROI_mask,Mid_edge_ROI)#6ms
			#cv2.imshow('Mid_edge_ROI',Mid_edge_ROI)
			
			# Using Both outer and middle information to create probable path and extract
			if config.Testing:
				Distance , Curvature = DrawProbablePath(OuterLane_OneSide,Mid_trajectory_largest,Mid_cnts,Outer_cnts_oneSide,Mid_edge_ROI,frame_cropped,Offset_correction)#20ms
				
				if(Distance != -1000 | Curvature != -1000):
					detected_frame_count = detected_frame_count + 1
					temp_dist = temp_dist + Distance
					if ( (detected_frame_count % num_Dist) == 0 ):
						avg_Dist_4 = int(temp_dist / num_Dist)
						temp_dist = 0
					result_txt =  str(detected_frame_count) + " -> [ "+ str(Distance) + " , " + str(Curvature) + " ]  -> [ "+ str(avg_Dist_4) + " ] \n" 
					LaneDetection_results.write(result_txt)
					if (config.debugging==False):		
						beInLane(int(frame.shape[1]/4), Distance,Curvature )
						
					
				
				Out_image,drawn = DrawProbablePath_(OuterLane_OneSide,Mid_trajectory_largest,Mid_cnts,Outer_cnts_oneSide,Mid_edge_ROI,frame_cropped,Offset_correction)#20ms
				cv2.imshow('frame',frame)
				k = cv2.waitKey(waitTime)
				if k==27:
					break
			else:
				Distance , Curvature = DrawProbablePath(OuterLane_OneSide,Mid_trajectory_largest,Mid_cnts,Outer_cnts_oneSide,Mid_edge_ROI,frame_cropped,Offset_correction)#20ms
				if(Distance != -1000 | Curvature != -1000):
					if (config.debugging==False):
						beInLane(int(frame.shape[1]/4), Distance,Curvature )
						# -1000 value in either of Dist or Curvature represents Unknown value (Lane not found)
						# + Distance represents Car On the right side of image center + x axis 
						# - Distance represents Car On the left  side of image center - x axis
						# + Curvature represents orientation of car w.r.t image central axis (Red Vertical line) 
						# - Curvature represents orientation of car w.r.t image central axis
			if(config.write):
				out.write(frame)#8ms
			if config.Profiling:
				loopCount=loopCount+1
				if(loopCount==50):
					break
		else:
			if(config.write):
				cv2.imshow("data",frame)
				cv2.waitKey(1)
			else:
				break
		end = time.time()
		print("Complete Loop took ",end - start," sec <-->  ",(1/(end - start)),"  FPS ")
	end = time.time()
	print("Complete Loop took ",end - start," sec <-->  ",(1/(end - start)),"  FPS ")	
	# When everything done, release the video capture and video write objects
	LaneDetection_results.close()
	cap.release()
	out.release()
	# Closes all the frames
	cv2.destroyAllWindows()
	if (config.debugging==False):
		vs.stop() 
	#exit()	


if __name__ == '__main__':
	main()

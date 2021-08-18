import config
import cv2
import os
import numpy as np 
import time
from Detection.Lanes.utilities import findLineParameter,findlaneCurvature,Distance_
from Detection.Lanes.Lane_Extractor import GetLaneROI
from Detection.Lanes.Morph_op import FindExtremas,BWContourOpen_speed


def Cord_Sort(cnts,order):

    if cnts:
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
    else:
        return cnts

def IsPathCrossingMid(Midlane,Mid_cnts,Outer_cnts):

	is_Ref_to_path_Left = 0
	Ref_To_Path_Image = np.zeros_like(Midlane)
	Midlane_copy = Midlane.copy()

	Mid_cnts_Rowsorted = Cord_Sort(Mid_cnts,"rows")
	Outer_cnts_Rowsorted = Cord_Sort(Outer_cnts,"rows")
	#print(Mid_cnts_Rowsorted)
	if not Mid_cnts:
		print(" MidCnts Empty!!!!")
	Mid_Rows = Mid_cnts_Rowsorted.shape[0]
	Outer_Rows = Outer_cnts_Rowsorted.shape[0]

	Mid_lowP = Mid_cnts_Rowsorted[Mid_Rows-1,:]
	Outer_lowP = Outer_cnts_Rowsorted[Outer_Rows-1,:]

	Traj_lowP = ( int( (Mid_lowP[0] + Outer_lowP[0]  ) / 2 ) , int( (Mid_lowP[1]  + Outer_lowP[1] ) / 2 ) )
	
	#cv2.line(Ref_To_Path_Image,Traj_lowP,(int(Ref_To_Path_Image.shape[1]/2),Traj_lowP[1]),(255,255,0),2)# distance of car center with lane path
	#cv2.line(Ref_To_Path_Image,(Traj_lowP[0],Ref_To_Path_Image.shape[0]),(int(Ref_To_Path_Image.shape[1]/2),Ref_To_Path_Image.shape[0]),(255,255,0),2)# distance of car center with lane path
	cv2.line(Ref_To_Path_Image,Traj_lowP,(int(Ref_To_Path_Image.shape[1]/2),Ref_To_Path_Image.shape[0]),(255,255,0),2)# distance of car center with lane path
	cv2.line(Midlane_copy,tuple(Mid_lowP),(Mid_lowP[0],Midlane_copy.shape[0]-1),(255,255,0),2)# distance of car center with lane path

	is_Ref_to_path_Left = ( (int(Ref_To_Path_Image.shape[1]/2) - Traj_lowP[0]) > 0 )
	Distance_And_Midlane = cv2.bitwise_and(Ref_To_Path_Image,Midlane_copy)

	if( np.any( (cv2.bitwise_and(Ref_To_Path_Image,Midlane_copy) > 0) ) ):
		# Midlane and CarPath Intersets (MidCrossing)
		return True,is_Ref_to_path_Left
	else:
		return False,is_Ref_to_path_Left

def FindClosestLane(OuterLanes,MidLane,OuterLane_Points):
	#  Fetching the closest outer lane to mid lane is the main goal here
	if (config.debugging):
		cv2.imshow("[C_Cleaning OuterLanes Before",OuterLanes)
	#Container for storing/returning closest Outer Lane

	Offset_correction = 0

	Outer_Lanes_ret= np.zeros(OuterLanes.shape,OuterLanes.dtype)
	
	Mid_cnts = cv2.findContours(MidLane, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
	Outer_cnts = cv2.findContours(OuterLanes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

	Ref=(0,0)
	if(Mid_cnts):
		Ref = tuple(Mid_cnts[0][0][0])

	if not Outer_cnts:
		NoOuterLane_before=True
	else:
		NoOuterLane_before=False
	#Condition 1 : if Both Midlane and Outlane is detected
	# Sepearate closest outlane to the midlane

	if  ( Mid_cnts and (len(OuterLane_Points)==2)):
		Point_a=OuterLane_Points[0]
		Point_b=OuterLane_Points[1]

		Closest_Index=0
		if(Distance_(Point_a,Ref)<=Distance_(Point_b,Ref)):
			Closest_Index=0
		elif(len(Outer_cnts)>1):
			Closest_Index=1

		Outer_Lanes_ret = cv2.drawContours(Outer_Lanes_ret, Outer_cnts, Closest_Index, 255, 1)
		Outer_cnts_ret = [Outer_cnts[Closest_Index]]
		# ================================ Adding the nEW stuff =====================================
		# The idea is to find lane points here and determine if trajectory is crossing midlane
		#If (Yes):
		# Discard
		#Else 
		# Continue
		IsPathCrossing , IsCrossingLeft = IsPathCrossingMid(MidLane,Mid_cnts,Outer_cnts_ret)
		if(IsPathCrossing):
			if(config.debugging):
				print("[FindClosestLane] [(len(OuterLane_Points)==2)] Zeroing OuterLanes because LAnes are crossing")
			OuterLanes = np.zeros_like(OuterLanes)#Empty outerLane
		else:
			#If no fllor crossing return results
			return Outer_Lanes_ret ,Outer_cnts_ret, Mid_cnts,0
	elif( Mid_cnts and np.any(OuterLanes>0) ):
		# Midlane and OuterLanes are both present but there was no OuterLanePoints Were not 2
		IsPathCrossing , IsCrossingLeft = IsPathCrossingMid(MidLane,Mid_cnts,Outer_cnts)
		if(IsPathCrossing):
			if (config.debugging):
				print("[FindClosestLane] [np.any(OuterLanes>0)] Zeroing OuterLanes because LAnes are crossing")
			OuterLanes = np.zeros_like(OuterLanes)#Empty outerLane
		else:
			if (config.debugging):
				print("[FindClosestLane] [np.any(OuterLanes>0)] Path are not crossing --> Ret as it is")
			#If no fllor crossing return results
			return OuterLanes ,Outer_cnts, Mid_cnts,0		

	#Condition 2 : if MidLane is present but no Outlane detected
	# Create Outlane on Side that represent the larger Lane as seen by camera
	if( Mid_cnts and ( not np.any(OuterLanes>0) ) ):
		
		if (config.debugging):
			print("[FindClosestLane] [OuterLanes is Empty] OuterLanes Not empty but points are empty")

		# Condition where MidCnts are detected 
		Mid_cnts_Rowsorted = Cord_Sort(Mid_cnts,"rows")
		Mid_Rows = Mid_cnts_Rowsorted.shape[0]

		Mid_lowP = Mid_cnts_Rowsorted[Mid_Rows-1,:]
		Mid_highP = Mid_cnts_Rowsorted[0,:]

		#Mid_median_Col = int( (Mid_lowP[0]  + Mid_highP[0] ) / 2 ) 
		Mid_low_Col = Mid_lowP[0]
		
		DrawRight = False
		
		if NoOuterLane_before:
			#print("Mid_lowP[0] ",Mid_lowP[0],"Mid_highP[0] ",Mid_highP[0])
			if (config.debugging):
				print("[FindClosestLane] [OuterLanes is Empty] No OuterLanes were detected at all so can only rely on Midlane Info!!")
			#if(Mid_median_Col < int(MidLane.shape[1]/2)):
			if(Mid_low_Col < int(MidLane.shape[1]/2)):
				# MidLane on left side of Col/2 of image --> Bigger side is right side draw there
				DrawRight = True
		else:
			if (config.debugging):
				print("[FindClosestLane] IsPathCrossing = ",IsPathCrossing," IsCrossingLeft = ",IsCrossingLeft)
			if IsCrossingLeft:
				# trajectory from reflane to lane path is crossing midlane while moving left --> Draw Right
				DrawRight = True
		if (config.debugging):
			print("[FindClosestLane] [OuterLanes is Empty] DrawRight = ",DrawRight)
		if not DrawRight:
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

		#print(" Mid_lower_row = ", Mid_lowP[1])
		#print(" Mid_higher_row = ", Mid_highP[1])
		OuterLanes = cv2.line(OuterLanes,LanePoint_lower,LanePoint_top,255,1)		
		Outer_cnts = cv2.findContours(OuterLanes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
		return OuterLanes, Outer_cnts, Mid_cnts, Offset_correction
	else:
		return OuterLanes, Outer_cnts, Mid_cnts, Offset_correction

def ExtendBothLane(Lane,OuterLane,Lane_RowSorted,RefLane_RowSorted):

	Image_bottom = Lane.shape[0]
	
	Lane_Rows = Lane_RowSorted.shape[0]
	Lane_Cols = Lane_RowSorted.shape[1]
	BottomPoint = Lane_RowSorted[Lane_Rows-1,:]	

	if (BottomPoint[1] < Image_bottom):
		Lane = cv2.line(Lane,tuple(BottomPoint),(BottomPoint[0],Image_bottom),255)


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

def ExtendShortLane(MidLane,Mid_cnts,Outer_cnts,OuterLane):

	if(Mid_cnts and Outer_cnts):
		Mid_cnts_Rowsorted = Cord_Sort(Mid_cnts,"rows")
		Outer_cnts_Rowsorted = Cord_Sort(Outer_cnts,"rows")

		MidLane,OuterLane = ExtendBothLane(MidLane,OuterLane,Mid_cnts_Rowsorted,Outer_cnts_Rowsorted)

	return MidLane,OuterLane

def RefineMidEdgeROi(Mid_trajectory,Mid_ROI_mask,Mid_edge_ROI):

	contours_trajectory = cv2.findContours(Mid_trajectory,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[1]
	contours = cv2.findContours(Mid_ROI_mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[1]
	Mid_ROI_mask_refined = np.zeros(Mid_ROI_mask.shape,Mid_ROI_mask.dtype)

	Modified= False
	for index, cnt in enumerate(contours):
		Draw = False
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
	Non_Mid_Mask=cv2.bitwise_not(Mid_Hull_Mask)
	return Non_Mid_Mask

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

def DrawProbablePath(Outer_Lane,Mid_lane,Mid_cnts,Outer_cnts,MidEdgeROi,frame,Offset_correction):
	Lanes_combined = cv2.bitwise_or(Outer_Lane,Mid_lane)
	#cv2.namedWindow("Lanes_combined",cv2.WINDOW_NORMAL)
	if config.Testing:
		cv2.imshow("[D_DataExt Lanes_combined]",Lanes_combined)
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
		cv2.imshow("[D_DataExt ProjectedLane]",ProjectedLane)

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
		textSize_ratio = 0.5
		cv2.putText(Out_image,texttoPut,(10,30),cv2.FONT_HERSHEY_DUPLEX,textSize_ratio,(0,255,255),1)
		cv2.putText(Out_image,texttoPut2,(10,50),cv2.FONT_HERSHEY_DUPLEX,textSize_ratio,(0,255,255),1)
		
		#Out_image = cv2.resize(Out_image,(1280,720))
		drawn=True		
		return Out_image,drawn
	else:
		return Outer_Lane,drawn

def fetch_LaneInformation(Outer_Lane,Mid_lane,Mid_cnts,Outer_cnts,MidEdgeROi,frame,Offset_correction):

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
	
	
def Detect_Lane(frame):

	Distance = -1000
	Curvature = -1000
	frame_cropped = frame[config.CropHeight_resized_crop:,:]
			
	if config.detect:
			
		start_getlanes = time.time()
        # Extracting Outer and Middle lanes using colour Segmentation in HSL mode
		Mid_edge_ROI,Mid_ROI_mask,Outer_edge_ROI,OuterLane_TwoSide,OuterLane_Points = GetLaneROI(frame_cropped,config.minArea_resized)#64 ms
		end_getlanes = time.time()
		print("[Profiling] GetLane Loop took ",end_getlanes - start_getlanes," sec <-->  ",(1/(end_getlanes - start_getlanes+0.00001)),"  FPS ")

        # Using polyfit to estimate the trajectory of outer lane
		Mid_trajectory = Mid_edge_ROI.copy()
		# Keeping only the Estimated trajectory upto the ROI we decided
		Mid_trajectory = cv2.bitwise_and(Mid_trajectory,Mid_ROI_mask)# 0.1ms
		# 3. Dilating Segmented ROI's
		kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(3,3))
		Mid_trajectory = cv2.morphologyEx(Mid_trajectory, cv2.MORPH_DILATE, kernel)
		Mid_trajectory = cv2.morphologyEx(Mid_trajectory, cv2.MORPH_ERODE, kernel)
		#if(config.debugging):
			#cv2.imshow('Mid_ROI_mask',Mid_ROI_mask)
		# Remove small contours and keep only largest
		Mid_trajectory_largest = BWContourOpen_speed(Mid_trajectory,config.MaxDist_resized)#13msec
		if(config.debugging):
			cv2.imshow('[B_Est Mid_trajectory Edge]',Mid_trajectory)
			cv2.imshow('[B_Est Mid_trajectory_Estimated]',Mid_trajectory_largest)
		
		# Once we have the mid and outer lanes Only keep the closest outer lane to mid lane
		OuterLane_OneSide,Outer_cnts_oneSide,Mid_cnts,Offset_correction = FindClosestLane(OuterLane_TwoSide,Mid_trajectory_largest,OuterLane_Points)#3ms

        # Extend Short Lane (Either Mid or Outer)So that both has same location of foot
		Mid_trajectory_largest,OuterLane_OneSide = ExtendShortLane(Mid_trajectory_largest,Mid_cnts,Outer_cnts_oneSide,OuterLane_OneSide)
		cv2.imshow('[C_Cleaning OuterLane_Cleaned]',OuterLane_OneSide)		
		Mid_trajectory = cv2.bitwise_and(Mid_trajectory,Mid_trajectory_largest)

		#Mid_edge_ROI = RefineMidEdgeROi(Mid_trajectory,Mid_ROI_mask,Mid_edge_ROI)#6ms
		Mid_edge_ROI = Mid_trajectory.copy()
		#if(config.debugging):
			#cv2.imshow('Mid_edge_ROI after',Mid_edge_ROI)
        # Using Both outer and middle information to create probable path and extract
		if config.Testing:
			Distance , Curvature = fetch_LaneInformation(OuterLane_OneSide,Mid_trajectory_largest,Mid_cnts,Outer_cnts_oneSide,Mid_edge_ROI,frame_cropped,Offset_correction)#20ms
			
			Out_image , drawn = DrawProbablePath(OuterLane_OneSide,Mid_trajectory_largest,Mid_cnts,Outer_cnts_oneSide,Mid_edge_ROI,frame_cropped,Offset_correction)#20ms
			#cv2.imshow('frame',frame)
			#k = cv2.waitKey(config.waitTime)
		else:
			Distance , Curvature = fetch_LaneInformation(OuterLane_OneSide,Mid_trajectory_largest,Mid_cnts,Outer_cnts_oneSide,Mid_edge_ROI,frame_cropped,Offset_correction)#20ms
		
	else:
		cv2.imshow("frame",frame)
		cv2.waitKey(100)
	
	return Distance,Curvature
import cv2
import os
from Detection.Lanes.a_Segmentation.colour_segmentation_final import Segment_Colour


def main():

    Relative_vid_path = "SW Modular Scripts/data/vids/Lane_vid.avi"
    cap = cv2.VideoCapture(os.path.abspath(Relative_vid_path))
    waitTime = 1

    while(1):
        ret,img = cap.read()
        if not ret:
            print("vid Not found")
            break
        else:
            print("### Vid Found ###")


        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>   TESTING   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        cv2.imshow("Frame",img)
        k = cv2.waitKey(waitTime)
        if k==27:
            break

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>   TESTING   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


if __name__ == '__main__':
	main()
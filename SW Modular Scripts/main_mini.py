import cv2
import os



def main():
    cap = cv2.VideoCapture(os.path.abspath("data/vids/signs_forward.mp4"))
    waitTime = 0

    while(1):
        ret,img = cap.read()
        if ret:
            img = cv2.resize(img,(320,240))
        else:
            break

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>   TESTING   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



        #cv2.imshow("Frame",img)
        k = cv2.waitKey(waitTime)
        if k==27:
            break

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>   TESTING   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


if __name__ == '__main__':
	main()
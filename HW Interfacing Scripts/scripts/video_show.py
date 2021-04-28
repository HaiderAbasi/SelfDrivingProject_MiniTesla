from imutils.video.pivideostream import PiVideoStream
import cv2

def main():
    video_in = cv2.VideoCapture(0)

    waitTime = 1
    while(1):
        _,frame = video_in.read()
        cv2.imshow("frame",frame)
        cv2.waitKey(1)

if __name__ == '__main__':
	main()
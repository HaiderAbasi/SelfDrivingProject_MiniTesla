from imutils.video.pivideostream import PiVideoStream
import cv2


def main():
    VideoCapture_Pi = PiVideoStream().start()

    while 1:
        frame = VideoCapture_Pi.read().copy()
        cv2.imshow("Live", frame)
        cv2.waitKey(0)



if __name__ == '__main__':
	main()
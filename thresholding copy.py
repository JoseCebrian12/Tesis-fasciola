import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import matplotlib.pyplot as plt

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
# allow the camera to warmup
time.sleep(0.1)
# initialize variables
img_counter = 0

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array
    cv2.imshow("Frame", image)
    # wait for user input
    key = cv2.waitKey(1) & 0xFF

    # if user presses spacebar, capture the frame
    if key == ord(' '):
        # create a filename for the captured image
        img_name = "images/opencv_frame_{}.png".format(img_counter)
        img_counter += 1

        # save the image
        cv2.imwrite(img_name, image)
        
        # read the image in grayscale
        img = cv2.imread(img_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)

        # circle detection
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                                   param1=50, param2=30, minRadius=0, maxRadius=0)
        
        # ensure at least some circles were found
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(image, (x, y), r, (0, 255, 0), 4)
        
        # display the image
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.tight_layout()
        plt.show()

    elif key == ord('q'):
        break

    rawCapture.truncate(0)
cv2.destroyAllWindows()


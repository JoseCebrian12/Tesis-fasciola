import cv2
import urllib.request
import matplotlib.pyplot as plt
import numpy as np

# define the URL and open the stream
url = 'http://192.168.148.63:81/stream'
cap = cv2.VideoCapture(url)

# initialize variables
img_counter = 0

# start reading frames from the stream
while True:
    ret, frame = cap.read()

    # display the frame
    cv2.imshow('frame', frame)

    # wait for user input
    key = cv2.waitKey(1) & 0xFF

    # if user presses spacebar, capture the frame
    if key == ord(' '):
        # create a filename for the captured image
        img_name = "images/opencv_frame_{}.png".format(img_counter)
        img_counter += 1

        # save the image
        cv2.imwrite(img_name, frame)
        print(f"{img_name} saved")

        # read the image in grayscale
        img = cv2.imread(img_name)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # apply median filter to reduce noise
        img_gray = cv2.medianBlur(img_gray, 5)

        # apply adaptive thresholding to binarize the image
        thresh_img = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # apply morphological closing to refine edges
        kernel = np.ones((5, 5), np.uint8)
        thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel)

        # find the contours in the image
        contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # select the contours within a specific size range
        min_area = 500
        max_area = 5000
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area and area < max_area:
                valid_contours.append(contour)
        largest_contour_idx = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
        mask = np.zeros_like(img_gray)

        cv2.drawContours(mask, [contours[largest_contour_idx]], 0, (255, 255, 255), -1)
        masked_img = cv2.bitwise_and(img, img, mask=mask)

        plt.imshow(masked_img[...,::-1])
        plt.show()

        

    elif key == ord('q'):
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
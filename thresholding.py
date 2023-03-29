import cv2
import urllib.request
import matplotlib.pyplot as plt
import numpy as np

url = 'http://192.168.156.63:81/stream'
cap = cv2.VideoCapture(url)
img_counter = 0

while True:
    ret,frame = cap.read()
    cv2.imshow('frame', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '): 
        img_name = "images/opencv_frame_{}.png".format(img_counter)
        img_counter += 1
        cv2.imwrite(img_name, frame)


        img = cv2.imread(img_name)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, thresh_img = cv2.threshold(img_gray, 123, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
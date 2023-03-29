import cv2
import numpy as np
import matplotlib.pyplot as plt

img_name = "images/opencv_frame_0.png"
img = cv2.imread(img_name)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh_img = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest_contour_idx = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
mask = np.zeros_like(img_gray)

cv2.drawContours(mask, [contours[largest_contour_idx]], 0, (255, 255, 255), -1)
masked_img = cv2.bitwise_and(img, img, mask=mask)

plt.imshow(masked_img[...,::-1])
plt.show()
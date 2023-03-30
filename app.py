import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np

st.title("My first Streamlit app")
st.write("Hello, world")


def callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_gray = cv2.medianBlur(img_gray, 5)

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
    
    return av.VideoFrame.from_ndarray(masked_img, format="bgr24")


webrtc_streamer(key="example", video_frame_callback=callback)
import streamlit as st
import cv2
import numpy as np

vid = cv2.VideoCapture( 'http://192.168.248.63:81/stream' )
st.markdown("""
    <style>
    .reportview-container {
        background: white;
    }
    
    .image-corner {
        position: absolute;
        top: 10px;
        right: 10px;
        width: 100px;
        height: 100px;
    }
    </style>
""", unsafe_allow_html=True)

st.title('Proyecto Detección Fasciola Hepática - Versión 1.0')
frame_window = st.image([])

# Agregar imagen en la esquina superior derecha
st.image('/home/popin/Documentos/Tesis_code/images/Logo_Oficial.png', 
         use_column_width=False, 
         width=100,  
         output_format='PNG')

take_picture_button = st.button('Take Picture')
img_counter = 0
while True:
    got_frame , frame = vid.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame_gray = cv2.cvtColor( frame , cv2.COLOR_BGR2GRAY )
    img_gray = cv2.medianBlur(frame_gray, 5)

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
    masked_img = cv2.bitwise_and(frame, frame, mask=mask)
    if got_frame:
        frame_window.image(masked_img)

    if take_picture_button:
        img_name = "images/opencv_frame_{}.png".format(img_counter)
        img_counter += 1

        # save the image
        cv2.imwrite(img_name, frame)
        print(f"{img_name} saved")
        break

vid.release()
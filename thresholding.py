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
    #wait for user input
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
        output = img.copy()
        # Convierte la imagen a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray,(5,5),0)

        # Realiza la transformada de Hough para 
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=60, param1=50, param2=30, minRadius=30, maxRadius=60)

        # Convierte la imagen de BGR a HSV
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define el rango del color amarillo en el espacio de color HSV
        bajo_amarillo = np.array([0, 0, 0])
        alto_amarillo = np.array([255, 255, 255])

        # Inicia una   para los  cuyo centroide  en el rango de amarillos
        mascara_final = np.zeros_like(gray)

        # Verifica si se encontraron algunos 
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # Crea una  para el 
                mascara_circulo = np.zeros_like(gray)
                cv2.circle(mascara_circulo, (x, y), r, 255, -1)

                #  el color HSV del centroide del 
                h, s, v = img_hsv[y, x]

                # Verifica si el color del centroide  en el rango de amarillos
                if bajo_amarillo[0] <= h <= alto_amarillo[0] and bajo_amarillo[1] <= s <= alto_amarillo[1] and bajo_amarillo[2] <= v <= alto_amarillo[2]:
                    # Si el color del centroide  en el rango de amarillos, agrega el  a la  final
                    mascara_final = cv2.bitwise_or(mascara_final, mascara_circulo)
                    
                    #  el color BGR del centroide del 
                    b, g, r = img[y, x]
                    
                    # Convierte el color a hexadecimal
                    color_hex = '#%02x%02x%02x' % (r, g, b)
                    
                    # Dibuja el color en el centroide del 
                    cv2.putText(output, color_hex, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Aplica la  final a la imagen original
        img_final = cv2.bitwise_and(output, output, mask=mascara_final)

        # Muestra la imagen final
        plt.subplots(1,2,figsize=(20,10),sharey=True)
        plt.subplot(1,2,1)
        plt.imshow(img[:, :, [2, 1, 0]])
        plt.subplot(1,2,2)
        plt.imshow(cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB))
        plt.tight_layout()
        plt.show()

        

    elif key == ord('q'):
        break

    rawCapture.truncate(0)
cv2.destroyAllWindows()
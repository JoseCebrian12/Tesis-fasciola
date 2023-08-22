import cv2
import numpy as np
import matplotlib.pyplot as plt

# Lee la imagen
img = cv2.imread("/home/josecebrian12/fasciola_project/images/opencv_frame_3.png")
output = img.copy()

# Convierte la imagen a escala de grises
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Aplica un suavizado Gaussiano para reducir el ruido y mejorar la detección de los círculos
gray = cv2.GaussianBlur(gray,(5,5),0)

# Realiza la transformada de Hough para círculos
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=60, param1=50, param2=30, minRadius=30, maxRadius=60)

# Convierte la imagen de BGR a HSV
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define el rango del color amarillo en el espacio de color HSV
#bajo_amarillo = np.array([20, 100, 100])
#alto_amarillo = np.array([30, 255, 255])
bajo_amarillo = np.array([0, 0, 0])
alto_amarillo = np.array([255, 255, 255])
# Inicia una máscara vacía para los círculos cuyo centroide está en el rango de amarillos
mascara_final = np.zeros_like(gray)

# Verifica si se encontraron algunos círculos
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        # Crea una máscara para el círculo
        mascara_circulo = np.zeros_like(gray)
        cv2.circle(mascara_circulo, (x, y), r, 255, -1)

        # Obtén el color HSV del centroide del círculo
        h, s, v = img_hsv[y, x]

        # Verifica si el color del centroide está en el rango de amarillos
        if bajo_amarillo[0] <= h <= alto_amarillo[0] and bajo_amarillo[1] <= s <= alto_amarillo[1] and bajo_amarillo[2] <= v <= alto_amarillo[2]:
            # Si el color del centroide está en el rango de amarillos, agrega el círculo a la máscara final
            mascara_final = cv2.bitwise_or(mascara_final, mascara_circulo)
            
            # Obtén el color BGR del centroide del círculo
            b, g, r = img[y, x]
            
            # Convierte el color a hexadecimal
            color_hex = '#%02x%02x%02x' % (r, g, b)
            
            # Dibuja el color en el centroide del círculo
            cv2.putText(output, color_hex, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

# Aplica la máscara final a la imagen original
img_final = cv2.bitwise_and(output, output, mask=mascara_final)

# Muestra la imagen final
plt.subplots(1,2,figsize=(20,10),sharey=True)
plt.subplot(1,2,1)
plt.imshow(img[:, :, [2, 1, 0]])
plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB))
plt.tight_layout()
plt.show()
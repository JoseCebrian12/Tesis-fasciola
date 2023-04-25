import cv2
import urllib.request
import numpy as np

url = 'http://192.168.248.63:81/stream'

# Definir límites para los colores que queremos detectar
color_limits = {
    'rojo': ([0, 70, 50], [10, 255, 255]), # límites inferiores y superiores para rojo en HSV
    'verde': ([35, 70, 70], [85, 255, 255]), # límites inferiores y superiores para verde en HSV
    'azul': ([100, 70, 50], [130, 255, 255]) # límites inferiores y superiores para azul en HSV
}
# Función para detectar el color en el ROI dado
def detect_color(frame, roi):
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    max_area = 0
    max_mask = None
    color_detected = None
    for color_name, (lower, upper) in color_limits.items():
        mask_color = cv2.inRange(hsv_roi, np.array(lower), np.array(upper))
        area = cv2.countNonZero(mask_color)
        if area > max_area:
            max_area = area
            max_mask = mask_color
            color_detected = color_name
    return color_detected

# Capturar video desde la cámara
cap = cv2.VideoCapture(url)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Definir ROI como un cuadrado en el centro de la imagen
    h, w, _ = frame.shape
    roi_size = min(h, w) // 2
    x = (w - roi_size) // 2
    y = (h - roi_size) // 2
    roi = frame[y:y+roi_size, x:x+roi_size]
    # Dibujar rectángulo alrededor del ROI
    cv2.rectangle(frame, (x, y), (x+roi_size, y+roi_size), (0, 255, 0), 2)
    # Mostrar imagen
    cv2.imshow('frame', frame)
    # Esperar a que se presione la barra espaciadora para detectar el color en el ROI
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        color_detected = detect_color(frame, roi)
        print(f"Color detectado: {color_detected}")

    # Salir si se presiona la tecla 'q'
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


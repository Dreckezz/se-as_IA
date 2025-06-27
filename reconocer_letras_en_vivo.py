import cv2
import mediapipe as mp
import joblib
import numpy as np
from collections import deque

# Carga el modelo entrenado y el codificador
modelo = joblib.load("modelo_knn.pkl")
le = joblib.load("label_encoder.pkl")

# Inicializa MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Variables para acumulación de letras
letra_anterior = ""
contador_estabilidad = 0
letra_actual = ""
palabra = ""
cola_letras = deque(maxlen=5)

# Cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            puntos = []
            for lm in hand_landmarks.landmark:
                puntos.extend([lm.x, lm.y, lm.z])

            if len(puntos) == 63:
                pred = modelo.predict([puntos])[0]
                letra = le.inverse_transform([pred])[0]

                # Estabilidad: solo agrega letra si se repite por varios frames
                if letra == letra_anterior:
                    contador_estabilidad += 1
                else:
                    contador_estabilidad = 0

                letra_anterior = letra

                if contador_estabilidad == 30:  # ajusta sensibilidad
                    if letra == "space":
                        palabra += " "
                    elif letra == "del":
                        palabra = palabra[:-1]
                    else:
                        palabra += letra
                    contador_estabilidad = 0

            # Dibujar mano
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Mostrar en pantalla
    cv2.putText(frame, f"Letra actual: {letra_anterior}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, f"Palabra: {palabra}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Reconocimiento de Letras", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC para salir
        break
    elif key == 13:  # Enter para enviar palabra
        print(f"\n📝 Palabra final: {palabra}\n")
        palabra = ""
    elif key == 8:  # Backspace para borrar una letra
        palabra = palabra[:-1]

cap.release()
cv2.destroyAllWindows()

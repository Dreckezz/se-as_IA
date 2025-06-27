import cv2
import mediapipe as mp
import os
import csv

# Ruta al dataset
DATASET_DIR = r"D:\Descargas\IA Proyecto\asl_alphabet_train\asl_alphabet_train"
OUTPUT_CSV = "datos_mediapipe.csv"

# Inicializa MediaPipe Hands con sensibilidad ajustada
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Lista para guardar resultados
data = []

# Recorre carpetas
for label in sorted(os.listdir(DATASET_DIR)):
    label_dir = os.path.join(DATASET_DIR, label)
    if not os.path.isdir(label_dir):
        continue

    print(f"Procesando letra: {label}")

    # Recorre las 3000 imágenes por letra
    for img_name in os.listdir(label_dir)[:3000]:
        img_path = os.path.join(label_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        # Redimensiona para mejorar la detección
        img = cv2.resize(img, (400, 400))

        # Convierte a RGB y procesa con MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        # Si detectó una mano, guarda los 21 puntos
        if result.multi_hand_landmarks:
            puntos = []
            for lm in result.multi_hand_landmarks[0].landmark:
                puntos.extend([lm.x, lm.y, lm.z])

            # Solo guarda si hay 21 puntos (21 * 3 = 63 valores)
            if len(puntos) == 63:
                puntos.append(label)  # Agrega la etiqueta
                data.append(puntos)

# Guarda todo en un archivo CSV
with open(OUTPUT_CSV, mode='w', newline='') as file:
    writer = csv.writer(file)
    header = [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)] + [f"z{i}" for i in range(21)] + ["label"]
    writer.writerow(header)
    writer.writerows(data)

print(f"\n✅ Datos guardados en {OUTPUT_CSV} con {len(data)} muestras.")

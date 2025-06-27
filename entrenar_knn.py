import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# Carga el archivo CSV generado previamente
df = pd.read_csv("datos_mediapipe.csv")

# Separa X (puntos) e y (etiqueta)
X = df.drop("label", axis=1)
y = df["label"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Divide los datos en entrenamiento y prueba (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Entrena el clasificador KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Evalúa el modelo
y_pred = knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Precisión del modelo: {acc:.4f}\n")
print("📊 Reporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Guarda el modelo y el codificador de etiquetas
joblib.dump(knn, "modelo_knn.pkl")
joblib.dump(le, "label_encoder.pkl")
print("\n💾 Modelo guardado como modelo_knn.pkl y label_encoder.pkl")

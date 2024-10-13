# src/preprocessing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    # Cargar el archivo Excel
    df = pd.read_excel(file_path)

    # Convertir la columna 'Estado' en variables numéricas (Alto=1, Bajo=0)
    df['Estado'] = LabelEncoder().fit_transform(df['Estado'])

    # Seleccionar características y variable objetivo
    X = df[['Precio actual', 'Precio final']]  # Solo estas columnas
    y = df['Estado']

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

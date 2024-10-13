# src/svm_model.py
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    # Crear el modelo SVM
    model = SVC()

    # Entrenar el modelo
    model.fit(X_train, y_train)

    # Realizar predicciones
    y_pred = model.predict(X_test)

    # Evaluar el modelo
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return accuracy, report, y_pred

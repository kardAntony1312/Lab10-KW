# src/decision_tree_model.py
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

def train_and_evaluate_decision_tree(X_train, X_test, y_train, y_test):
    # Crear el modelo de Árbol de Decisión
    model = DecisionTreeClassifier(random_state=42)
    
    # Entrenar el modelo
    model.fit(X_train, y_train)
    
    # Realizar predicciones
    y_pred = model.predict(X_test)
    
    # Evaluar el modelo
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Graficar resultados
    plt.scatter(X_test['Precio actual'], X_test['Precio final'], c=y_pred, cmap='coolwarm')
    plt.title('Predicciones Árbol de Decisión')
    plt.xlabel('Precio actual')
    plt.ylabel('Precio final')
    plt.show()

    return accuracy, report

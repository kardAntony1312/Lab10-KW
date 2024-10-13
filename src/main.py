# src/main.py
import matplotlib.pyplot as plt
from preprocessing import load_and_preprocess_data
from svm_model import train_and_evaluate_model
from kmeans_model import apply_kmeans
from decision_tree_model import train_and_evaluate_decision_tree

file_path = '../data/Data10-1.xlsx'

X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)

accuracy_svm, report_svm, y_pred_svm = train_and_evaluate_model(X_train, X_test, y_train, y_test)
print(f'\nAccuracy SVM: {accuracy_svm}')
print(f'Classification Report SVM:\n{report_svm}')

labels_kmeans = apply_kmeans(X_test)

accuracy_tree, report_tree = train_and_evaluate_decision_tree(X_train, X_test, y_train, y_test)
print(f'\nAccuracy Árbol de Decisión: {accuracy_tree}')
print(f'Classification Report Árbol de Decisión:\n{report_tree}')

plt.scatter(X_test['Precio actual'], X_test['Precio final'], c=y_pred_svm, cmap='coolwarm')
plt.title('Predicciones SVM')
plt.xlabel('Precio actual')
plt.ylabel('Precio final')
plt.show()

print("\nEjecución completada.")

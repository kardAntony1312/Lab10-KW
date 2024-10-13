# src/kmeans_model.py
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def apply_kmeans(X, n_clusters=2):
    # Aplicar KMeans con el número de clusters especificado
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    
    # Obtener las etiquetas del cluster
    labels = kmeans.labels_
    
    # Graficar los resultados
    plt.scatter(X['Precio actual'], X['Precio final'], c=labels, cmap='viridis')
    plt.title('K-Means Clustering')
    plt.xlabel('Precio actual')
    plt.ylabel('Precio final')
    plt.show()

    return labels

# titanic_unsupervised.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# 1. Cargar dataset transformado (el que te pasé: titanic_sintetico_transformado.csv)
df = pd.read_csv("titanic_sintetico_transformado.csv")

# 2. Seleccionar las columnas escaladas (z-score)
scaled_cols = [
    "Pclass_scaled", "Sex_scaled", "Age_scaled",
    "Fare_scaled", "SibSp_scaled", "Parch_scaled"
]
X = df[scaled_cols].values

# 3. PCA a 2 componentes
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

# 4. Probar varios valores de k
resultados = []
ks = range(2, 7)

for k in ks:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)

    sil = silhouette_score(X_pca, labels)
    ch = calinski_harabasz_score(X_pca, labels)
    db = davies_bouldin_score(X_pca, labels)

    resultados.append((k, sil, ch, db))

# 5. Elegir el k con mayor Silhouette
best_k, best_sil, _, _ = max(resultados, key=lambda x: x[1])

print("Resultados por k (k, Silhouette, Calinski-Harabasz, Davies-Bouldin):")
for r in resultados:
    print(r)

print(f"\nMejor k según Silhouette: {best_k} (Silhouette = {best_sil:.3f})")

# 6. Entrenar modelo final
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels_final = kmeans_final.fit_predict(X_pca)

# Guardar resultados de PCA + clúster
df_clusters = pd.DataFrame({
    "PassengerId": df["PassengerId"],
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "cluster": labels_final,
    "Survived": df["Survived"]
})

df_clusters.to_csv("titanic_clusters_pca.csv", index=False)

# Gráfica: Silhouette vs k
sil_values = [r[1] for r in resultados]

plt.figure()
plt.plot(list(ks), sil_values, marker="o")
plt.xlabel("Número de clústeres (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score en función de k")
plt.tight_layout()
plt.savefig("fig_silhouette_k.png")
plt.close()

# Gráfica: dispersión PC1 vs PC2 coloreada por clúster
plt.figure()
for c in range(best_k):
    mask = df_clusters["cluster"] == c
    plt.scatter(df_clusters.loc[mask, "PC1"],
                df_clusters.loc[mask, "PC2"],
                label=f"Clúster {c}")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Clusters de pasajeros en el espacio PCA")
plt.legend()
plt.tight_layout()
plt.savefig("fig_pca_clusters.png")
plt.close()
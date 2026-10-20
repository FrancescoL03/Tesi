import numpy as np
import pandas as pd

from scipy.spatial.distance import mahalanobis

import json

#Embedding delle immagini
x = np.load("full_embeddings_artifacts/proj_train_head.npy")
#Etichette dei generatori
y = np.load("full_embeddings_artifacts/y_train.npy")

with open("full_embeddings_artifacts/meta.json", "r") as f:
    meta = json.load(f)

classes = meta["classes"]

#Dizionario, hashmap
centroids = {}

#Calcolo dei centroidi
for c in np.unique(y):
    centroids[classes[c]] = x[y == c].mean(axis=0)


names = list(centroids.keys())
centroid_matrix = np.array(list(centroids.values()))

cov_matrix = np.cov(x, rowvar=False)
#Calcola la matrice di covarianza delle 256 features
#Le righe sono i campioni e le colonne le features
#Otterremmo una matrice (256, 256)

inv_cov_matrix = np.linalg.pinv(cov_matrix)
#Covarianza = tendenza di 2 variabili a variare insieme
#Facciamo pinv invece di inv per costruire una matrice
#più robusta, siccome con inv deve essere perfettamente invertibile

n = len(centroid_matrix)
distance_matrix = np.zeros((n,n))

for i in range(n):
    for j in range(n):
        distance_matrix[i,j] = mahalanobis(
            centroid_matrix[i],
            centroid_matrix[j],
            inv_cov_matrix
        )

df_distances = pd.DataFrame(distance_matrix, index=names, columns=names)
print(df_distances)

#df_distances.to_excel("output.xlsx", index=True)

mean_distances = {}

for i in range(n):
    row = np.delete(distance_matrix[i], i)
    mean_distances[names[i]] = row.mean()

df_mean_distances = pd.DataFrame({
    "Generator": list(mean_distances.keys()),
    "MeanMahalanobisDistance": list(mean_distances.values())
})

df_mean_distances = df_mean_distances.sort_values("MeanMahalanobisDistance")
print(df_mean_distances)

#df_mean_distances.to_excel("output.xlsx", index=True)

print("\n\nMigliori 5")
print(df_mean_distances.head(5))

top_five = df_mean_distances.head(5)["Generator"].tolist()

for gen in top_five:
    i = names.index(gen)

    row = distance_matrix[i].copy()
    row[i] = np.inf

    nearest_idx = np.argsort(row)[:4]

    print(f"\nGeneratore centrale: {gen}")
    for idx in nearest_idx:
        print(f"Vicino: {names[idx]} | Mahalanobis distance: {row[idx]:.6f}")


#REAL
real_idx = names.index("REAL")

real_distances = distance_matrix[real_idx].copy()
real_distances[real_idx] = np.inf

nearest_idx = np.argsort(real_distances)[:5]

print("\n\nTop 5 nearest generator to REAL")
for idx in nearest_idx:
    print(f"{names[idx]} | Mahalanobis distance: {real_distances[idx]:.6f}")
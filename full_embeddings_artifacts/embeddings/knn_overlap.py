import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

import json

#Embedding delle immagini
x = np.load("full_embeddings_artifacts/proj_train_head.npy")
#Etichette dei generatori
y = np.load("full_embeddings_artifacts/y_train.npy")

with open("full_embeddings_artifacts/meta.json", "r") as f:
    meta = json.load(f)

classes = meta["classes"]

k = 10


#COSINE
x_norm = normalize(x, norm="l2")
nn = NearestNeighbors(n_neighbors=k+1, metric=("cosine"))
nn.fit(x_norm)

'''
#EUCLIDEAN
nn = NearestNeighbors(n_neighbors=k+1, metric=("euclidean"))
nn.fit(x)
'''

distances, indices = nn.kneighbors(x)

#Rimozione del punto stesso
indices = indices[:, 1:]
distances = distances[:, :1]

n_classes = len(classes)
overlap_matrix = np.zeros((n_classes, n_classes))

for i in range(len(x)):
    true_class = y[i] #Classe
    neighbor_classes = y[indices[i]]
    #Indice dei 10 vicini del punto 1

    for nc in neighbor_classes:
        overlap_matrix[true_class, nc] += 1

#Conteggio del punti di ogni cluster
class_counts = np.bincount(y)

#Normalizzazione
for c in range(n_classes):
    overlap_matrix[c] = overlap_matrix[c] / (class_counts[c] * k)

df_overlap = pd.DataFrame(overlap_matrix, index=classes, columns=classes)
print(df_overlap)

for c in range(n_classes):
    row = overlap_matrix[c].copy()
    row[c] = -1 

    nearest_idx = np.argsort(row)[::-1][:4]

    print(f"\nCluster: {classes[c]}")
    for idx in nearest_idx:
        print(f"Vicino: {classes[idx]} | overlap: {row[idx]:.6f}")


mean_overlap = {}

for c in range(n_classes):
    row = np.delete(overlap_matrix[c], c)
    mean_overlap[classes[c]] = row.mean()

df_mean_overlap = pd.DataFrame({
        "Generator": list(mean_overlap.keys()),
        "MeanKNNOverlap": list(mean_overlap.values())
})

df_mean_overlap = df_mean_overlap.sort_values("MeanKNNOverlap", ascending=False)
print(df_mean_overlap)

print("\n\nMigliori 5")
print(df_mean_overlap.head(5))

real_idx = classes.index("REAL")

real_row = overlap_matrix[real_idx].copy()
real_row[real_idx] = -1

nearest_idx = np.argsort(real_row)[::-1][:5]

print("\n\nTop 5 nearest generator to REAL")
for idx in nearest_idx:
    print(f"{classes[idx]} | overlap: {real_row[idx]:.6f}")

#df_overlap.to_excel("output.xlsx", index=True)
#df_mean_overlap.to_excel("output.xlsx", index=True)

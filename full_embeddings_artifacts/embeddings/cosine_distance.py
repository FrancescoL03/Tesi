import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_distances
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
    #seleziono solo i punti di x che appartgono alla classe c (maschera booleana)
    #axis=0 media fatta colonna x colonna


names = list(centroids.keys())
centroid_matrix = np.array(list(centroids.values()))

n = len(centroid_matrix)
distance_matrix = np.zeros((n,n))

print("=========================")
print("COSINE DISTANCE")
print("=========================")
'''
Valori vicino allo 0 indicano una somiglianza maggiore,
valori che invece sono vicini all'1 indicano il contrario.
In determinati casi può anche dare valori vicini al 2,
che significa che si ha una "similarità negativa"
Formula: 1 - cosine similarity
'''
distance_matrix = cosine_distances(centroid_matrix)
#Cosine distane gestisce in automatico la normalizzazione

df_distances = pd.DataFrame(distance_matrix, index=names, columns=names)
print(df_distances)

#df_distances.to_excel("output.xlsx", index=True)

mean_distances = {}
n = len(centroid_matrix)

for i in range(n):
    row = np.delete(distance_matrix[i], i) #Rimozione diagonale nulla
    mean_distances[names[i]] = row.mean()

df_mean_distances = pd.DataFrame({
    "Generator": list(mean_distances.keys()),
    "MeanCosineDistance": list(mean_distances.values())
})


df_mean_distances = df_mean_distances.sort_values("MeanCosineDistance")
print(df_mean_distances)

print("\n\nMigliori 5")
print(df_mean_distances.head(5))

#df_mean_distances.to_excel("output.xlsx", index=True)

#Top 5 generatori più centrali
top_five = df_mean_distances.head(5)["Generator"].tolist()

for gen in top_five:
    i = names.index(gen)

    row = distance_matrix[i].copy()

    row[i] = np.inf

    nearest_idx = np.argsort(row)[:4]

    print(f"\nGeneratore centrale: {gen}")
    for idx in nearest_idx:
        print(f" Vicino: {names[idx]} | Cosine Distance: {row[idx]:.6f}")


#REAL
real_idx = names.index("REAL")

real_distances = distance_matrix[real_idx].copy()

real_distances[real_idx] = np.inf

nearest_idx = np.argsort(real_distances)[:5]

print("\n\nTop 5 nearest generator to REAL")
for idx in nearest_idx:
    print(f"{names[idx]} | Cosine Distance: {real_distances[idx]:.6f}")
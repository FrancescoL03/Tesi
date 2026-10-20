import numpy as np
import pandas as pd

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
print("DISTANZA EUCLIDEA")
print("=========================")

for i in range(n):
    for j in range(n):
        distance = np.linalg.norm(centroid_matrix[i] - centroid_matrix[j]) #Normalizzazione in scalare
        distance_matrix[i,j] = distance


df_distances = pd.DataFrame(distance_matrix, index=names, columns=names)
print(df_distances)

#df_distances.to_excel("output.xlsx", index=True)

mean_distances = {}

for i in range(n):
    row = np.delete(distance_matrix[i], i) #Rimozione degli 0
    mean_distances[names[i]] = row.mean()

df_mean_distances = pd.DataFrame({
    "Generator": list(mean_distances.keys()),
    "MeanDistance": list(mean_distances.values())
})

df_mean_distances = df_mean_distances.sort_values("MeanDistance")
print(df_mean_distances)

#df_mean_distances.to_excel("output.xlsx", index=True)

print("\n\nMigliori 5")
print(df_mean_distances.head(5))

top_five = df_mean_distances.head(5)["Generator"].tolist()

for gen in top_five:
    i = names.index(gen)

    row = distance_matrix[i].copy()
    #Riga intera per quel determinato generatore
    #Copia per evitare di intaccare la matrice originale

    row[i] = np.inf
    #Poniamo ad infinito la distanza di un generatore da se stesso

    nearest_idx = np.argsort(row)[:4]
    #Argsort restituisce gli indici

    print(f"\nGeneratore centrale: {gen}")
    for idx in nearest_idx:
        print(f" Vicino: {names[idx]} | distanza: {row[idx]:.6f}")


#REAL
real_idx = names.index("REAL")

real_distances = distance_matrix[real_idx].copy()

real_distances[real_idx] = np.inf

nearest_idx = np.argsort(real_distances)[:5]

print("\n\nTop 5 nearest generator to REAL")
for idx in nearest_idx:
    print(f"{names[idx]} | distanza: {real_distances[idx]:.6f}")


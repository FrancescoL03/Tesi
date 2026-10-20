import numpy as np
import pandas as pd

from sklearn.metrics import silhouette_samples
import json

#Embedding delle immagini
x = np.load("full_embeddings_artifacts/proj_train_head.npy")
#Etichette dei generatori
y = np.load("full_embeddings_artifacts/y_train.npy")

with open("full_embeddings_artifacts/meta.json", "r") as f:
    meta = json.load(f)

classes = meta["classes"]

#silhouette_values = silhouette_samples(x, y, metric="euclidean")
silhouette_values = silhouette_samples(x, y, metric="cosine")

#silhouette_values = silhouette_samples(x, y, metric="mahalanobis") #delicata

mean_silhouette = {}

for c in np.unique(y):
    mean_silhouette[classes[c]] = silhouette_values[y==c].mean()

df_mean_silhouette = pd.DataFrame({
    "Generator": list(mean_silhouette.keys()),
    "MeanSilhouette": list(mean_silhouette.values())
})

df_mean_silhouette = df_mean_silhouette.sort_values("MeanSilhouette", ascending=True)
print(df_mean_silhouette)

df_mean_silhouette.to_excel("output.xlsx", index=True)

print("\n\nMigliori 5")
print(df_mean_silhouette.head(5))

top_five = df_mean_silhouette.head(5)["Generator"].tolist()

for gen in top_five:
    c = classes.index(gen)

    class_values = silhouette_values[y==c]

    print(f"\nGeneratore centrale: {gen}")
    print(f" Mean silhouette: {class_values.mean():.6f}") #media dei punti del cluster
    print(f" Min silhouette: {class_values.min():.6f}") #valore min tra i vari punti
    print(f" Max silhouette: {class_values.max():.6f}") #valore max tra i vari punti


#REAL
real_idx = classes.index("REAL")
real_values = silhouette_values[y == real_idx]

print("\n\nREAL")
print(f" Mean silhouette: {real_values.mean():.6f}")
print(f" Min silhouette: {real_values.min():.6f}")
print(f" Max silhouette: {real_values.max():.6f}")
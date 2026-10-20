from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score

from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans

import numpy as np
import pandas as pd

import json

#Embedding delle immagini
x = np.load("full_embeddings_artifacts/proj_train_head.npy")
#Etichette dei generatori
y = np.load("full_embeddings_artifacts/y_train.npy")

kmeans = KMeans(n_clusters=26, random_state=42)
labels_pred = kmeans.fit_predict(x)


#Metriche esterne (NMI e ARI)
#Confronto tra le etichette y e la costruzione dei cluster di Kmeans
score_NMI = normalized_mutual_info_score(y, labels_pred) #Massimo 1
print("Score NMI: ", score_NMI)

score_ARI = adjusted_rand_score(y, labels_pred) #Tra 0 e 1
print("Score ARI: ", score_ARI)

print("===============")

#Metriche Interne
#Silhouette alto (-1 e 1)
score_silhouette_y = silhouette_score(x,y)
print("Silhouette score (y):", score_silhouette_y)

score_silhouette_labels = silhouette_score(x, labels_pred)
print("Silhouette score (labels):", score_silhouette_labels)

#Davies-Bouldin basso, caso migliore 0
score_db_y = davies_bouldin_score(x,y)
print("Score Davies-Bouldin (y): ", score_db_y)

score_db_labels = davies_bouldin_score(x, labels_pred)
print("Score Davies-Bouldin (labels)", score_db_labels)

#Calinski-Harabasz alto
score_ch_y = calinski_harabasz_score(x,y)
print("Score Calinski-Harabasz (y): ", score_ch_y)

score_ch_labels = calinski_harabasz_score(x, labels_pred)
print("Score Calinski-Harabasz (labels): ", score_ch_labels)

#WCSS basso
score_wcss = kmeans.inertia_
print("Score WCSS: ", score_wcss)



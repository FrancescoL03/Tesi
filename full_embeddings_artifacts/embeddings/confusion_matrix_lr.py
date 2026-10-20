import numpy as np 
import pandas as pd 
import json 

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix, classification_report 

# Embedding delle immagini 
x = np.load("full_embeddings_artifacts/proj_train_head.npy") 
# Etichette dei generatori 
y = np.load("full_embeddings_artifacts/y_train.npy") 

with open("full_embeddings_artifacts/meta.json", "r") as f: 
    meta = json.load(f) 


classes = meta["classes"] 
label_indices = list(range(len(classes))) 

pd.set_option("display.float_format", lambda x: f"{x:.6f}")

#80 e 20
x_train, x_temp, y_train, y_temp = train_test_split( 
    x, 
    y, 
    test_size=0.2, 
    stratify=y, #Mantiene la proporzione il + simile possibile 
    random_state=42
) 


#20 in 10 e 10
x_val, x_test, y_val, y_test = train_test_split( 
    x_temp, 
    y_temp, 
    test_size=0.5, 
    stratify=y_temp, 
    random_state=42
) 


print("Dimensioni split:") 
print(f"Train: {x_train.shape[0]} campioni") 
print(f"Validation: {x_val.shape[0]} campioni") 
print(f"Test: {x_test.shape[0]} campioni") 

#Conteggio split
def print_class_distribution(name, labels, classes):
    counts = pd.Series(labels).value_counts().sort_index()
    print(f"\nDistribuzione classi in {name}:")
    for i, cls in enumerate(classes):
        print(f"{cls}: {counts.get(i,0)}")

print_class_distribution("TOTALE", y, classes)
print_class_distribution("TRAIN", y_train, classes)
print_class_distribution("VALIDATION", y_val, classes)
print_class_distribution("TEST", y_test, classes)

# ========================= 
# CLASSIFICATORE 
# ========================= 
clf = LogisticRegression( 
    max_iter=2000, 
    random_state=42 
) 
  
clf.fit(x_train, y_train) 

# ========================= 
# FUNZIONE DI ANALISI 
# ========================= 
def analyze_split(split_name, y_true, y_pred, classes): 
    print(f"\n{'='*60}") 
    print(f"RISULTATI SU {split_name}") 
    print(f"{'='*60}") 

    # Confusion matrix con ordine fisso delle classi 
    cm = confusion_matrix(y_true, y_pred, labels=label_indices) 

    # Normalizzazione per riga 
    row_sums = cm.sum(axis=1, keepdims=True) 
    cm_norm = np.divide(cm.astype(float), row_sums, where=row_sums != 0) 

    df_cm = pd.DataFrame(cm_norm, index=classes, columns=classes) 
    print("\nConfusion Matrix normalizzata:") 
    print(df_cm) 

    # Report della classificazione 
    print(f"\nClassification report - {split_name}:") 
    print(classification_report(y_true, y_pred, labels=label_indices, target_names=classes, zero_division=0)) 

    # Classi più confuse 
    print(f"\nClassi più confuse - {split_name}:") 
    for c in range(len(classes)): 
        row = cm_norm[c].copy() 
        row[c] = -1  # escludo la diagonale 

        nearest_idx = np.argsort(row)[::-1][:4] 

        print(f"\nClasse: {classes[c]}") 

        for idx in nearest_idx: 
            print(f" Confusa con: {classes[idx]} | valore: {row[idx]:.6f}") 

  
    # REAL 
    real_idx = classes.index("REAL") 
    real_row = cm_norm[real_idx].copy() 
    real_row[real_idx] = -1 

    nearest_idx = np.argsort(real_row)[::-1][:5] 

    print(f"\nTop 5 classi con cui REAL è più confuso - {split_name}") 
    for idx in nearest_idx: 
        print(f"{classes[idx]} | valore: {real_row[idx]:.6f}") 

    # Media della confusione 
    mean_confusion = {} 
    for c in range(len(classes)): 
        row = np.delete(cm_norm[c], c) 
        mean_confusion[classes[c]] = row.mean() 

    df_mean_confusion = pd.DataFrame({ 
        "Generator": list(mean_confusion.keys()), 
        "MeanConfusion": list(mean_confusion.values()) 
    })

    df_mean_confusion = df_mean_confusion.sort_values("MeanConfusion", ascending=False) 

    print(f"\nMedia della confusione - {split_name}") 
    print(df_mean_confusion) 

    print(f"\nTop 5 classi meno distinguibili - {split_name}") 
    print(df_mean_confusion.head(5)) 

    # Salvataggio 
    df_cm.to_excel(f"confusion_matrix_{split_name.lower()}.xlsx", index=True) 
    df_mean_confusion.to_excel(f"mean_confusion_{split_name.lower()}.xlsx", index=False) 

    return df_cm, df_mean_confusion 

# ========================= 
# VALIDATION 
# ========================= 
y_val_pred = clf.predict(x_val) 
df_cm_val, df_mean_conf_val = analyze_split("VALIDATION", y_val, y_val_pred, classes) 

# ========================= 
# TEST 
# ========================= 
y_test_pred = clf.predict(x_test) 
df_cm_test, df_mean_conf_test = analyze_split("TEST", y_test, y_test_pred, classes) 
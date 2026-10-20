import numpy as np
import pandas as pd
import json
import inspect
 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import qualitative
 
 
# =========================
# CARICAMENTO DATI
# =========================
 
# Embedding delle immagini
x = np.load("full_embeddings_artifacts/proj_train_head.npy")
 
# Etichette dei generatori
y = np.load("full_embeddings_artifacts/y_train.npy")
 
with open("full_embeddings_artifacts/meta.json", "r") as f:
    meta = json.load(f)
 
classes = meta["classes"]
label_indices = list(range(len(classes)))
 
pd.set_option("display.float_format", lambda x: f"{x:.6f}")
 
 
# =========================
# TRAIN / VALIDATION / TEST SPLIT
# =========================
 
# 80 e 20
x_train, x_temp, y_train, y_temp = train_test_split(
    x,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
 
# 20 in 10 e 10
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
 
 
def print_class_distribution(name, labels, classes):
    counts = pd.Series(labels).value_counts().sort_index()
    print(f"\nDistribuzione classi in {name}:")
    for i, cls in enumerate(classes):
        print(f"{cls}: {counts.get(i, 0)}")
 
 
print_class_distribution("TOTALE", y, classes)
print_class_distribution("TRAIN", y_train, classes)
print_class_distribution("VALIDATION", y_val, classes)
print_class_distribution("TEST", y_test, classes)
 
 
# =========================
# DBSCAN
# =========================
 
dbscan = DBSCAN(eps=0.22, min_samples=15)
dbscan.fit(x_train)
 
 
def get_point_types(dbscan_model, x):
    all_indices = np.arange(len(x))
    core_indices = dbscan_model.core_sample_indices_
    noise_mask = dbscan_model.labels_ == -1
    noise_indices = all_indices[noise_mask]
 
    core_set = set(core_indices)
    noise_set = set(noise_indices)
 
    border_indices = np.array([
        i for i in all_indices
        if i not in core_set and i not in noise_set
    ])
 
    result = {
        "core": core_indices,
        "border": border_indices,
        "noise": noise_indices,
    }
 
    print("\nConteggio punti DBSCAN su x_train:")
    for ptype, idx in result.items():
        print(f"  {ptype:6s}: {len(idx)}")
 
    return result
 
 
def filter_by_type(x, y, point_types, remove):
    remove = remove.lower()
 
    if remove not in point_types:
        raise ValueError(f"'remove' deve essere uno tra: {list(point_types.keys())}")
 
    indices_to_remove = set(point_types[remove])
    all_indices = np.arange(len(x))
 
    keep_mask = np.array([
        i not in indices_to_remove
        for i in all_indices
    ])
 
    x_filtered = x[keep_mask]
    y_filtered = y[keep_mask]
 
    print(f"\nFiltro '{remove}' applicato:")
    print(f"  Punti rimossi : {keep_mask.size - keep_mask.sum()}")
    print(f"  Punti rimasti : {keep_mask.sum()}")
 
    return x_filtered, y_filtered, keep_mask
 
 
point_types = get_point_types(dbscan, x_train)
 

# Valori validi:
#   "core"
#   "border"
#   "noise"
REMOVE_TYPE = "core"
REMOVE_TYPE = REMOVE_TYPE.lower()
 
valid_remove_types = {"core", "border", "noise"}
if REMOVE_TYPE not in valid_remove_types:
    raise ValueError(f"REMOVE_TYPE deve essere uno tra: {valid_remove_types}")
 
removed_type = REMOVE_TYPE
kept_types = valid_remove_types - {removed_type}
 
 
x_train_filtered, y_train_filtered, train_keep_mask = filter_by_type(
    x_train,
    y_train,
    point_types,
    remove=REMOVE_TYPE
)
 
 
# =========================
# ETICHETTE GEOMETRICHE DBSCAN
# =========================
# Per ogni punto di x_train:
#   "core"
#   "border"
#   "noise"
# =========================
 
all_indices = np.arange(len(x_train))
core_set = set(dbscan.core_sample_indices_)
noise_set = set(all_indices[dbscan.labels_ == -1])
 
geom_label = np.array([
    "noise" if i in noise_set else ("core" if i in core_set else "border")
    for i in all_indices
], dtype=object)
 
 
# =========================
# t-SNE 3D INTERATTIVO HTML
# =========================
 
MAX_SAMPLES = None
RANDOM_STATE = 42
PERPLEXITY = 40
TSNE_ITERS = 1000
 
# IMPORTANTE:
# False = confronto corretto per vedere lo stesso embedding con punti rimossi.
# True  = ricalcola t-SNE dopo la rimozione; i due plot possono cambiare molto.
RECOMPUTE_AFTER_TSNE = False
 
 
# -------------------------
# CAMPIONAMENTO UNICO
# -------------------------
 
rng = np.random.default_rng(RANDOM_STATE)
 
if MAX_SAMPLES is not None and len(x_train) > MAX_SAMPLES:
    idx_sample = np.sort(
        rng.choice(len(x_train), size=MAX_SAMPLES, replace=False)
    )
else:
    idx_sample = np.arange(len(x_train))
 
x_plot = x_train[idx_sample]
y_plot = y_train[idx_sample]
geom_plot = geom_label[idx_sample]
 
 
# -------------------------
# MASCHERE BEFORE / AFTER
# -------------------------
 
keep_after_mask = geom_plot != removed_type
removed_mask = geom_plot == removed_type
 
print(f"\nTipo rimosso per visualizzazione AFTER: {removed_type}")
print(f"Punti rimossi nel campione HTML: {removed_mask.sum()}")
print(f"Punti rimasti nel campione HTML: {keep_after_mask.sum()}")
 
 
# -------------------------
# FUNZIONE COMPATIBILE PER TSNE
# -------------------------
# Alcune versioni di sklearn usano "n_iter",
# quelle più recenti usano "max_iter".
# Questa funzione gestisce entrambe.
# -------------------------
 
def make_tsne(
    n_components=3,
    perplexity=40,
    n_iter_or_max_iter=1000,
    random_state=42,
    init="pca",
    learning_rate="auto"
):
    tsne_signature = inspect.signature(TSNE)
    tsne_params = {
        "n_components": n_components,
        "perplexity": perplexity,
        "random_state": random_state,
        "init": init,
        "learning_rate": learning_rate,
    }
 
    if "max_iter" in tsne_signature.parameters:
        tsne_params["max_iter"] = n_iter_or_max_iter
    else:
        tsne_params["n_iter"] = n_iter_or_max_iter
 
    return TSNE(**tsne_params)
 
 
# -------------------------
# t-SNE BEFORE
# -------------------------
 
print(f"\nt-SNE BEFORE su {len(x_plot)} punti — attendere...")
 
tsne_before = make_tsne(
    n_components=3,
    perplexity=PERPLEXITY,
    n_iter_or_max_iter=TSNE_ITERS,
    random_state=RANDOM_STATE,
    init="pca",
    learning_rate="auto"
)
 
emb_before = tsne_before.fit_transform(x_plot)
 
print("  Done.")
 
 
# -------------------------
# t-SNE AFTER
# -------------------------
 
if RECOMPUTE_AFTER_TSNE:
    print(f"\nt-SNE AFTER ricalcolato su {keep_after_mask.sum()} punti — attendere...")
 
    x_after_for_tsne = x_plot[keep_after_mask]
 
    # Se il numero di punti rimasti è troppo basso rispetto alla perplexity,
    # riduciamo automaticamente la perplexity.
    after_perplexity = min(PERPLEXITY, max(5, (len(x_after_for_tsne) - 1) // 3))
 
    tsne_after = make_tsne(
        n_components=3,
        perplexity=after_perplexity,
        n_iter_or_max_iter=TSNE_ITERS,
        random_state=RANDOM_STATE,
        init="pca",
        learning_rate="auto"
    )
 
    emb_after = tsne_after.fit_transform(x_after_for_tsne)
 
    print("  Done.")
else:
    print(
        "\nAFTER ottenuto rimuovendo i punti selezionati "
        "dallo stesso embedding del BEFORE."
    )
 
    emb_after = emb_before[keep_after_mask]
 
 
y_after = y_plot[keep_after_mask]
geom_after = geom_plot[keep_after_mask]
 
 
# -------------------------
# COLORI
# -------------------------
 
palette = (
    qualitative.Alphabet
    + qualitative.Dark24
    + qualitative.Light24
    + qualitative.Set3
)
 
class_colors = {
    cls: palette[i % len(palette)]
    for i, cls in enumerate(classes)
}
 
 
# -------------------------
# CREAZIONE FIGURA PLOTLY
# -------------------------
 
fig = make_subplots(
    rows=1,
    cols=2,
    specs=[[{"type": "scene"}, {"type": "scene"}]],
    subplot_titles=(
        "BEFORE — core · border · noise",
        f"AFTER — senza {removed_type}"
    )
)
 
trace_types = []
 
 
def add_trace(
    fig,
    row,
    col,
    emb,
    mask,
    name,
    color,
    symbol,
    size,
    opacity,
    legendgroup,
    showlegend,
    point_type,
    class_name=None
):
    if mask.sum() == 0:
        return
 
    hover_text = []
    selected_indices = np.where(mask)[0]
 
    for local_idx in selected_indices:
        if class_name is None:
            hover_text.append(f"Tipo: {point_type}")
        else:
            hover_text.append(f"Classe: {class_name}<br>Tipo: {point_type}")
 
    fig.add_trace(
        go.Scatter3d(
            x=emb[mask, 0],
            y=emb[mask, 1],
            z=emb[mask, 2],
            mode="markers",
            name=name,
            legendgroup=legendgroup,
            showlegend=showlegend,
            marker=dict(
                size=size,
                color=color,
                opacity=opacity,
                symbol=symbol
            ),
            text=hover_text,
            hovertemplate=(
                "%{text}"
                "<br>Dim 1=%{x:.2f}"
                "<br>Dim 2=%{y:.2f}"
                "<br>Dim 3=%{z:.2f}"
                "<extra></extra>"
            )
        ),
        row=row,
        col=col
    )
 
    trace_types.append(point_type)
 
 
# -------------------------
# TRACCE PER CLASSI E TIPI
# -------------------------
'''
for c_idx, cls_name in enumerate(classes):
    color = class_colors[cls_name]
 
    # BEFORE — CORE
    mask_b_core = (y_plot == c_idx) & (geom_plot == "core")
    add_trace(
        fig=fig,
        row=1,
        col=1,
        emb=emb_before,
        mask=mask_b_core,
        name=f"{cls_name} · core",
        color=color,
        symbol="circle",
        size=4,
        opacity=0.85,
        legendgroup=f"{cls_name} · core",
        showlegend=True,
        point_type="core",
        class_name=cls_name
    )
 
    # AFTER — CORE
    mask_a_core = (y_after == c_idx) & (geom_after == "core")
    add_trace(
        fig=fig,
        row=1,
        col=2,
        emb=emb_after,
        mask=mask_a_core,
        name=f"{cls_name} · core",
        color=color,
        symbol="circle",
        size=4,
        opacity=0.85,
        legendgroup=f"{cls_name} · core",
        showlegend=False,
        point_type="core",
        class_name=cls_name
    )
 
    # BEFORE — BORDER
    mask_b_border = (y_plot == c_idx) & (geom_plot == "border")
    add_trace(
        fig=fig,
        row=1,
        col=1,
        emb=emb_before,
        mask=mask_b_border,
        name=f"{cls_name} · border",
        color=color,
        symbol="diamond",
        size=3,
        opacity=0.55,
        legendgroup=f"{cls_name} · border",
        showlegend=True,
        point_type="border",
        class_name=cls_name
    )
 
    # AFTER — BORDER
    mask_a_border = (y_after == c_idx) & (geom_after == "border")
    add_trace(
        fig=fig,
        row=1,
        col=2,
        emb=emb_after,
        mask=mask_a_border,
        name=f"{cls_name} · border",
        color=color,
        symbol="diamond",
        size=3,
        opacity=0.55,
        legendgroup=f"{cls_name} · border",
        showlegend=False,
        point_type="border",
        class_name=cls_name
    )
 
    # BEFORE — NOISE
    mask_b_noise_cls = (y_plot == c_idx) & (geom_plot == "noise")
    add_trace(
        fig=fig,
        row=1,
        col=1,
        emb=emb_before,
        mask=mask_b_noise_cls,
        name=f"{cls_name} · noise",
        color="rgba(40,40,40,0.45)",
        symbol="x",
        size=3,
        opacity=0.35,
        legendgroup=f"{cls_name} · noise",
        showlegend=True,
        point_type="noise",
        class_name=cls_name
    )
 
    # AFTER — NOISE
    # Verrà aggiunto solo se REMOVE_TYPE non è "noise".
    mask_a_noise_cls = (y_after == c_idx) & (geom_after == "noise")
    add_trace(
        fig=fig,
        row=1,
        col=2,
        emb=emb_after,
        mask=mask_a_noise_cls,
        name=f"{cls_name} · noise",
        color="rgba(40,40,40,0.45)",
        symbol="x",
        size=3,
        opacity=0.35,
        legendgroup=f"{cls_name} · noise",
        showlegend=False,
        point_type="noise",
        class_name=cls_name
    )
'''
REMOVED_COLOR = "rgba(80,80,80,0.55)"
KEPT_OPACITY = 0.85
REMOVED_OPACITY = 0.45
 
for c_idx, cls_name in enumerate(classes):
    for point_type in ["core", "border", "noise"]:
        is_removed_type = point_type == removed_type
        if is_removed_type:
            color = REMOVED_COLOR
            symbol = "x"
            size = 5
            opacity = REMOVED_OPACITY
        else:
            color = class_colors[cls_name]
            symbol = "circle"
            size = 4
            opacity = KEPT_OPACITY

        # BEFORE
        mask_b = (y_plot == c_idx) & (geom_plot == point_type)
        add_trace(
            fig=fig,
            row=1,
            col=1,
            emb=emb_before,
            mask=mask_b,
            name=f"{cls_name} · {point_type}",
            color=color,
            symbol=symbol,
            size=size,
            opacity=opacity,
            legendgroup=f"{cls_name} · {point_type}",
            showlegend=True,
            point_type=point_type,
            class_name=cls_name
        )
 
        # AFTER
        # Nell'AFTER il tipo eliminato non compare,
        # perché geom_after contiene solo i punti rimasti.
        mask_a = (y_after == c_idx) & (geom_after == point_type)
        add_trace(
            fig=fig,
            row=1,
            col=2,
            emb=emb_after,
            mask=mask_a,
            name=f"{cls_name} · {point_type}",
            color=color,
            symbol=symbol,
            size=size,
            opacity=opacity,
            legendgroup=f"{cls_name} · {point_type}",
            showlegend=False,
            point_type=point_type,
            class_name=cls_name
        )
# -------------------------
# BOTTONI DINAMICI
# -------------------------
 
def visible_mask(show_types):
    return [t in show_types for t in trace_types]
 
 
all_types = {"core", "border", "noise"}
 
buttons = [
    dict(
        label="Mostra tutto",
        method="update",
        args=[{"visible": visible_mask(all_types)}]
    ),
    dict(
        label=f"Nascondi {removed_type}",
        method="update",
        args=[{"visible": visible_mask(kept_types)}]
    ),
    dict(
        label="Solo punti rimasti",
        method="update",
        args=[{"visible": visible_mask(kept_types)}]
    ),
    dict(
        label=f"Solo punti rimossi ({removed_type})",
        method="update",
        args=[{"visible": visible_mask({removed_type})}]
    ),
    dict(
        label="Solo core",
        method="update",
        args=[{"visible": visible_mask({"core"})}]
    ),
    dict(
        label="Solo border",
        method="update",
        args=[{"visible": visible_mask({"border"})}]
    ),
    dict(
        label="Solo noise",
        method="update",
        args=[{"visible": visible_mask({"noise"})}]
    )
]
 
 
# -------------------------
# LAYOUT
# -------------------------
 
fig.update_layout(
    title=(
        "t-SNE 3D — Embedding prima e dopo la rimozione DBSCAN "
        f"({removed_type})"
    ),
    height=850,
    width=1750,
    margin=dict(l=10, r=360, t=80, b=20),
 
    legend=dict(
        title=f"Legenda - x grigia = punti rimossi ({removed_type})",
        x=1.02,
        y=0.78,
        xanchor="left",
        yanchor="top",
        groupclick="togglegroup"
    ),
 
    updatemenus=[
        dict(
            type="buttons",
            direction="down",
            x=1.02,
            y=1.0,
            xanchor="left",
            yanchor="top",
            showactive=True,
            buttons=buttons
        )
    ]
)
 
fig.update_scenes(
    xaxis_title="Dim 1",
    yaxis_title="Dim 2",
    zaxis_title="Dim 3"
)
 
fig.layout.scene.camera.eye = dict(x=1.5, y=1.5, z=1.2)
fig.layout.scene2.camera.eye = dict(x=1.5, y=1.5, z=1.2)
 
 
# -------------------------
# SALVATAGGIO HTML
# -------------------------
 
html_path = f"tsne_before_after_remove_{removed_type}.html"
 
fig.write_html(
    html_path,
    include_plotlyjs="cdn",
    full_html=True
)
 
print(f"\nSalvato: {html_path}")
 
fig.show()
 
 
# =========================
# CLASSIFICATORE
# =========================
 
clf = clf = GaussianNB()
 
clf.fit(x_train_filtered, y_train_filtered)
 
 
# =========================
# FUNZIONE DI ANALISI
# =========================
 
def analyze_split(split_name, y_true, y_pred, classes):
    print(f"\n{'=' * 60}")
    print(f"RISULTATI SU {split_name}")
    print(f"{'=' * 60}")
 
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=label_indices
    )
 
    row_sums = cm.sum(axis=1, keepdims=True)
 
    cm_norm = np.divide(
        cm.astype(float),
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0
    )
 
    df_cm = pd.DataFrame(
        cm_norm,
        index=classes,
        columns=classes
    )
 
    print("\nConfusion Matrix normalizzata:")
    print(df_cm)
 
    print(f"\nClassification report - {split_name}:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=label_indices,
            target_names=classes,
            zero_division=0
        )
    )
 
    mean_confusion = {}
 
    for c in range(len(classes)):
        row = np.delete(cm_norm[c], c)
        mean_confusion[classes[c]] = row.mean()
 
    df_mean_confusion = pd.DataFrame({
        "Generator": list(mean_confusion.keys()),
        "MeanConfusion": list(mean_confusion.values())
    }).sort_values("MeanConfusion", ascending=False)
 
    print(f"\nMedia della confusione - {split_name}")
    print(df_mean_confusion)
 
    print(f"\nTop 5 classi meno distinguibili - {split_name}")
    print(df_mean_confusion.head(5))
 
    suffix = f"{split_name.lower()}_remove_{removed_type}"
 
    df_cm.to_excel(
        f"confusion_matrix_{suffix}.xlsx",
        index=True
    )
 
    df_mean_confusion.to_excel(
        f"mean_confusion_{suffix}.xlsx",
        index=False
    )
 
    return df_cm, df_mean_confusion
 
 
# =========================
# VALIDATION
# =========================
 
y_val_pred = clf.predict(x_val)
 
df_cm_val, df_mean_conf_val = analyze_split(
    "VALIDATION",
    y_val,
    y_val_pred,
    classes
)
 
 
# =========================
# TEST
# =========================
 
y_test_pred = clf.predict(x_test)
 
df_cm_test, df_mean_conf_test = analyze_split(
    "TEST",
    y_test,
    y_test_pred,
    classes
)
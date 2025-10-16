import os
import pandas as pd
from ultralytics import YOLO

# Umgebung lesen
perm_set = os.environ["PERM_SET"]
fold_id = os.environ["FOLD_ID"]
data_yaml = os.environ["DATA_YAML"]

# Pfade
if perm_set == "obb":
    model_path = f"/scratch/tmp/t_liet02/obb_aab_runs/{perm_set}/orTrue_Ep500_F{fold_id}/weights/best.pt"
    output_csv = f"/scratch/tmp/t_liet02/obb_aab_runs/{perm_set}/orTrue_Ep500_F{fold_id}/metrics_and_confusion_val.csv"
else:
    model_path = f"/scratch/tmp/t_liet02/obb_aab_runs/{perm_set}/orFalse_Ep500_F{fold_id}/weights/best.pt"
    output_csv = f"/scratch/tmp/t_liet02/obb_aab_runs/{perm_set}/orFalse_Ep500_F{fold_id}/metrics_and_confusion_val.csv"

# Modell laden
model = YOLO(model_path)

# Validierung ausfuehren
results = model.val(
    data=data_yaml,
    split='val',
    name='val_at_val',
    imgsz=1024,
    batch=4,
    device=0,
    save_txt=False,
    save_json=True,
    save_conf=True,
    exist_ok=True,
    project=f"/scratch/tmp/t_liet02/new_val_detect/{perm_set}/fold{fold_id}"
)

# Konfusionsmatrix (Matrix + Klassennamen)
conf_matrix = results.confusion_matrix
matrix = conf_matrix.matrix

# Klassennamen vorbereiten
model_class_names = list(model.names.values())
if matrix.shape[0] > len(model_class_names):
    class_names = model_class_names + ["background"]
else:
    class_names = model_class_names

# DataFrame bauen
conf_df = pd.DataFrame(matrix, index=class_names, columns=class_names)

conf_df = pd.DataFrame(matrix, index=class_names, columns=class_names)
conf_df.index.name = 'True\\Predicted'

# Allgemeine Metriken
metrics = results.results_dict  # dict mit mAP, Precision, Recall usw.
metrics_df = pd.DataFrame([metrics])  # einzelne Zeile mit allen globalen Metriken

# Optionale pro-Klassen-Metriken (falls verfuegbar)
try:
    per_class_df = pd.DataFrame(results.class_metrics, columns=[
        "precision", "recall", "mAP50", "mAP50-95", "cls_id"
    ])
    per_class_df["class_name"] = per_class_df["cls_id"].apply(lambda i: model.names.get(int(i), f"class_{i}"))
    per_class_df = per_class_df[["class_name", "cls_id", "precision", "recall", "mAP50", "mAP50-95"]]
except AttributeError:
    per_class_df = pd.DataFrame()

# Speichern � alle DataFrames untereinander
with open(output_csv, 'w') as f:
    f.write("=== Overall Metrics ===\n")
    metrics_df.to_csv(f, index=False)
    f.write("\n=== Per-Class Metrics ===\n")
    if not per_class_df.empty:
        per_class_df.to_csv(f, index=False)
    else:
        f.write("Keine per-class Metriken verfuegbar\n")
    f.write("\n=== Confusion Matrix ===\n")
    conf_df.to_csv(f)

print(f"Alle Metriken gespeichert unter: {output_csv}")


if perm_set == "obb":
    model_path = f"/scratch/tmp/t_liet02/obb_aab_runs/{perm_set}/orTrue_Ep500_F{fold_id}/weights/best.pt"
    output_csv = f"/scratch/tmp/t_liet02/obb_aab_runs/{perm_set}/orTrue_Ep500_F{fold_id}/metrics_and_confusion_test.csv"
else:
    model_path = f"/scratch/tmp/t_liet02/obb_aab_runs/{perm_set}/orFalse_Ep500_F{fold_id}/weights/best.pt"
    output_csv = f"/scratch/tmp/t_liet02/obb_aab_runs/{perm_set}/orFalse_Ep500_F{fold_id}/metrics_and_confusion_test.csv"

# Modell laden
model = YOLO(model_path)

# Validierung ausfuehren
results = model.val(
    data=data_yaml,
    split='test',
    name='val_at_test',
    imgsz=1024,
    batch=4,
    device=0,
    save_txt=False,
    save_json=True,
    save_conf=True,
    exist_ok=True,
    project=f"/scratch/tmp/t_liet02/new_val_detect/{perm_set}/fold{fold_id}"
)

# Konfusionsmatrix (Matrix + Klassennamen)
conf_matrix = results.confusion_matrix
matrix = conf_matrix.matrix

# Klassennamen vorbereiten
model_class_names = list(model.names.values())
if matrix.shape[0] > len(model_class_names):
    class_names = model_class_names + ["background"]
else:
    class_names = model_class_names

# DataFrame bauen
conf_df = pd.DataFrame(matrix, index=class_names, columns=class_names)

conf_df = pd.DataFrame(matrix, index=class_names, columns=class_names)
conf_df.index.name = 'True\\Predicted'

# Allgemeine Metriken
metrics = results.results_dict  # dict mit mAP, Precision, Recall usw.
metrics_df = pd.DataFrame([metrics])  # einzelne Zeile mit allen globalen Metriken

# Optionale pro-Klassen-Metriken (falls verfuegbar)
try:
    per_class_df = pd.DataFrame(results.class_metrics, columns=[
        "precision", "recall", "mAP50", "mAP50-95", "cls_id"
    ])
    per_class_df["class_name"] = per_class_df["cls_id"].apply(lambda i: model.names.get(int(i), f"class_{i}"))
    per_class_df = per_class_df[["class_name", "cls_id", "precision", "recall", "mAP50", "mAP50-95"]]
except AttributeError:
    per_class_df = pd.DataFrame()

# Speichern � alle DataFrames untereinander
with open(output_csv, 'w') as f:
    f.write("=== Overall Metrics ===\n")
    metrics_df.to_csv(f, index=False)
    f.write("\n=== Per-Class Metrics ===\n")
    if not per_class_df.empty:
        per_class_df.to_csv(f, index=False)
    else:
        f.write("Keine per-class Metriken verfuegbar\n")
    f.write("\n=== Confusion Matrix ===\n")
    conf_df.to_csv(f)

print(f"Alle Metriken gespeichert unter: {output_csv}")


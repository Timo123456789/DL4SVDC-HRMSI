import os
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

def convert_yolo_obb_to_coco(yolo_labels_dir, images_dir, output_path):
    images = []
    annotations = []
    categories = []
    category_set = set()
    annotation_id = 0
    image_id = 0

    for txt_file in sorted(os.listdir(yolo_labels_dir)):
        if not txt_file.endswith(".txt"):
            continue

        image_filename = txt_file.replace(".txt", ".png")  # oder .jpg?
        image_path = os.path.join(images_dir, image_filename)
        img = Image.open(image_path)
        width, height = img.size

        images.append({
            "id": image_id,
            #"file_name": image_filename,
            "width": width,
            "height": height
        })

        with open(os.path.join(yolo_labels_dir, txt_file), "r") as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                class_id = int(parts[0])
                coords = parts[1:]

                # Denormalisieren
                polygon = [coords[i] * width if i % 2 == 0 else coords[i] * height for i in range(8)]

                # Bounding Box berechnen
                xs = polygon[::2]
                ys = polygon[1::2]
                x_min = min(xs)
                y_min = min(ys)
                bbox_w = max(xs) - x_min
                bbox_h = max(ys) - y_min
                area = bbox_w * bbox_h

                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "segmentation": [polygon],
                    "bbox": [x_min, y_min, bbox_w, bbox_h],
                    "area": area,
                    "iscrowd": 0
                })
                annotation_id += 1
                category_set.add(class_id)

        image_id += 1

    # Kategorien erstellen
    categories = [{"id": cid, "name": f"class_{cid}"} for cid in sorted(category_set)]

    coco_dict = {
        "info": {},
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_path, "w") as out_file:
        json.dump(coco_dict, out_file, indent=2)

# Beispielaufruf:
images_dir = r"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\master_thesis\Code\data\rgbir_F4\images"
labels_dir = r"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\master_thesis\Code\data\rgbir_F4\labels"
pred_dir = r"C:\Users\timol\OneDrive - Universität Münster\14. Fachsemester_SS_24\Palma_Runs\cross_validation\rgbir\fold4\new_val\predictions.json"
gt_output_path = "ground_truth_coco.json" # Definieren Sie den Pfad für die GT
convert_yolo_obb_to_coco(labels_dir, images_dir, gt_output_path)

# Lade Ground Truths
cocoGt = COCO(gt_output_path)  # Dies ist das COCO-Objekt für die Ground Truth

# Lade die Vorhersagen (Detection Results)
# ERSTELLEN SIE EIN NEUES COCO-OBJEKT DIREKT AUS DEN ERGEBNISSEN
# Dies ist der entscheidende Unterschied!
# Die loadRes-Methode wird auf einem temporären oder dem Ground Truth COCO-Objekt aufgerufen,
# um ein NEUES COCO-Objekt zu erzeugen, das nur die Ergebnisse enthält.
# Es ist wichtig, dass die image_ids in predictions.json mit denen in ground_truth_coco.json übereinstimmen.
cocoDt = cocoGt.loadRes(pred_dir) # Dies lädt die Ergebnisse in ein Dt-Objekt

# Optional: Überprüfen Sie, ob die image_ids übereinstimmen
# imgIds = sorted(cocoGt.getImgIds())
# cocoDt_imgIds = sorted(cocoDt.getImgIds())
# print(f"GT Image IDs: {imgIds}")
# print(f"DT Image IDs: {cocoDt_imgIds}")

# Evaluation
# cocoEval benötigt das cocoGt Objekt und das cocoDt Objekt (mit den geladenen Ergebnissen)
cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')

# Optional: Beschränken Sie die Auswertung auf bestimmte Bild-IDs, wenn nötig
# imgIds = sorted(cocoGt.getImgIds())
# cocoEval.params.imgIds = imgIds

cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
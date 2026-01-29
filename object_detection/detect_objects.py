from ultralytics import YOLO
import cv2
from collections import Counter


# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolov8n.pt")
results = model("bus.jpg", conf=0.25)

source_image = cv2.imread("bus.jpg")
if source_image is None:
    print("Source Image not found");
    raise FileNotFoundError("Could not find Test file jpg")

result_count = len(results)

# Enumerate Results (one result per image)
for res_idx, res_val in enumerate(results):
    boxes = res_val.boxes
    class_names = res_val.names

    objects_found = len(boxes)
    print(f"Number ob Objects found in image number {res_idx+1}: {objects_found}")

    # print("Found class Names: ", class_names)

    cls_ids = boxes.cls.cpu().numpy().astype(int) if boxes else []
    print(cls_ids)

    # Counting per Detected Class
    count_obj = {}
    for idx_clsid, val_cls_id in enumerate(cls_ids):

        if f"{val_cls_id}" in count_obj:
            count_obj[f"{val_cls_id}"] += 1
        else:
            count_obj[f"{val_cls_id}"] = 1

    
    for class_id, count_amount in count_obj.items():
        print(f"{class_names[int(class_id)]}: {count_amount}")

    
    # Enumerate boxes xyxy
    crops = []
    for i, (xyxy, conf_val) in enumerate(zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy())):
        x1, y1, x2, y2 = map(int, xyxy)

        # Fallback to image boundaries
        h, w = source_image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        crop = source_image[y1:y2, x1:x2].copy()
        crops.append({
            "index": i,
            "class_id": cls_ids[i],
            "class_name": class_names[int(cls_ids[i])],
            "confidence": float(conf_val),
            "bbox_xyxy": (x1, y1, x2, y2),
            "image": crop
        })

        # save crop (optional, useful for OCR debugging)
        out_name = f"crop_{i}_{class_names[int(cls_ids[i])]}_{conf_val:.2f}.png"
        cv2.imwrite(f"./extracted-crops/{out_name}", crop)

    print(f"Saved {len(crops)} crops as crop_*.png")
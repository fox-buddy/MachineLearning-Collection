from ultralytics import YOLO
import cv2
import shutil
import os
import glob


def check_found_objects(results):
    for res_idx, res_val in enumerate(results):
        boxes = res_val.boxes
        class_names = res_val.names

        objects_found = len(boxes)
        print(f"Number ob Objects found in image number {res_idx+1}: {objects_found}")

        # print("Found class Names: ", class_names)

        cls_ids = boxes.cls.cpu().numpy().astype(int) if boxes else []
    

        # Counting per Detected Class
        count_obj = {}
        for idx_clsid, val_cls_id in enumerate(cls_ids):

            if f"{val_cls_id}" in count_obj:
                count_obj[f"{val_cls_id}"] += 1
            else:
                count_obj[f"{val_cls_id}"] = 1

        return {"boxes": boxes, "classnames": class_names, "foundclasses": count_obj, "classids": cls_ids}
        

def extract_crops_from_image(boxes, source_image, class_ids, class_names, save):
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
            "class_id": class_ids[i],
            "class_name": class_names[int(class_ids[i])],
            "confidence": float(conf_val),
            "bbox_xyxy": (x1, y1, x2, y2),
            "image": crop
        })

        if(save):
            out_name = f"crop_{i}_{class_names[int(class_ids[i])]}_{conf_val:.2f}.png"
            cv2.imwrite(f"./extracted-crops/{out_name}", crop)


def delete_destination_directory():
    shutil.rmtree("./extracted-crops")
    os.mkdir("./extracted-crops")






delete_destination_directory()

# Loading pretrained yolo model
model = YOLO("yolov8n.pt")

# Load image List --> Which do we need to process at the moment
directory = './imagesToProcess/*'
image_list = glob.glob(directory)
print(image_list)

for image_path_idx, image_path_val in enumerate(image_list):
    # if(image_path_idx > 1):
    #     break

    results = model(image_path_val, conf=0.25)
    current_work_image = cv2.imread(image_path_val)

    if current_work_image is None:
        print("Source Image not found");
        continue

    results_of_image = len(results)

    print(f"Found {results_of_image} crops on image {image_path_val}")

    found_in_results = check_found_objects(results)

    for class_id, count_amount in found_in_results["foundclasses"].items():
            print(f"{found_in_results["classnames"][int(class_id)]}: {count_amount}")

    extract_crops_from_image(found_in_results["boxes"], current_work_image, found_in_results["classids"], found_in_results["classnames"], True)

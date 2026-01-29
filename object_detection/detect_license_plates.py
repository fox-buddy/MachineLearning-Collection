from ultralytics import YOLOvv11

model = YOLOvv11.from_pretrained("morsetechlab/yolov11-license-plate-detection")
source = 'http://images.cocodataset.org/val2017/000000039769.jpg'
model.predict(source=source, save=True)

# https://huggingface.co/morsetechlab/yolov11-license-plate-detection
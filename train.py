from ultralytics import YOLO

model =YOLO("weights/yolov8n-seg.pt") # 加载预训练模型
   
model.train(data='myseg.yaml', epochs=100, imgsz=1280)
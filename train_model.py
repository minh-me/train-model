from ultralytics import YOLO

# Load model YOLOv8 pre-trained (classification mode)
model = YOLO("yolo11n-cls.pt")  

# Huấn luyện với tập dữ liệu đã tải về
model.train(data="D:/yolo2/gemstone-1", epochs=50, imgsz=224)

from ultralytics import YOLO

# Load model YOLOv8 pre-trained (classification mode)
model = YOLO("yolo11n-cls.pt")  

# Huấn luyện với tập dữ liệu đã tải về
model.train(data="/home/ubuntu/train-model/EuroCoins-1", epochs=100, imgsz=224)

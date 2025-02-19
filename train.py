from ultralytics import YOLO

# Load model YOLOv8 nano (nhẹ, nhanh)
model = YOLO("yolo11n.pt")  

# Train mô hình
model.train(
    data="D:/yolo/car_dataset/data.yaml",  # Đường dẫn file data.yaml
    epochs=50,                     # Số epochs (có thể tăng nếu muốn model mạnh hơn)
    imgsz=640,                     # Kích thước ảnh (mặc định 640)
    batch=8,                        # Batch size
    device="cpu"                   # Dùng GPU nếu có (hoặc "cpu" nếu không có)
)

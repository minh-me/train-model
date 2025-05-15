from ultralytics import YOLO

# Load model YOLOv8 pre-trained (classification mode)
model = YOLO("yolo11n-cls.pt")  

# Huấn luyện với tập dữ liệu đã tải về
model.train(
    data="D:/Izi/train-model/euro-coins",
    epochs=100,
    imgsz=224,
    batch=16,              # Kích thước batch, điều chỉnh theo GPU
    patience=10,           # Early stopping nếu không cải thiện sau 10 epoch
    device=0,              # Sử dụng GPU (0) nếu có, hoặc 'cpu'
    optimizer='AdamW',     # Bộ tối ưu hóa (AdamW thường tốt cho phân loại)
    lr0=0.001,             # Learning rate ban đầu
    augment=True,          # Bật tăng cường dữ liệu
    preprocess='normalize' # Chuẩn hóa ảnh đầu vào
)

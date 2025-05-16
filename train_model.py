from ultralytics import YOLO

# Load mô hình phân loại
model = YOLO("yolo11l-cls.pt", task="classify")

# Freeze 7 tầng đầu (giảm từ 10 để mô hình học tốt hơn nếu dữ liệu lớn)
freeze_layers = 7
for i, (name, param) in enumerate(model.model.named_parameters()):
    if i < freeze_layers and "classifier" not in name:
        param.requires_grad = False
    else:
        param.requires_grad = True

# Huấn luyện
model.train(
    data="/home/ubuntu/train-model/euro-coins",
    epochs=100,
    imgsz=224,
    # batch=16,  # Tăng batch nếu dùng GPU
    batch=8,  # Tăng batch nếu dùng GPU
    patience=20,  # Tăng patience để hội tụ tốt hơn
    # device=0,  # Sử dụng GPU nếu có
    device='cpu',  # Sử dụng GPU nếu có
    optimizer='AdamW',
    lr0=0.001,  # Tăng learning rate nhẹ
    lrf=0.2,
    cos_lr=True,
    augment=True,
    hsv_h=0.1,
    hsv_s=0.5,  # Giảm để bảo toàn màu sắc
    hsv_v=0.5,
    degrees=30.0,  # Tăng góc xoay
    translate=0.2,
    scale=0.5,
    shear=10.0,
    fliplr=0.5,
    mosaic=0.5,  # Thêm mosaic
    mixup=0.2,  # Thêm mixup
    dropout=0.2,  # Giảm dropout để tránh underfit
    verbose=True,
    save_period=10
)

# Đánh giá trên tập validation
results = model.val()
print(results)

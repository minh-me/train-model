from ultralytics import YOLO
import sys
import os

# Load mô hình đã huấn luyện
model_path = "D:\\Izi\\train-model\\runs\\classify\\train6\\weights\\best.pt"
if not os.path.exists(model_path):
    print(f"Lỗi: Mô hình không tồn tại tại {model_path}")
    sys.exit(1)

model = YOLO(model_path)

# Ảnh cần phân loại
image_path = sys.argv[1] if len(sys.argv) > 1 else "D:\\Izi\\train-model\\1936-penny-r.jpg"    

if not os.path.exists(image_path):
    print(f"Lỗi: Ảnh không tồn tại tại {image_path}")
    sys.exit(1)

# Dự đoán
try:
    results = model(image_path, imgsz=224, conf=0.5)
    
    # Hiển thị kết quả
    probs = results[0].probs
    top1_idx = probs.top1
    top1_conf = probs.top1conf.item()  # Convert tensor to float
    class_name = results[0].names[top1_idx]
    print(f"Dự đoán: {class_name} (Index: {top1_idx}, Xác suất: {top1_conf:.2f})")
    
    # In thêm tất cả xác suất cho các lớp (để debug)
    print("\nXác suất cho tất cả các lớp:")
    for i, prob in enumerate(probs.data):
        print(f"{results[0].names[i]}: {prob:.2f}")

except Exception as e:
    print(f"Lỗi khi dự đoán: {e}")
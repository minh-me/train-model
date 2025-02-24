from ultralytics import YOLO
import sys

# Load mô hình đã huấn luyện
model = YOLO("runs/classify/train/weights/best.pt")

# Ảnh cần phân loại (truyền qua dòng lệnh hoặc đặt sẵn)
image_path = sys.argv[1] if len(sys.argv) > 1 else "D:/yolo3/gemstone-1/train/Alexandrite/alexandrite_0_jpg.rf.a84da9aea97a2968a3366a26454eda56.jpg"

# Dự đoán
results = model(image_path)

# Hiển thị kết quả
print(f"Dự đoán: {results[0].probs.top1}, Nhãn: {results[0].names[results[0].probs.top1]}")
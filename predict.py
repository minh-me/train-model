from ultralytics import YOLO
import sys

# Load mô hình đã huấn luyện
model = YOLO("D:/yolo3/runs/classify/train10/weights/best.pt")

# Ảnh cần phân loại (truyền qua dòng lệnh hoặc đặt sẵn)
image_path = sys.argv[1] if len(sys.argv) > 1 else "D:/yolo3/leaf/guava/tomato.png"

# Dự đoán
results = model(image_path)

print(results)
# Hiển thị kết quả
print(f"Dự đoán: {results[0].probs.top1}, Nhãn: {results[0].names[results[0].probs.top1]}")

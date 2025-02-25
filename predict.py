from ultralytics import YOLO
import sys

# Load mô hình đã huấn luyện
model = YOLO("D:/yolo3/runs/classify/train2/weights/best.pt")

# Ảnh cần phân loại (truyền qua dòng lệnh hoặc đặt sẵn)
image_path = sys.argv[1] if len(sys.argv) > 1 else "D:/yolo3/Jewellery-Classification-1/train/WRISTWATCH/IMG-20181209-WA0029_jpg.rf.10e7445d49185a4044b02c83515aa5e5.jpg"

# Dự đoán
results = model(image_path)

# Hiển thị kết quả
print(f"Dự đoán: {results[0].probs.top1}, Nhãn: {results[0].names[results[0].probs.top1]}")

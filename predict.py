from ultralytics import YOLO
import sys

# Load mô hình đã huấn luyện
model = YOLO("/home/ubuntu/train-model/runs/classify/euro-coins/weights/best.pt")

# Ảnh cần phân loại (truyền qua dòng lệnh hoặc đặt sẵn)
image_path = sys.argv[1] if len(sys.argv) > 1 else "D:/Izi/train-model/coin-10-1/train/100 Euro (Unity)/5f8caa9d53e093-28917510-180_jpg.rf.2c5733d58283c5fabe0eac0efb72adbd.jpg"

# Dự đoán
results = model(image_path)

print(results)
# Hiển thị kết quả
print(f"Dự đoán: {results[0].probs.top1}, Nhãn: {results[0].names[results[0].probs.top1]}")

from ultralytics import YOLO
import sys

# Load mô hình đã huấn luyện
model = YOLO("/home/ubuntu/train-model/runs/classify/euro-coins/weights/best.pt")

# Ảnh cần phân loại (truyền qua dòng lệnh hoặc đặt sẵn)
image_path = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/train-model/EuroCoins-1/test/2018/comm_2018_latvia_joint_jpg.rf.40af94668d10e043af6c2428a51a8714.jpg"

# Dự đoán
results = model(image_path)

print(results)
# Hiển thị kết quả
print(f"Dự đoán: {results[0].probs.top1}, Nhãn: {results[0].names[results[0].probs.top1]}")

from ultralytics import YOLO
import cv2
import os

# Load model YOLO pre-trained
model = YOLO("yolo11n.pt")  # Model YOLO nhỏ nhất

# Danh sách các thư mục ảnh cần gán nhãn
dataset_dirs = ["train", "val"]
base_dir = "car_dataset"

for dataset in dataset_dirs:
    img_dir = os.path.join(base_dir, "images", dataset)
    label_dir = os.path.join(base_dir, "labels", dataset)
    os.makedirs(label_dir, exist_ok=True)  # Tạo thư mục nhãn nếu chưa có

    # Duyệt qua từng ảnh trong thư mục
    for img_name in os.listdir(img_dir):
        if not img_name.endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(img_dir, img_name)
        label_path = os.path.join(label_dir, img_name.replace(".jpg", ".txt").replace(".png", ".txt"))

        # Chạy YOLO để detect object
        results = model(img_path)
        
        with open(label_path, "w") as f:
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls)  # Lấy ID class
                    confidence = float(box.conf)  # Độ chính xác
                    x_center, y_center, width, height = box.xywhn[0]  # Tọa độ YOLO format (normalized)
                    
                    # Lưu vào file
                    f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

        print(f"✅ Gán nhãn cho {img_name}")

print("\n🎉 Hoàn thành auto-labeling!")

from ultralytics import YOLO
import cv2

# Load mô hình YOLO đã train để nhận diện hãng xe
model = YOLO("D:/yolo/runs/detect/train6/weights/best.pt")  # Thay bằng model đã train riêng cho hãng xe nếu có

# Dự đoán trên ảnh ô tô
image_path = "D:/yolo/over-20-car-brands-dataset/Audi/Audi/1-audi-rs3-2021-first-drive-review-hero-front.jpg"  # Đường dẫn ảnh đầu vào

# results = model.train(data="data.yaml", epochs=5, imgsz=640)

results = model(image_path, )

# Log kết quả ra console
for result in results:
    print("\n--- Detection Results ---")
    for box in result.boxes:
        class_id = int(box.cls)  # ID của hãng xe
        confidence = float(box.conf)  # Độ chính xác
        x, y, w, h = box.xywh[0]  # Lấy tọa độ Bounding Box
        brand = model.names[class_id]  # Lấy tên hãng xe

        print(f"Brand: {brand}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Bounding Box (x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f})")
        print("----------------------")


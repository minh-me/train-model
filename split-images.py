import os
import shutil
import random

# Danh sách các hãng xe
brands = [
    "alfa romeo",
    "Audi",
    "Bentley",
    "Benz",
    "Bmw",
    "Cadillac",
    "Dodge",
    "Ferrari",
    "Ford",
    "Ford mustang",
    "hyundai",
    "Kia",
    "Lamborghini",
    "Lexus",
    "Maserati",
    "Porsche",
    "Rolls royce",
    "Tesla",
    "Toyota",
]

# Tạo thư mục theo cấu trúc YOLO nếu chưa có
base_dir = "car_dataset"
sub_dirs = ["images/train", "images/val", "labels/train", "labels/val"]

for sub in sub_dirs:
    os.makedirs(os.path.join(base_dir, sub), exist_ok=True)

# Duyệt qua từng hãng xe và di chuyển ảnh
dataset_path = "over-20-car-brands-dataset"  # Thư mục gốc chứa ảnh theo hãng

for class_id, brand in enumerate(brands):
    brand_path = os.path.join(dataset_path, brand)
    if not os.path.exists(brand_path):
        print(f"❌ Không tìm thấy thư mục {brand_path}, bỏ qua...")
        continue

    # Lấy danh sách ảnh
    images = [f for f in os.listdir(brand_path) if f.endswith((".jpg", ".png", ".jpeg"))]
    random.shuffle(images)  # Xáo trộn ngẫu nhiên

    # Chia 80% train, 20% val
    train_size = int(0.8 * len(images))
    train_images = images[:train_size]
    val_images = images[train_size:]

    # Di chuyển ảnh vào thư mục YOLO
    for img_name in train_images:
        shutil.copy(os.path.join(brand_path, img_name), os.path.join(base_dir, "images/train", img_name))

    for img_name in val_images:
        shutil.copy(os.path.join(brand_path, img_name), os.path.join(base_dir, "images/val", img_name))

    print(f"✅ Đã xử lý {brand}: {len(train_images)} ảnh train, {len(val_images)} ảnh val.")

print("\n🎉 Tạo dataset hoàn tất!")

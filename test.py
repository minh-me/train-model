import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
import torch
from tqdm import tqdm

# === CẤU HÌNH ===
MODEL_PATH = "yolo11x-cls.pt"  # Hoặc mô hình khác
IMAGE_FOLDER = "D:\\Izi\\train-model\\euro-coins\\train"
QUERY_IMAGE = "D:\\Izi\\train-model\\images.jpg"

# === Load mô hình và loại bỏ lớp softmax (classifier) ===
model = YOLO(MODEL_PATH)
model.model.classifier = torch.nn.Identity()  # Xóa lớp phân loại (chỉ lấy feature)
model.eval()

# === Hàm xử lý ảnh về input 224x224 ===
def preprocess_image(img_path, imgsz=224):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (imgsz, imgsz))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # [C, H, W]
    img = torch.tensor(img).unsqueeze(0)  # [1, C, H, W]
    return img

# === Trích tất cả embeddings trong thư mục ===
embeddings = []
file_paths = []

print("🔍 Đang trích embedding ảnh...")
for root, dirs, files in os.walk(IMAGE_FOLDER):
    for file in tqdm(files):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(root, file)
            img = preprocess_image(path)
            with torch.no_grad():
                embedding = model.model(img)[0].cpu().numpy().flatten()
            embeddings.append(embedding)
            file_paths.append(path)

embeddings = np.vstack(embeddings)

# === Trích embedding ảnh cần truy vấn ===
print(f"\n🔍 Tìm ảnh tương tự cho: {QUERY_IMAGE}")
query_img = preprocess_image(QUERY_IMAGE)
with torch.no_grad():
    query_embedding = model.model(query_img)[0].cpu().numpy().flatten().reshape(1, -1)

# === Tính cosine similarity ===
similarities = cosine_similarity(query_embedding, embeddings)[0]
top_k_idx = np.argsort(similarities)[::-1][:5]  # Top 5

print("\n✅ Top 5 ảnh gần nhất:")
for rank, idx in enumerate(top_k_idx, 1):
    print(f"{rank}. {file_paths[idx]} (score: {similarities[idx]:.4f})")

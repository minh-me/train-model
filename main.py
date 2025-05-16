from fastapi import FastAPI
from pydantic import BaseModel
from ultralytics import YOLO
import requests
import os
import clip
import torch
from build_index import build_index  # Import từ build_index.py
from query_image import query_image  # Import từ query_image.py

# Khởi tạo FastAPI
app = FastAPI()

# Load từng mô hình YOLO riêng
MODELS = {
    "gemstone": YOLO("D:\\Izi\\train-model\\runs\\classify\\gemstone\\weights\\best.pt"),
    "jewellery": YOLO("D:\\Izi\\train-model\\runs\\classify\\jewellery\\weights\\best.pt"),
    "leaf": YOLO("D:\\Izi\\train-model\\runs\\classify\\leaf\\weights\\best.pt"),
}

# Thư mục lưu ảnh tạm thời
UPLOAD_FOLDER = "D:\\Izi\\train-model\\uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Cấu hình cho CLIP+FAISS
DATASET_DIR = "D:\\Izi\\train-model\\euro-coins\\train"
INDEX_PATH = "D:\\Izi\\train-model\\faiss_index.index"
LABELS_PATH = "D:\\Izi\\train-model\\labels.pkl"
DEVICE = "cpu"  # hoặc "cuda" nếu có GPU
CLIP_MODEL, CLIP_PREPROCESS = clip.load("ViT-B/32", device=DEVICE)
print("✅ CLIP loaded!")

class ImageURL(BaseModel):
    url: str

@app.post("/predict/{category}/")
async def predict(category: str, image: ImageURL):
    # Kiểm tra danh mục hợp lệ
    if category not in MODELS:
        return {"error": "Invalid category. Choose from: euro-coins, gemstone, jewellery, leaf"}

    # Tải ảnh từ URL
    response = requests.get(image.url, stream=True)
    if response.status_code != 200:
        return {"error": "Failed to download image from URL"}

    # Lưu ảnh tạm thời
    filename = os.path.basename(image.url.split("?")[0])  # Loại bỏ query string nếu có
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    
    with open(file_path, "wb") as buffer:
        for chunk in response.iter_content(1024):
            buffer.write(chunk)

    # Chọn model phù hợp
    model = MODELS[category]

    # Dự đoán
    results = model(file_path)

    # Lấy danh sách nhãn và xác suất dự đoán
    probabilities = results[0].probs.data.tolist()
    class_names = results[0].names

    # Tìm lớp có xác suất cao nhất
    top_indices = sorted(range(len(probabilities)), key=lambda i: probabilities[i], reverse=True)
    top_predictions = [
        {
            "label": class_names[idx],
            "confidence": f"{probabilities[idx] * 100:.2f}%"
        }
        for idx in top_indices[:3]  # Lấy 3 kết quả có xác suất cao nhất
    ]

    # Xóa ảnh sau khi dự đoán
    try:
        os.remove(file_path)
    except Exception as e:
        print(f"Error deleting file: {e}")

    return {
        "image_url": image.url,
        "category": category,
        "predicted_label": top_predictions[0]["label"],
        "confidence": top_predictions[0]["confidence"],
        "topList": top_predictions[1:]  # 2 lựa chọn liên quan khác
    }

@app.post("/predict-clip/euro-coins/")
async def predict_clip(image: ImageURL):
    # Chỉ dành cho euro-coins
    category = "euro-coins"

    # Tải ảnh từ URL
    response = requests.get(image.url, stream=True)
    if response.status_code != 200:
        return {"error": "Failed to download image from URL"}

    # Lưu ảnh tạm thời
    filename = os.path.basename(image.url.split("?")[0])
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    
    with open(file_path, "wb") as buffer:
        for chunk in response.iter_content(1024):
            buffer.write(chunk)

    # Dự đoán với CLIP+FAISS
    try:
        results = query_image(file_path, CLIP_MODEL, CLIP_PREPROCESS, top_k=4, device=DEVICE)
        if not results:
            os.remove(file_path)
            return {"error": "Không thể dự đoán. Kiểm tra ảnh hoặc chỉ mục FAISS."}

        # Định dạng kết quả giống YOLO
        top_predictions = [
            {
                "label": label,
                "confidence": f"{sim * 100:.2f}%"
            }
            for label, sim in results
        ]

        # Xóa ảnh sau khi dự đoán
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file: {e}")

        return {
            "image_url": image.url,
            "category": category,
            "predicted_label": top_predictions[0]["label"],
            "confidence": top_predictions[0]["confidence"],
            "topList": top_predictions[1:]  # 2 lựa chọn liên quan
        }

    except Exception as e:
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file: {e}")
        return {"error": f"Lỗi khi dự đoán: {str(e)}"}
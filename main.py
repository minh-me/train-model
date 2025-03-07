from fastapi import FastAPI
from pydantic import BaseModel
from ultralytics import YOLO
import requests
import os

# Khởi tạo FastAPI
app = FastAPI()

# Load từng mô hình riêng
MODELS = {
    "euro-coins": YOLO("runs/classify/euro-coins/weights/best.pt"),
    "gemstone": YOLO("runs/classify/gemstone/weights/best.pt"),
    "jewellery": YOLO("runs/classify/jewellery/weights/best.pt"),
    "leaf": YOLO("runs/classify/leaf/weights/best.pt"),
}

# Thư mục lưu ảnh tạm thời
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
        "topList": top_predictions[1:]  # 3 lựa chọn liên quan khác
    }

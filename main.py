from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import shutil
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

@app.post("/predict/{category}/")
async def predict(category: str, file: UploadFile = File(...)):
    # Kiểm tra danh mục hợp lệ
    if category not in MODELS:
        return {"error": "Invalid category. Choose from: euro-coins, gemstone, jewellery, leaf"}

    # Lưu ảnh tạm thời
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Chọn model phù hợp
    model = MODELS[category]

    # Dự đoán
    results = model(file_path)

    # Lấy danh sách nhãn và xác suất dự đoán
    probabilities = results[0].probs.data.tolist()
    class_names = results[0].names

    # Tìm lớp có xác suất cao nhất
    top_class_idx = results[0].probs.top1
    predicted_label = class_names[top_class_idx]
    confidence = probabilities[top_class_idx] * 100  # Chuyển sang %

    return {
        "filename": file.filename,
        "category": category,
        "predicted_label": predicted_label,
        "confidence": f"{confidence:.2f}%"
    }

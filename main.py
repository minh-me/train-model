from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import shutil
import os

# Khởi tạo FastAPI
app = FastAPI()

# Load mô hình đã huấn luyện
MODEL_PATH = "runs/classify/train/weights/best.pt"
model = YOLO(MODEL_PATH)

# Thư mục lưu ảnh tạm thời
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Lưu ảnh tạm thời
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Dự đoán
    results = model(file_path)

    # Lấy danh sách nhãn và xác suất dự đoán
    probabilities = results[0].probs.data.tolist()  # Chuyển tensor thành danh sách
    class_names = results[0].names  # Tên các lớp

    # Tìm lớp có xác suất cao nhất
    top_class_idx = results[0].probs.top1
    predicted_label = class_names[top_class_idx]
    confidence = probabilities[top_class_idx] * 100  # Chuyển sang %

    return {
        "filename": file.filename,
        "predicted_label": predicted_label,
        "confidence": f"{confidence:.2f}%"
    }

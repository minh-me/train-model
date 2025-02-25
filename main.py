from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import io
from PIL import Image

# Khởi tạo FastAPI
app = FastAPI()

# Load mô hình đã huấn luyện
MODEL_PATH = "D:/yolo3/runs/classify/train/weights/best.pt"
model = YOLO(MODEL_PATH)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Đọc file ảnh vào bộ nhớ
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    
    # Dự đoán
    results = model(image)
    
    # Lấy nhãn dự đoán
    predicted_label = results[0].names[results[0].probs.top1]
    
    return {"filename": file.filename, "predicted_label": predicted_label}

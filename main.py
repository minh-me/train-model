from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import io
from PIL import Image

# Khởi tạo FastAPI
app = FastAPI()

# Load mô hình đã huấn luyện
MODEL_PATH = "runs/classify/train/weights/best.pt"
model = YOLO(MODEL_PATH)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Đọc file ảnh vào bộ nhớ
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    
    # Dự đoán
    results = model(image)
    
    # Kiểm tra xem có xác suất dự đoán không
    if not hasattr(results[0], "probs"):
        return {"error": "Model output does not contain probability scores"}

    # Lấy top 5 nhãn dự đoán và xác suất tương ứng
    top5_indices = results[0].probs.top5  # Lấy index của 5 kết quả cao nhất
    top5_labels = [results[0].names[i] for i in top5_indices]
    top5_confidences = [results[0].probs.data[i].item() for i in top5_indices]  # Chuyển đổi sang float
    
    predictions = [{"label": label, "confidence": conf} for label, conf in zip(top5_labels, top5_confidences)]
    
    return {"filename": file.filename, "predictions": predictions}

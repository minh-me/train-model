# Hướng dẫn chạy YOLOv8 để phân loại đá quý

## Cài đặt môi trường
```bash
pip install -r requirements.txt
```

## Tải dữ liệu từ Roboflow
```bash
python gen-gemstone-dataset.py
```

## Huấn luyện mô hình
```bash
python train.py
```

## Dự đoán với ảnh mới
```bash
python predict.py 
```


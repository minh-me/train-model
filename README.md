# Hướng dẫn chạy YOLOv8 để phân loại đá quý

## Cài đặt môi trường

```bash
git checkout yolo-v2
```

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

## Chạy server

```bash
python main.py
```

## Build với Clip
```bash
pip install clip
pip install git+https://github.com/openai/CLIP.git
source /path/to/your/venv/bin/activate  # On Linux/Mac
python /home/train-model/build_index.py
```

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import faiss
from PIL import Image
from tqdm import tqdm
import pickle
import torch
import clip
import sys

# ========== CONFIG ==========
DATASET_DIR = "D:\\Izi\\train-model\\euro-coins\\train"
INDEX_PATH = "D:\\Izi\\train-model\\faiss_index.index"
LABELS_PATH = "D:\\Izi\\train-model\\labels.pkl"
DEVICE = "cpu"  # hoặc "cuda" nếu có GPU

# Kiểm tra môi trường
print(f"Python version: {sys.version}")
print(f"Torch available: {torch.__version__}")
print(f"CLIP available: {clip.__version__ if hasattr(clip, '__version__') else 'Unknown'}")
print(f"FAISS available: {faiss.__version__ if hasattr(faiss, '__version__') else 'Unknown'}")
print(f"Device: {DEVICE}")
print(f"Dataset directory: {DATASET_DIR}")
if not os.path.exists(DATASET_DIR):
    print(f"ERROR: Directory {DATASET_DIR} does not exist!")
    sys.exit(1)

# ========== Load CLIP ==========
try:
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    print("✅ CLIP loaded!")
except Exception as e:
    print(f"ERROR: Failed to load CLIP model: {e}")
    sys.exit(1)

# ========== Trích xuất vector từ 1 ảnh ==========
def extract_features(image_path, model, preprocess, device=DEVICE):
    try:
        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image)
        return embedding.cpu().numpy()[0]
    except Exception as e:
        print(f"Lỗi ảnh {image_path}: {e}")
        return None

# ========== Xây dựng chỉ mục FAISS ==========
def build_index(dataset_dir, model, preprocess, device=DEVICE):
    print(f"Scanning directory: {dataset_dir}")
    classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    print(f"Found classes: {classes}")
    if not classes:
        print(f"ERROR: No valid class directories found in {dataset_dir}")
        sys.exit(1)

    vectors = []
    labels = []

    for class_name in tqdm(classes, desc="🔍 Đang xử lý class"):
        class_path = os.path.join(dataset_dir, class_name)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png'))]
        print(f"Class {class_name}: Found {len(images)} images")

        for img_file in images:
            img_path = os.path.join(class_path, img_file)
            vec = extract_features(img_path, model, preprocess, device)
            if vec is not None:
                vectors.append(vec)
                labels.append(class_name)

    if not vectors:
        print(f"ERROR: No valid images processed in {dataset_dir}")
        sys.exit(1)

    vectors_np = np.stack(vectors).astype("float32")
    print(f"✅ Đã trích xuất {len(vectors)} vector với shape {vectors_np.shape}")

    index = faiss.IndexFlatL2(vectors_np.shape[1])
    faiss.normalize_L2(vectors_np)
    index.add(vectors_np)

    faiss.write_index(index, INDEX_PATH)
    with open(LABELS_PATH, "wb") as f:
        pickle.dump(labels, f)

    print(f"✅ Đã xây xong chỉ mục FAISS với {len(labels)} ảnh.")

# ========== Chạy ==========
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Xây dựng chỉ mục FAISS từ tập dữ liệu ảnh")
    parser.add_argument("--dataset", type=str, default=DATASET_DIR, help="Đường dẫn thư mục dữ liệu")
    args = parser.parse_args()

    print(f"Starting build_index with dataset: {args.dataset}")
    build_index(args.dataset, model, preprocess, DEVICE)
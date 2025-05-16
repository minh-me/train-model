import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import faiss
from PIL import Image
import clip
import torch
from tqdm import tqdm
import pickle

model, preprocess = clip.load("ViT-B/32", device="cpu")
print("✅ CLIP loaded!")


# ========== CONFIG ==========
DATASET_DIR = "D:\\Izi\\train-model\\euro-coins\\train"  # Folder chứa ảnh tổ chức theo class
INDEX_PATH = "faiss_index.index"
LABELS_PATH = "labels.pkl"
DEVICE = "cpu"  # hoặc "cuda" nếu có GPU

# ========== Load CLIP ==========
model, preprocess = clip.load("ViT-B/32", device=DEVICE)

# ========== Extract vector từ 1 ảnh ==========
def extract_features(image_path):
    try:
        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            embedding = model.encode_image(image)
        return embedding.cpu().numpy()[0]
    except Exception as e:
        print(f"Lỗi ảnh {image_path}: {e}")
        return None

# ========== Build FAISS index ==========
def build_index(dataset_dir):
    vectors = []
    labels = []

    for class_name in tqdm(os.listdir(dataset_dir), desc="🔍 Đang xử lý class"):
        class_path = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_path): continue

        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            vec = extract_features(img_path)
            if vec is not None:
                vectors.append(vec)
                labels.append(class_name)

    # FAISS index
    vectors_np = np.stack(vectors).astype("float32")
    index = faiss.IndexFlatL2(vectors_np.shape[1])  # dùng cosine = normalize + L2
    faiss.normalize_L2(vectors_np)
    index.add(vectors_np)

    # Lưu
    faiss.write_index(index, INDEX_PATH)
    with open(LABELS_PATH, "wb") as f:
        pickle.dump(labels, f)

    print(f"✅ Đã xây xong index FAISS với {len(labels)} ảnh.")

# ========== Truy vấn ảnh tương tự ==========
def query_image(image_path, top_k=5):
    # Load index và label
    index = faiss.read_index(INDEX_PATH)
    with open(LABELS_PATH, "rb") as f:
        labels = pickle.load(f)

    vec = extract_features(image_path)
    if vec is None:
        return []

    vec = vec.astype("float32").reshape(1, -1)
    faiss.normalize_L2(vec)

    distances, indices = index.search(vec, top_k)
    results = [(labels[i], float(distances[0][idx])) for idx, i in enumerate(indices[0])]
    return results

# ========== Chạy ==========
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Tạo lại index từ dataset")
    parser.add_argument("--query", type=str, help="Đường dẫn ảnh cần truy vấn")
    args = parser.parse_args()

    if args.build:
        build_index(DATASET_DIR)
    elif args.query:
        result = query_image(args.query)
        print("📌 Kết quả gần nhất:")
        for label, score in result:
            print(f"  ➤ {label} (score: {score:.4f})")
    else:
        print("⚠️ Thêm --build để tạo index hoặc --query path.jpg để tìm class")

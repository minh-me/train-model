import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import faiss
from PIL import Image
import pickle
import torch  # Thêm import torch
# from build_index import build_index  # Absolute import

# ========== CONFIG ==========
# DATASET_DIR = "D:\\Izi\\train-model\\euro-coins\\train"  # Thư mục chứa ảnh
INDEX_PATH = "D:\\Izi\\train-model\\faiss_index.index"   # Đường dẫn tuyệt đối
LABELS_PATH = "D:\\Izi\\train-model\\labels.pkl"         # Đường dẫn tuyệt đối
DEVICE = "cpu"  # hoặc "cuda" nếu có GPU

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

# ========== Truy vấn ảnh tương tự ==========
def query_image(image_path, model, preprocess, top_k=5, device=DEVICE):
    # Kiểm tra file chỉ mục và nhãn
    # if not os.path.exists(INDEX_PATH) or not os.path.exists(LABELS_PATH):
    #     print(f"Chỉ mục ({INDEX_PATH}) hoặc nhãn ({LABELS_PATH}) không tồn tại. Xây dựng lại...")
    #     build_index(DATASET_DIR, model, preprocess, device)

    # Load chỉ mục và nhãn
    index = faiss.read_index(INDEX_PATH)
    with open(LABELS_PATH, "rb") as f:
        labels = pickle.load(f)

    # Trích xuất đặc trưng ảnh truy vấn
    vec = extract_features(image_path, model, preprocess, device)
    if vec is None:
        print(f"Lỗi: Không thể trích xuất đặc trưng từ {image_path}")
        return []

    # Chuẩn bị vector truy vấn
    vec = vec.astype("float32").reshape(1, -1)
    faiss.normalize_L2(vec)

    # Tìm kiếm top-K
    distances, indices = index.search(vec, top_k)
    
    # Chuyển đổi khoảng cách L2 thành độ tương đồng cosine
    similarities = [1 - (dist / 2) for dist in distances[0]]  # Cosine similarity ≈ 1 - L2^2/2
    results = [(labels[i], similarities[idx]) for idx, i in enumerate(indices[0])]
    
    return results


#   mark---karl-ix-second-serie-1607-1609
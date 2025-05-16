import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import faiss
from PIL import Image
from tqdm import tqdm
import pickle

# ========== CONFIG ==========
DATASET_DIR = "D:\\Izi\\train-model\\euro-coins\\train"  # Thư mục chứa ảnh tổ chức theo lớp
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

# ========== Xây dựng chỉ mục FAISS ==========
def build_index(dataset_dir, model, preprocess, device=DEVICE):
    vectors = []
    labels = []

    # Duyệt qua các lớp (thư mục con)
    for class_name in tqdm(os.listdir(dataset_dir), desc="🔍 Đang xử lý class"):
        class_path = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            vec = extract_features(img_path, model, preprocess, device)
            if vec is not None:
                vectors.append(vec)
                labels.append(class_name)

    # Kiểm tra nếu không có vector
    if not vectors:
        raise ValueError(f"Không tìm thấy ảnh hợp lệ trong {dataset_dir}. Kiểm tra thư mục hoặc định dạng ảnh.")

    # Chuyển thành mảng NumPy
    vectors_np = np.stack(vectors).astype("float32")
    print(f"✅ Đã trích xuất {len(vectors)} vector với shape {vectors_np.shape}")

    # Tạo chỉ mục FAISS
    index = faiss.IndexFlatL2(vectors_np.shape[1])  # Dùng L2, chuẩn hóa để tính cosine
    faiss.normalize_L2(vectors_np)
    index.add(vectors_np)

    # Lưu chỉ mục và nhãn
    faiss.write_index(index, INDEX_PATH)
    with open(LABELS_PATH, "wb") as f:
        pickle.dump(labels, f)

    print(f"✅ Đã xây xong chỉ mục FAISS với {len(labels)} ảnh.")
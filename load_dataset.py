import tensorflow as tf
import matplotlib.pyplot as plt

# Định nghĩa đường dẫn dataset
train_dir = "D:/yolo2/gemstone/train/"
valid_dir = "D:/yolo2/gemstone/valid/"

# Load dataset từ thư mục
batch_size = 32
img_size = (224, 224)  # Resize ảnh về 224x224

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=img_size,
    batch_size=batch_size
)

valid_dataset = tf.keras.utils.image_dataset_from_directory(
    valid_dir,
    image_size=img_size,
    batch_size=batch_size
)

# In danh sách class labels
class_names = train_dataset.class_names
print("Classes:", class_names)

# Hiển thị một số ảnh mẫu
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):  
    for i in range(9):  
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

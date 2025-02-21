import tensorflow as tf
from tensorflow.keras import layers, models
import pickle  # Thêm thư viện pickle để lưu history

# Định nghĩa đường dẫn dataset
train_dir = "D:/yolo2/gemstone/train/"
valid_dir = "D:/yolo2/gemstone/valid/"

batch_size = 32
img_size = (224, 224)


# Load dataset
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

class_names = train_dataset.class_names
print("Classes:", class_names)

# Chuẩn bị dataset
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
valid_dataset = valid_dataset.prefetch(buffer_size=AUTOTUNE)

# Xây dựng model CNN
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(224, 224, 3)),  # Chuẩn hóa ảnh

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(len(class_names), activation="softmax")  # Output: số class tương ứng
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"]) 

model.summary()

# Train model
history = model.fit(
    train_dataset,
    epochs=100,
    validation_data=valid_dataset
)

# Lưu model
model.save("car_classifier.keras")  
print("Model saved as car_classifier.h5")

with open("train_history.pkl", "wb") as f:
    pickle.dump(history.history, f)  # Chỉ lưu history.history
print("Training history saved as train_history.pkl")

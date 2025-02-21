import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model đã train
model = tf.keras.models.load_model("car_classifier.keras")

# Danh sách class labels
class_names = [
    "Alexandrite", "Almandine", "Amazonite", "Amber", "Amethyst", "Ametrine", "Andalusite", "Andradite",
    "Aquamarine", "Aventurine Green", "Aventurine Yellow", "Benitoite", "Beryl Golden", "Bixbite", "Bloodstone",
    "Blue Lace Agate", "Carnelian", "Cats Eye", "Chalcedony", "Chalcedony Blue", "Chrome Diopside", "Chrysoberyl",
    "Chrysocolla", "Chrysoprase", "Citrine", "Coral", "Danburite", "Diamond", "Diaspore", "Dumortierite", "Emerald",
    "Fluorite", "Garnet Red", "Goshenite", "Grossular", "Hessonite", "Hiddenite", "Iolite", "Jade", "Jasper",
    "Kunzite", "Kyanite", "Labradorite", "Lapis Lazuli", "Larimar", "Malachite", "Moonstone", "Morganite",
    "Onyx Black", "Onyx Green", "Onyx Red", "Opal", "Pearl", "Peridot", "Prehnite", "Pyrite", "Pyrope",
    "Quartz Beer", "Quartz Lemon", "Quartz Rose", "Quartz Rutilated", "Quartz Smoky", "Rhodochrosite", "Rhodolite",
    "Rhodonite", "Ruby", "Sapphire Blue", "Sapphire Pink", "Sapphire Purple", "Sapphire Yellow", "Scapolite",
    "Serpentine", "Sodalite", "Spessartite", "Sphene", "Spinel", "Spodumene", "Sunstone", "Tanzanite",
    "Tigers Eye", "Topaz", "Tourmaline", "Tsavorite", "Turquoise", "Variscite", "Zircon", "Zoisite"
]

def predict_image(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    class_name = class_names[class_index]

    print(f"Prediction: {class_name}")  # Chỉ in tên, không hiển thị ảnh

# Dự đoán với ảnh cụ thể
predict_image("D:/yolo2/gemstone/train/Alexandrite/alexandrite_5_jpg.rf.967d227ce399feba5d0b9b0dc6c11455.jpg", model)

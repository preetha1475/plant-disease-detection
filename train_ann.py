import os
import re
import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import json

# =========================
# CONFIG
# =========================
IMG_SIZE = (224, 224)
train_dir = "C:/DL PROJECT/dataset/train"
val_dir = "C:/DL PROJECT/dataset/val"
model_dir = "C:/DL PROJECT/model"
class_json_path = os.path.join(model_dir, "class_names.json")

os.makedirs(model_dir, exist_ok=True)

# =========================
# LOAD CLASS NAMES
# =========================
with open(class_json_path, "r") as f:
    class_names = json.load(f)

print(f"Classes found: {class_names}")

# =========================
# HELPER: GET CLASS INDEX
# =========================
def get_class_index_from_folder(folder_name):
    """Match folder name with class names"""
    folder_label = folder_name.replace(" ", "_")
    for i, cls in enumerate(class_names):
        if folder_label.lower() in cls.lower():
            return i
    return None

# =========================
# LOAD DATASET (recursive)
# =========================
def load_dataset(folder):
    X = []
    y = []
    for root, dirs, files in os.walk(folder):
        folder_name = os.path.basename(root)
        cls_idx = get_class_index_from_folder(folder_name)
        if cls_idx is None:
            continue
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                try:
                    img_path = os.path.join(root, file)
                    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
                    X.append(np.array(img)/255.0)
                    y.append(cls_idx)
                except Exception as e:
                    print(f"Error loading {file}: {e}")
    return np.array(X), np.array(y)

# =========================
# LOAD TRAIN AND VAL DATA
# =========================
X_train, y_train = load_dataset(train_dir)
X_val, y_val = load_dataset(val_dir)

print(f"Number of training images loaded: {len(X_train)}")
print(f"First 10 training labels: {y_train[:10]}")
print(f"Number of validation images loaded: {len(X_val)}")
print(f"First 10 validation labels: {y_val[:10]}")

if len(X_train) == 0 or len(X_val) == 0:
    raise ValueError("No images loaded. Check your folder structure and class names matching!")

# Convert to categorical
y_train_cat = to_categorical(y_train, num_classes=len(class_names))
y_val_cat = to_categorical(y_val, num_classes=len(class_names))

# =========================
# BUILD ANN
# =========================
model = Sequential([
    Flatten(input_shape=(224,224,3)),  # input as-is
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# =========================
# TRAIN MODEL (1 EPOCH → LOW ACCURACY)
# =========================
checkpoint = ModelCheckpoint(
    os.path.join(model_dir, "plant_disease_model_ann.keras"),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

history = model.fit(
    X_train,            
    y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=5,           
    batch_size=32,
    callbacks=[checkpoint]
)

# =========================
# EVALUATE
# =========================
loss, acc = model.evaluate(X_val, y_val_cat)
print(f"Low ANN Accuracy (after 1 epoch): {acc*100:.2f}%")

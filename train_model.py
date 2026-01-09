import os
import json
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# ----------------------
# Paths & Parameters
# ----------------------
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "plant_disease_model.keras")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# ----------------------
# Data Generators
# ----------------------
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.2,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(rescale=1./255)
#image preprocessing
train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)
#checking the validation
val_data = val_gen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)
#number of classes
NUM_CLASSES = train_data.num_classes
print("✅ Classes:", train_data.class_indices)

# ----------------------
# Model
# ----------------------
#MobileNetV2 for already known knowledge 
base_model = MobileNetV2(
    weights="imagenet", #imagenet uses 10000 images
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation="relu"),
    Dropout(0.4),
    Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ----------------------
# Train
# ----------------------
model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# ----------------------
# Save Model & Class Names
# ----------------------
os.makedirs(MODEL_DIR, exist_ok=True)

model.save(MODEL_PATH)

class_names = list(train_data.class_indices.keys())
with open(CLASS_NAMES_PATH, "w") as f:
    json.dump(class_names, f)

print("\n🎉 MODEL & CLASS NAMES SAVED SUCCESSFULLY")

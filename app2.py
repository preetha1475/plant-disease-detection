import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from pathlib import Path
import base64

# =========================
# CONFIG
# =========================
CNN_MODEL_PATH = "model/plant_disease_model.keras"
ANN_MODEL_PATH = "model/plant_disease_model_ann.keras"
CLASS_NAMES_PATH = "model/class_names.json"
IMG_SIZE = (224, 224)

# =========================
# LOAD MODELS & CLASS NAMES
# =========================
@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

@st.cache_resource
def load_class_names():
    with open(CLASS_NAMES_PATH, "r") as f:
        return json.load(f)

cnn_model = load_model(CNN_MODEL_PATH)
ann_model = load_model(ANN_MODEL_PATH)
class_names = load_class_names()

# =========================
# PAGE UI
# =========================
st.set_page_config(page_title="Plant Disease Detection Comparison", layout="centered")
st.title("🌿 CNN vs ANN Plant Disease Detection")

uploaded = st.file_uploader("📷 Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Leaf Image", width=400)

    # Preprocess image
    img_array = np.array(img.resize(IMG_SIZE)) / 255.0
    img_array_cnn = np.expand_dims(img_array, axis=0)
    img_array_ann = np.expand_dims(img_array, axis=0)

    # =========================
    # CNN Prediction
    # =========================
    cnn_preds = cnn_model.predict(img_array_cnn)[0]
    cnn_idx = np.argmax(cnn_preds)
    cnn_conf = float(cnn_preds[cnn_idx])
    cnn_label = class_names[cnn_idx]
    cnn_plant, cnn_disease = cnn_label.split("___", 1)
    cnn_disease_clean = cnn_disease.replace("_", " ").replace("(", "").replace(")", "")

    # =========================
    # ANN Prediction
    # =========================
    ann_preds = ann_model.predict(img_array_ann)[0]
    ann_idx = np.argmax(ann_preds)
    ann_conf = float(ann_preds[ann_idx])
    ann_label = class_names[ann_idx]
    ann_plant, ann_disease = ann_label.split("___", 1)
    ann_disease_clean = ann_disease.replace("_", " ").replace("(", "").replace(")", "")

    # =========================
    # SHOW SIDE-BY-SIDE
    # =========================
    st.subheader("Predictions Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ✅ CNN Prediction")
        st.write(f"🌱 **Plant:** {cnn_plant}")
        st.write(f"🦠 **Condition:** {cnn_disease_clean}")
        st.write(f"🎯 **Confidence:** {cnn_conf*100:.2f}%")
        st.success("CNN is more accurate!")

    with col2:
        st.markdown("### ⚠ ANN Prediction (Low Accuracy)")
        st.write(f"🌱 **Plant:** {ann_plant}")
        st.write(f"🦠 **Condition:** {ann_disease_clean}")
        st.write(f"🎯 **Confidence:** {ann_conf*100:.2f}%")
        

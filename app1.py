import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from pathlib import Path
import base64


MODEL_PATH = "model/plant_disease_model.keras"
CLASS_NAMES_PATH = "model/class_names.json"
IMG_SIZE = (224, 224)


st.set_page_config(page_title="Plant Disease Detection", layout="centered")


leaf_path = Path("C:/dl project/leaf.png")
with open(leaf_path, "rb") as f:
    leaf_base64 = base64.b64encode(f.read()).decode()


st.markdown(
    f"""
    <style>
    @keyframes fallCornerLeaves {{
        0% {{ transform: translateY(-50px) rotate(0deg); opacity: 0; }}
        10% {{ opacity: 1; }}
        100% {{ transform: translateY(100vh) rotate(360deg); opacity: 0; }}
    }}

    .corner-leaf {{
        position: fixed;
        width: 60px;
        height: 60px;
        background-image: url("data:image/png;base64,{leaf_base64}");
        background-size: contain;
        background-repeat: no-repeat;
        z-index: 9999;
        animation-name: fallCornerLeaves;
        animation-timing-function: linear;
        animation-iteration-count: infinite;
        pointer-events: none;
    }}

    .leaf-left1 {{ left: 5px; animation-duration: 8s; animation-delay: 0s; }}
    .leaf-left2 {{ left: 15px; animation-duration: 10s; animation-delay: 2s; }}
    .leaf-left3 {{ left: 25px; animation-duration: 9s; animation-delay: 1s; }}
    .leaf-right1 {{ right: 5px; animation-duration: 8s; animation-delay: 1s; }}
    .leaf-right2 {{ right: 15px; animation-duration: 11s; animation-delay: 0s; }}
    .leaf-right3 {{ right: 25px; animation-duration: 9s; animation-delay: 2s; }}

    .title-box {{
        background-color: #b3e6b3; 
        color: #06470c; 
        padding: 30px; 
        border-radius: 15px; 
        text-align: center; 
        font-size: 36px; 
        font-weight: bold; 
        margin-bottom: 30px; 
    }}

    .uploader-box {{
        background-color: #e0f7e0; 
        padding: 25px; 
        border-radius: 15px; 
        margin-bottom: 25px; 
        font-size: 20px; 
    }}

    .result-box {{
        background-color: #d8f0d8; 
        padding: 25px; 
        border-radius: 15px; 
        margin-top: 20px; 
        margin-bottom: 20px; 
        font-size: 20px; 
        line-height: 1.6;
    }}

    .result-box h2, .result-box h3, .result-box h4 {{
        font-size: 28px !important;
        font-weight: bold;
    }}

    .result-box b, .result-box strong {{
        font-size: 22px !important;
    }}

    .result-box a {{
        font-size: 20px !important;
    }}

    .st-image > img {{
        max-width: 450px !important;
    }}
    </style>

    <div class="corner-leaf leaf-left1"></div>
    <div class="corner-leaf leaf-left2"></div>
    <div class="corner-leaf leaf-left3"></div>
    <div class="corner-leaf leaf-right1"></div>
    <div class="corner-leaf leaf-right2"></div>
    <div class="corner-leaf leaf-right3"></div>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource
def load_class_names():
    with open(CLASS_NAMES_PATH, "r") as f:
        return json.load(f)

model = load_model()
class_names = load_class_names()

DISEASE_INFO = {
    "healthy": {
        "cause": "No disease detected. The leaf shows normal color, texture, and shape.",
        "cure": "Maintain proper watering, sunlight, soil nutrition, and pest control."
    },
    "apple scab": {
        "cause": "Fungal infection caused by Venturia inaequalis, common in wet climates.",
        "cure": "Remove infected leaves and apply captan or mancozeb fungicide.",
        "link": "https://extension.umn.edu/plant-diseases/apple-scab"
    },
    "black rot": {
        "cause": "Fungal infection entering through wounds or cracks in plant tissue.",
        "cure": "Prune infected branches and apply copper-based fungicides.",
        "link": "https://extension.psu.edu/black-rot-on-grapes-in-home-gardens"
    },
    "powdery mildew": {
        "cause": "Fungal disease favored by dry leaves and humid air conditions.",
        "cure": "Apply neem oil, potassium bicarbonate, or sulfur fungicide.",
        "link": "https://treefruit.wsu.edu/crop-protection/disease-management/cherry-powdery-mildew/"
    },
    "early blight": {
        "cause": "Caused by Alternaria fungi due to moisture stress and poor airflow.",
        "cure": "Use chlorothalonil fungicide and rotate crops regularly.",
        "link": "https://www.cropscience.bayer.us/articles/cp/early-blight-potatoes"
    },
    "late blight": {
        "cause": "Caused by Phytophthora infestans in cool, wet conditions.",
        "cure": "Destroy infected plants and apply metalaxyl fungicide early.",
        "link": "https://www.cabi.org/isc/datasheet/40970"
    },
    "common rust": {
        "cause": "Caused by the fungus Puccinia sorghi, common in corn under moist conditions.",
        "cure": "Remove infected leaves, improve airflow, and apply fungicide if severe.",
        "link": "https://cropprotectionnetwork.org/encyclopedia/common-rust-of-corn"
    },
    "northern leaf blight": {
        "cause": "Fungal disease caused by Exserohilum turcicum in corn plants.",
        "cure": "Use resistant hybrids and apply azoxystrobin fungicide.",
        "link": "https://cropprotectionnetwork.org/encyclopedia/northern-corn-leaf-blight"
    }
}

# =========================
# STREAMLIT UI
# =========================
st.markdown('<div class="title-box">🌿 Explainable AI Plant Disease Detection & Advisory System</div>', unsafe_allow_html=True)

st.markdown('<div class="uploader-box">', unsafe_allow_html=True)
uploaded = st.file_uploader("📷 Upload a leaf image", type=["jpg", "jpeg", "png"])
st.markdown('</div>', unsafe_allow_html=True)

if uploaded:
    st.markdown('<div class="result-box">', unsafe_allow_html=True)

    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Leaf Image", width=450)

    # Preprocess and predict
    img_array = np.array(img.resize(IMG_SIZE)) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)[0]
    idx = np.argmax(preds)
    confidence = float(preds[idx])
    label = class_names[idx]
    plant, disease = label.split("___", 1)
    disease_clean = disease.replace("_", " ").replace("(", "").replace(")", "")
    disease_key = disease_clean.lower()

    if "rust" in disease_key:
        disease_key = "common rust"
    elif "northern" in disease_key and "blight" in disease_key:
        disease_key = "northern leaf blight"
    elif "healthy" in disease_key:
        disease_key = "healthy"

    info = DISEASE_INFO.get(disease_key, None)

    st.success("✅ Prediction Completed")
    st.write(f"🌱 **Plant:** {plant}")
    st.write(f"🦠 **Condition:** {disease_clean}")

    st.subheader("🔍 Explanation (Explainable AI)")
    if disease_key == "healthy":
        st.write("The model identified uniform green color, smooth texture, and absence of lesions, indicating a healthy leaf.")
    else:
        st.write("The model focused on discolored regions, lesion patterns, and irregular textures commonly associated with plant diseases.")

    st.subheader("🌿 Advisory & Cure")
    if info:
        st.write(f"🧬 **Cause:** {info['cause']}")
        st.info(f"💊 **Cure / Care:** {info['cure']}")
        if disease_key != "healthy" and "link" in info:
            st.markdown(f'🔗 <a href="{info["link"]}" target="_blank">Learn more about this disease</a>', unsafe_allow_html=True)
    else:
        st.warning("No advisory information available for this disease.")

    if confidence < 0.70:
        st.warning("⚠ Low confidence prediction. Please upload a clearer image with a visible leaf.")

    st.markdown('</div>', unsafe_allow_html=True)

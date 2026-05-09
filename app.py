import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from PIL import Image, ImageOps
import numpy as np

# =========================================
# CONFIG
# =========================================
st.set_page_config(
    page_title="Deteksi Sayuran AI",
    page_icon="🥬"
)

# =========================================
# LOAD MODEL
# =========================================
@st.cache_resource
def load_model():

    # Load architecture
    with open("model.json", "r") as json_file:
        loaded_model_json = json_file.read()

    model = model_from_json(loaded_model_json)

    # Load weights
    model.load_weights("model.weights.h5")

    return model

model = load_model()

# =========================================
# LABEL KELAS
# =========================================
class_names = ['Cabai', 'Terong', 'Tomat']

# =========================================
# TITLE
# =========================================
st.title("🥬 Klasifikasi Sayuran AI")

st.write(
    "Upload gambar Cabai, Terong, atau Tomat."
)

# =========================================
# UPLOAD GAMBAR
# =========================================
uploaded_file = st.file_uploader(
    "Pilih gambar",
    type=["jpg", "jpeg", "png"]
)

# =========================================
# PREDIKSI
# =========================================
def predict_image(image):

    size = (224, 224)

    image = ImageOps.fit(
        image,
        size,
        Image.Resampling.LANCZOS
    )

    img = np.asarray(image)

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    return prediction

# =========================================
# HASIL
# =========================================
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, use_container_width=True)

    prediction = predict_image(image)

    index = np.argmax(prediction)

    label = class_names[index]

    confidence = prediction[0][index] * 100

    st.success(f"Hasil Prediksi: {label}")

    st.info(f"Tingkat Keyakinan: {confidence:.2f}%")

else:
    st.warning("Silakan upload gambar.")
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import zipfile
import os

# =========================================
# CONFIG
# =========================================
st.set_page_config(
    page_title="Deteksi Sayuran AI",
    page_icon="🥬"
)

# =========================================
# EXTRACT MODEL
# =========================================
if not os.path.exists("saved_model"):

    with zipfile.ZipFile("saved_model.zip", 'r') as zip_ref:
        zip_ref.extractall("saved_model")

# =========================================
# LOAD MODEL
# =========================================
@st.cache_resource
def load_model():

    model = tf.saved_model.load("saved_model")

    return model

model = load_model()

# =========================================
# LABEL
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
# UPLOAD
# =========================================
uploaded_file = st.file_uploader(
    "Upload gambar",
    type=["jpg", "jpeg", "png"]
)

# =========================================
# PREPROCESS
# =========================================
def preprocess_image(image):

    size = (224, 224)

    image = ImageOps.fit(
        image,
        size,
        Image.Resampling.LANCZOS
    )

    img = np.asarray(image)

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    return img.astype(np.float32)

# =========================================
# HASIL
# =========================================
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, use_container_width=True)

    img = preprocess_image(image)

    infer = model.signatures["serving_default"]

    prediction = infer(tf.constant(img))

    prediction = list(prediction.values())[0].numpy()

    index = np.argmax(prediction)

    label = class_names[index]

    confidence = prediction[0][index] * 100

    st.success(f"Hasil Prediksi: {label}")

    st.info(f"Tingkat Keyakinan: {confidence:.2f}%")

else:
    st.warning("Silakan upload gambar.")

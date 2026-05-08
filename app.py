import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# =========================================
# CONFIG HALAMAN
# =========================================
st.set_page_config(
    page_title="Deteksi Sayuran AI",
    page_icon="🥬",
    layout="centered"
)

# =========================================
# LOAD MODEL
# =========================================
@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "model_sayur.keras",
        compile=False
    )

    return model

model = load_model()

# =========================================
# NAMA KELAS
# =========================================
class_names = ['Cabai', 'Terong', 'Tomat']

# =========================================
# TAMPILAN
# =========================================
st.title("🥬 Klasifikasi Sayuran AI")

st.write(
    "Website AI untuk mendeteksi gambar Cabai, Terong, dan Tomat menggunakan Transfer Learning."
)

# =========================================
# UPLOAD FILE
# =========================================
uploaded_file = st.file_uploader(
    "Upload gambar sayuran",
    type=["jpg", "jpeg", "png"]
)

# =========================================
# FUNGSI PREDIKSI
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

    st.image(
        image,
        caption="Gambar Upload",
        use_container_width=True
    )

    prediction = predict_image(image)

    index = np.argmax(prediction)

    label = class_names[index]

    confidence = prediction[0][index] * 100

    st.success(f"Hasil Prediksi: {label}")

    st.info(f"Tingkat Keyakinan: {confidence:.2f}%")

else:
    st.warning("Silakan upload gambar terlebih dahulu.")
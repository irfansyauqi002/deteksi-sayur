import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import model_from_json
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
# CUSTOM CSS UI MODERN
# =========================================
st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #2E8B57;
    margin-top: 10px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #666666;
    margin-bottom: 30px;
}

.upload-box {
    padding: 20px;
    border-radius: 15px;
    background-color: white;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    background: linear-gradient(135deg, #d4fc79, #96e6a1);
    color: black;
    margin-top: 20px;
    text-align: center;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
}

.result-title {
    font-size: 20px;
    font-weight: bold;
}

.result-label {
    font-size: 35px;
    font-weight: bold;
    margin-top: 10px;
}

.result-confidence {
    font-size: 18px;
    margin-top: 10px;
}

.stButton>button {
    border-radius: 10px;
    background-color: #2E8B57;
    color: white;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

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
# HEADER
# =========================================
st.markdown(
    '<div class="title">🥬 Klasifikasi Sayuran AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Deteksi Cabai, Terong, dan Tomat menggunakan Artificial Intelligence</div>',
    unsafe_allow_html=True
)

# =========================================
# UPLOAD FILE
# =========================================
st.markdown('<div class="upload-box">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📤 Upload gambar sayuran",
    type=["jpg", "jpeg", "png"]
)

st.markdown('</div>', unsafe_allow_html=True)

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
# HASIL PREDIKSI
# =========================================
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="📷 Gambar Upload",
        use_container_width=True
    )

    prediction = predict_image(image)

    index = np.argmax(prediction)

    label = class_names[index]

    confidence = prediction[0][index] * 100

    st.markdown(f"""
    <div class="result-box">

    <div class="result-title">
    Hasil Prediksi
    </div>

    <div class="result-label">
    {label}
    </div>

    <div class="result-confidence">
    Tingkat Keyakinan: {confidence:.2f}%
    </div>

    </div>
    """, unsafe_allow_html=True)

    # =========================================
    # DOWNLOAD HASIL
    # =========================================
    hasil = f"""
    HASIL KLASIFIKASI SAYURAN AI

    Hasil Prediksi : {label}

    Tingkat Keyakinan : {confidence:.2f}%
    """

    st.download_button(
        label="📥 Download Hasil Prediksi",
        data=hasil,
        file_name="hasil_prediksi.txt",
        mime="text/plain"
    )

else:
    st.warning("⚠️ Silakan upload gambar terlebih dahulu.")

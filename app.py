import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import zipfile
import os

# =========================================
# KONFIGURASI HALAMAN
# =========================================
st.set_page_config(
    page_title="Deteksi Sayuran AI",
    page_icon="🥬",
    layout="centered"
)

# =========================================
# CUSTOM CSS
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
    color: gray;
    margin-bottom: 30px;
}

.upload-box {
    background-color: white;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}

.result-box {
    background: linear-gradient(135deg, #56ab2f, #a8e063);
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    margin-top: 20px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
}

.unknown-box {
    background: linear-gradient(135deg, #ffb199, #ff0844);
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    margin-top: 20px;
    color: white;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
}

.result-label {
    font-size: 35px;
    font-weight: bold;
}

.result-confidence {
    font-size: 18px;
}

.history-box {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# =========================================
# EXTRACT MODEL
# =========================================
if not os.path.exists("saved_model"):

    with zipfile.ZipFile("saved_model.zip", "r") as zip_ref:
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
# NAMA KELAS
# =========================================
class_names = [
    "Non-Sayuran",
    "Sayuran"
]

# =========================================
# HEADER
# =========================================
st.markdown(
    '<div class="title">🥬 Deteksi Sayuran AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Artificial Intelligence untuk mendeteksi Sayuran dan Non-Sayuran</div>',
    unsafe_allow_html=True
)

# =========================================
# SESSION HISTORY
# =========================================
if "history" not in st.session_state:

    st.session_state.history = []

# =========================================
# PILIH INPUT
# =========================================
option = st.radio(
    "Pilih Metode Input",
    ["Upload Gambar", "Kamera Realtime"]
)

# =========================================
# INPUT GAMBAR
# =========================================
image = None

st.markdown('<div class="upload-box">', unsafe_allow_html=True)

if option == "Upload Gambar":

    uploaded_file = st.file_uploader(
        "📤 Upload gambar",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

else:

    camera_image = st.camera_input("📷 Ambil gambar")

    if camera_image is not None:

        image = Image.open(camera_image).convert("RGB")

st.markdown('</div>', unsafe_allow_html=True)

# =========================================
# PREPROCESS GAMBAR
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
# PREDIKSI
# =========================================
if image is not None:

    st.image(
        image,
        caption="📷 Gambar Input",
        use_container_width=True
    )

    img = preprocess_image(image)

    infer = model.signatures["serving_default"]

    prediction = infer(tf.constant(img))

    prediction = list(prediction.values())[0].numpy()

    confidence = float(prediction[0][0])

    # =========================================
    # LABEL
    # =========================================
    if confidence >= 0.5:

        label = "Sayuran"

        final_confidence = confidence * 100

    else:

        label = "Non-Sayuran"

        final_confidence = (1 - confidence) * 100

    # =========================================
    # HASIL
    # =========================================
    if label == "Sayuran":

        st.markdown(f"""
        <div class="result-box">

        <h2>✅ Hasil Prediksi</h2>

        <div class="result-label">
        {label}
        </div>

        <br>

        <div class="result-confidence">
        Tingkat Keyakinan: {final_confidence:.2f}%
        </div>

        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown(f"""
        <div class="unknown-box">

        <h2>⚠️ Hasil Prediksi</h2>

        <div class="result-label">
        {label}
        </div>

        <br>

        <div class="result-confidence">
        Tingkat Keyakinan: {final_confidence:.2f}%
        </div>

        </div>
        """, unsafe_allow_html=True)

    # =========================================
    # CONFIDENCE BAR
    # =========================================
    st.subheader("📊 Confidence")

    sayuran_score = confidence * 100

    non_score = (1 - confidence) * 100

    st.write(f"🥬 Sayuran : {sayuran_score:.2f}%")
    st.progress(int(sayuran_score))

    st.write(f"❌ Non-Sayuran : {non_score:.2f}%")
    st.progress(int(non_score))

    # =========================================
    # HISTORY
    # =========================================
    st.session_state.history.append(
        f"{label} ({final_confidence:.2f}%)"
    )

    # =========================================
    # DOWNLOAD HASIL
    # =========================================
    hasil = f"""
HASIL DETEKSI SAYURAN AI

Hasil Prediksi : {label}

Tingkat Keyakinan : {final_confidence:.2f}%
"""

    st.download_button(
        label="📥 Download Hasil Prediksi",
        data=hasil,
        file_name="hasil_prediksi.txt",
        mime="text/plain"
    )

# =========================================
# HISTORY PREDIKSI
# =========================================
if len(st.session_state.history) > 0:

    st.markdown("""
    <div class="history-box">
    <h3>📜 Riwayat Prediksi</h3>
    </div>
    """, unsafe_allow_html=True)

    for item in reversed(st.session_state.history):

        st.write("✅", item)

else:

    st.warning("⚠️ Silakan upload gambar atau gunakan kamera.")

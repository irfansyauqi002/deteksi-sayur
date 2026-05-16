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
    layout="wide"  # Mengubah ke wide agar layout kolom lebih maksimal
)

# =========================================
# CUSTOM CSS (Modern & Clean)
# =========================================
st.markdown("""
<style>
    /* Styling text judul */
    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(45deg, #2E8B57, #a8e063);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-title {
        text-align: center;
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    
    /* Box container hasil */
    .custom-card {
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    .custom-card:hover {
        transform: translateY(-2px);
    }
    .bg-vegetable {
        background: linear-gradient(135deg, #11998e, #38ef7d);
    }
    .bg-non-vegetable {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
    }
    
    /* Styling teks di dalam kartu */
    .card-label {
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 0.5rem;
    }
    .card-confidence {
        font-size: 1.1rem;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

# =========================================
# EXTRACT & LOAD MODEL
# =========================================
if not os.path.exists("saved_model"):
    with zipfile.ZipFile("saved_model.zip", "r") as zip_ref:
        zip_ref.extractall("saved_model")

@st.cache_resource
def load_model():
    return tf.saved_model.load("saved_model")

model = load_model()

# =========================================
# SESSION HISTORY
# =========================================
if "history" not in st.session_state:
    st.session_state.history = []

# =========================================
# HEADER
# =========================================
st.markdown('<p class="main-title">🥬 Deteksi Sayuran AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Klasifikasi pintar berbasis Deep Learning untuk membedakan Sayuran dan Non-Sayuran</p>', unsafe_allow_html=True)

# =========================================
# UTAMA: LAYOUT 2 KOLOM
# =========================================
col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("📥 Input Gambar")
    
    # Menggunakan komponen Tabs bawaan Streamlit agar UI terlihat bersih
    tab_upload, tab_camera = st.tabs(["📤 Upload File", "📷 Gunakan Kamera"])
    
    image = None
    
    with tab_upload:
        uploaded_file = st.file_uploader(
            "Pilih gambar dari perangkat Anda",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            
    with tab_camera:
        camera_image = st.camera_input("Ambil foto langsung", label_visibility="collapsed")
        if camera_image is not None:
            image = Image.open(camera_image).convert("RGB")

    # Tampilkan gambar preview tepat di bawah input
    if image is not None:
        st.write("")
        st.image(image, caption="Gambar yang akan dianalisis", use_container_width=True)

# =========================================
# PREPROCESS & PREDIKSI fungsi
# =========================================
def preprocess_image(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img = np.asarray(image) / 255.0
    img = np.expand_dims(img, axis=0)
    return img.astype(np.float32)

# =========================================
# PROSES HASIL (KOLOM KANAN)
# =========================================
with col_result:
    st.subheader("📊 Hasil Analisis")
    
    if image is not None:
        with st.spinner("Sedang menganalisis gambar..."):
            img = preprocess_image(image)
            infer = model.signatures["serving_default"]
            prediction = infer(tf.constant(img))
            prediction = list(prediction.values())[0].numpy()
            
            confidence = float(prediction[0][0])
            
            # Penentuan Label
            if confidence >= 0.5:
                label = "Sayuran"
                final_confidence = confidence * 100
                card_class = "bg-vegetable"
                icon = "✅"
            else:
                label = "Non-Sayuran"
                final_confidence = (1 - confidence) * 100
                card_class = "bg-non-vegetable"
                icon = "⚠️"
            
            # Simpan ke riwayat jika belum ada (mencegah duplikasi saat rerun)
            history_string = f"{label} ({final_confidence:.2f}%)"
            if not st.session_state.history or st.session_state.history[-1] != history_string:
                st.session_state.history.append(history_string)
                
        # Card Hasil Prediksi Modern
        st.markdown(f"""
        <div class="custom-card {card_class}">
            <div>{icon} HASIL PREDIKSI</div>
            <div class="card-label">{label}</div>
            <div class="card-confidence">Akurasi: {final_confidence:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Detail Probabilitas dengan Progress Bar bawaan
        with st.container(border=True):
            st.markdown("**Detail Probabilitas Kelas:**")
            sayuran_score = confidence * 100
            non_score = (1 - confidence) * 100
            
            col_v, col_nv = st.columns(2)
            col_v.metric("🥬 Sayuran", f"{sayuran_score:.1f}%")
            col_nv.metric("❌ Non-Sayuran", f"{non_score:.1f}%")
            
            st.progress(int(sayuran_score))
            
        # Tombol download diletakkan di bawah dengan rapi
        st.write("")
        hasil_teks = f"HASIL DETEKSI AI\n----------------\nPrediksi: {label}\nConfidence: {final_confidence:.2f}%"
        st.download_button(
            label="📥 Download Laporan Hasil (.txt)",
            data=hasil_teks,
            file_name=f"hasil_deteksi_{label.lower()}.txt",
            mime="text/plain",
            use_container_width=True
        )
    else:
        # Tampilan state kosong yang estetik saat user belum memasukkan gambar
        st.info("💡 Menunggu unggahan atau tangkapan gambar di sebelah kiri untuk memulai analisis.")

# =========================================
# FOOTER & RIWAYAT (DI BAWAH LAYOUT UTAMA)
# =========================================
st.divider()

if st.session_state.history:
    with st.expander("📜 Lihat Riwayat Analisis Sesi Ini", expanded=False):
        # Tampilkan riwayat dalam bentuk list/grid terbalik (terbaru di atas)
        for idx, item in enumerate(reversed(st.session_state.history)):
            st.markdown(f"**{idx+1}.** {item}")

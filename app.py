import streamlit as st
import torch
import numpy as np
from PIL import Image

from src.cabai.config import CLASS_NAMES, DISPLAY_NAMES, CHECKPOINTS_DIR
from src.cabai.model import create_model
from src.cabai.data import get_transforms
from src.cabai.recommend import format_recommendation_markdown
from src.cabai.gradcam import generate_gradcam

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="CabAI - Klasifikasi Penyakit Cabai",
    page_icon="🌶️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Deteksi Device ---
@st.cache_resource
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = get_device()

# --- Load Model (Cached) ---
@st.cache_resource(show_spinner="Memuat model...")
def load_model():
    checkpoint_path = CHECKPOINTS_DIR / 'efficientnet_b0_demo.pt'
    if checkpoint_path.exists():
        # Solusi Self-Trained: Memuat model hasil fine-tuning
        model = create_model(model_name="efficientnet_b0", num_classes=5, pretrained=False)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model, True
    else:
        # Fallback ke model baseline (pretrained ImageNet) untuk keperluan demo progress jika training belum selesai
        model = create_model(model_name="efficientnet_b0", num_classes=5, pretrained=True)
        model.to(device)
        model.eval()
        return model, False

model, is_trained = load_model()
transform = get_transforms(train=False)

# --- UI Sidebar (Spesifikasi Tugas Besar) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004143.png", width=100) # Placeholder logo
    st.title("Tentang CabAI")
    st.markdown("Sistem Klasifikasi Penyakit Cabai Berbasis Computer Vision.")
    
    st.divider()
    st.markdown("### Pemenuhan Spesifikasi")
    
    st.markdown("**☁️ 1. Solusi AIaaS**")
    st.caption("Aplikasi web *hosted* yang siap didemonstrasikan ke publik tanpa instalasi lokal (Streamlit Community Cloud).")
    
    st.markdown("**🧠 2. Self-Trained ML**")
    if is_trained:
        st.success("✅ Menggunakan model *EfficientNet-B0* yang di-fine-tune dengan dataset klasifikasi daun cabai.")
    else:
        st.warning("⚠️ Menjalankan purwarupa dengan model *Baseline*. Model *Self-Trained* akan tersedia setelah pipeline training selesai.")
    
    st.markdown("**💡 3. Solusi AI Bebas**")
    st.caption("Memadukan *Explainability* (Grad-CAM) untuk menunjukkan area gambar yang dianalisis, dan *Sistem Rekomendasi* (Knowledge Base) untuk penanganan penyakit cabai.")
    
    st.divider()
    st.caption("Tugas Besar II4012 | Inteligensi Artifisial untuk Bisnis")

# --- UI Header ---
st.title("🌶️ CabAI: Identifikasi Penyakit Daun Cabai")
st.markdown("""
Unggah foto daun cabai yang ingin diperiksa, dan sistem akan memberikan prediksi penyakit beserta rekomendasi tindakan penanganan awal.
""")

if not is_trained:
    st.info("Pemberitahuan Demo Progress: Hasil prediksi mungkin masih acak karena aplikasi saat ini menggunakan model baseline purwarupa.")

# --- File Uploader ---
st.markdown("### 📷 Unggah Foto")
uploaded_file = st.file_uploader("Pilih gambar daun cabai (.jpg, .jpeg, .png)...", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")

if uploaded_file is not None:
    # --- Proses Gambar ---
    image = Image.open(uploaded_file).convert("RGB")
    
    # Tampilkan progress
    with st.spinner("Menganalisis gambar menggunakan CabAI..."):
        # Preprocessing tensor untuk model
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Preprocessing gambar numpy (0-1) untuk overlay Grad-CAM (ukuran 224x224 sesuai input size config)
        img_resized = image.resize((224, 224))
        rgb_img = np.float32(img_resized) / 255.0
        
        # --- Inferensi ---
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
        # Ambil kelas dengan probabilitas tertinggi
        confidence, class_idx = torch.max(probabilities, dim=0)
        confidence = confidence.item()
        class_idx = class_idx.item()
        predicted_class = CLASS_NAMES[class_idx]
        display_class_name = DISPLAY_NAMES.get(predicted_class, predicted_class)
        
        # --- Generate Grad-CAM ---
        cam_image = generate_gradcam(model, input_tensor, rgb_img, target_category=class_idx)
        
    # --- Tampilkan Hasil ---
    st.divider()
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 🔍 Hasil Analisis")
        st.image(image, caption="Gambar Asli yang Diunggah", use_container_width=True)
        
        st.markdown("#### Area Perhatian Model (Grad-CAM)")
        st.image(cam_image, caption="Heatmap menandakan area daun yang difokuskan oleh AI", use_container_width=True)

    with col2:
        st.markdown("### 📊 Prediksi Penyakit")
        st.markdown(f"<h2 style='color: #E63946;'>{display_class_name.title()}</h2>", unsafe_allow_html=True)
        
        # Format progress bar color based on confidence
        st.progress(confidence, text=f"Tingkat Kepercayaan Model: {confidence:.1%}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("### 📝 Rekomendasi Penanganan")
        with st.container(border=True):
            recommendation_md = format_recommendation_markdown(predicted_class, confidence)
            st.markdown(recommendation_md)

# --- Footer ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.caption("Peringatan: Sistem ini dirancang untuk bantuan identifikasi awal dan bukan merupakan pengganti observasi oleh ahli agronomi.")

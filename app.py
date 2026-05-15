import os
import base64
import streamlit as st
import torch
import numpy as np
from PIL import Image

from src.cabai.config import CLASS_NAMES, DISPLAY_NAMES, CHECKPOINTS_DIR
from src.cabai.model import create_model
from src.cabai.data import get_transforms
from src.cabai.recommend import get_recommendation
from src.cabai.gradcam import generate_gradcam
from src.cabai.explain import generate_explanation_gemini, answer_followup_question, _fallback_explanation

# Helper for Base64 
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Helper to render HTML safely (strips leading whitespace per line to avoid
# Streamlit treating indented HTML as a Markdown code block)
def render_html(html: str):
    cleaned = "\n".join(line.lstrip() for line in html.splitlines())
    st.markdown(cleaned, unsafe_allow_html=True)

logo_base64 = get_base64_of_bin_file("static\logo_cabAI.png")

# Page Config
st.set_page_config(
    page_title="CabAI – Identifikasi Penyakit Tanaman Cabai",
    page_icon="🌶️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
@import url('https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css');

.bi {
    vertical-align: -0.125em;
    fill: currentColor;
}

/* ── Override Streamlit theme-injected colors ── */
#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="collapsedControl"] { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }

/* Force the entire Streamlit app background to light grey */
.stApp, [data-testid="stAppViewContainer"] {
    background-color: #f7fef9 !important;
}

/* Force Streamlit header bar (dark green from config) to be invisible */
[data-testid="stHeader"] {
    background-color: transparent !important;
    visibility: hidden !important;
    height: 0 !important;
}

/* Reset block container */
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
    background-color: #f7fef9 !important;
}

/* Force all text color to dark */
.stApp, .stApp p, .stApp span, .stApp label, .stApp div {
    color: #1A1A2E;
}

* { font-family: 'Inter', sans-serif; box-sizing: border-box; }

/* ── Navbar ── */
.cabai-nav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 85px; height: 64px;
    background: #FFFFFF; border-bottom: 1px solid #e8e8e8;
    position: fixed; top: 0; left: 0; z-index: 1000; width: 100%;
    box-shadow: 0 1px 4px rgba(0,0,0,.04);
}
.cabai-nav .logo img {
    width: 120px; height: auto; object-fit: contain; padding-top: 5px;
}
.cabai-nav .links { display:flex; gap:48px; position: absolute; left:50%; transform: translateX(-50%); }
.cabai-nav .links a { text-decoration:none; color:#555; font-size:15px; font-weight:500; transition:color .2s; }
.cabai-nav .links a.active { color:#C82121; border-bottom:2px solid #C82121; padding-bottom:2px; }
.cabai-nav .links a:hover { color:#C82121; }
.nav-cta {
    background:#062C1B; color:#fff !important; padding:10px 20px;
    border-radius:8px; font-size:13px; font-weight:600; letter-spacing:.5px;
    text-decoration:none; transition:background .2s;
}
.nav-cta:hover { background:#C82121 !important; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #F8F9FA 0%, #f0fdf4 100%);
    padding: 140px 48px 60px; text-align: center; position: relative; overflow: hidden;
}
.hero::before {
    content:''; position:absolute; top:-80px; right:-80px;
    width:400px; height:400px; border-radius:50%;
    background:rgba(6, 44, 27, 0.08);
}
.hero h1 { font-size:clamp(36px,5vw,58px); font-weight:800; line-height:1.2; color:#1A1A2E; margin:0 0 8px; }
.hero h1 .red { color:#C82121; }
.hero p { font-size:15px; color:#666; margin:16px auto 40px; max-width:500px; line-height:1.7; }


/* ── Results layout ── */
.results-wrapper { padding:60px 80px; background:#F8F9FA; max-width: 1300px; margin: 0 auto;}
.section-label {
    font-size:11px; font-weight:700; letter-spacing:1.5px; color:#C82121;
    text-transform:uppercase; margin-bottom:4px;
}
.section-title { font-size:26px; font-weight:800; color:#1A1A2E; margin:0 0 24px; }

/* Image card */
.img-card {
    background:#FFFFFF; border-radius:16px; overflow:hidden;
    box-shadow:0 4px 24px rgba(0,0,0,.07);
}
.img-card-footer {
    padding:14px 20px; display:flex; justify-content:space-between; align-items:center;
    background:#FFFFFF;
}
.img-card-footer .meta { font-size:12px; color:#888; }
.img-card-footer .meta strong { display:block; color:#1A1A2E; font-size:14px; font-weight:600; }
.analyzing-badge {
    background:#fff3cd; color:#856404; padding:4px 10px;
    border-radius:20px; font-size:11px; font-weight:600;
}

/* Prediction card */
.pred-card {
    background:#DCEAE4; border-radius:16px; padding:24px;
    box-shadow:0 4px 12px rgba(0,0,0,0.05); height:100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.pred-card-header {
    display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;
}
.pred-card .section-label { margin-bottom: 0; }
.pred-card .disease-name {
    font-size:32px; font-weight:800; color:#1A1A2E;
    margin:0 0 16px; line-height:1.2;
    word-wrap:break-word; overflow-wrap:break-word; max-width: 100%;
}
.pred-card .description { font-size:14px; color:#666; line-height:1.75; margin-bottom:20px; }
.severity-badge {
    display:inline-block; padding:4px 12px; border-radius:20px;
    font-size:11px; font-weight:700; letter-spacing:.5px; white-space:nowrap;
}
.sev-high { background:#ffeaea; color:#C82121; }
.sev-medium { background:#fff3e0; color:#e67e22; }
.sev-low { background:#e8f5e9; color:#2e7d32; }
.sev-none { background:#e8f5e9; color:#2e7d32; }

/* Confidence bar */
.conf-label { font-size:12px; font-weight:600; color:#555; margin-bottom:4px; }
.conf-bar-bg { background:#eee; border-radius:4px; height:8px; overflow:hidden; }
.conf-bar-fill { height:100%; background:#C82121; }
.conf-pct { text-align:right; font-size:14px; font-weight:700; color:#1A1A2E; margin-top:4px; }

/* Stats row */
.stats-row { display:flex; gap:16px; margin-top:20px; }
.stat-box { flex:1; background:#ffffff; border-radius:12px; padding:14px 16px; }
.stat-box .stat-label { font-size:11px; font-weight:700; letter-spacing:1px; color:#888; text-transform:uppercase; margin-bottom:4px; }
.stat-box .stat-value { font-size:18px; font-weight:700; color:#1A1A2E; }
.stat-box .stat-value.danger { color:#C82121; }
.stat-box .stat-value.warning { color:#e67e22; }
.stat-box .stat-value.ok { color:#2e7d32; }

/* Explanation box */
.explain-box {
    background:#FFFFFF; border-radius:14px; padding:20px;
    border-left:4px solid #C82121; margin-top:20px;
    box-shadow:0 2px 12px rgba(0,0,0,.05);
}
.explain-box p { font-size:13px; color:#444; line-height:1.75; margin:0; }

/* ── Recommendations ── */
.rec-section { padding:40px 48px; background:#F8F9FA; }
.rec-cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:20px; margin-top:24px; }
.rec-card {
    background:#FFFFFF; border-radius:16px; overflow:hidden;
    box-shadow:0 2px 16px rgba(0,0,0,.06); transition:transform .2s, box-shadow .2s;
    display: flex;
    flex-direction: column;
}
.rec-card:hover { transform:translateY(-4px); box-shadow:0 8px 32px rgba(0,0,0,.12); }
.rec-card-img { width:100%; height:160px; object-fit:cover; background:#e8f5e9; display:flex; align-items:center; justify-content:center; }
.rec-card-img span { font-size:48px; }
.rec-card-body { padding:16px; display: flex; flex-direction: column; flex-grow: 1; }
.rec-card-num {
    width:24px; height:24px; border-radius:6px; background:#062C1B;
    color:#fff !important; font-size:11px; font-weight:700; display:inline-flex;
    align-items:center; justify-content:center; margin-bottom:12px;
}
.rec-card-content {
    display: flex;
    flex-direction: column;
    gap: 8px;
}
.rec-card-body h4 { font-size:15px; font-weight:700; color:#1A1A2E; margin:0 !important; line-height: 1.4; }
.rec-card-body p { font-size:12px; color:#777; line-height:1.6; margin:0 !important; }

/* ── Chatbot ── */
.chat-title { font-size:22px; font-weight:800; color:#1A1A2E; margin:0 0 8px; }
.chat-warning {
    background:#fff8e1; border:1px solid #ffe082; border-radius:10px;
    padding:12px 16px; font-size:12px; color:#795548; margin-bottom:20px;
}
.chat-container-inner {
    padding: 0 80px 60px !important; 
    max-width: 1200px;
    margin: 0 auto !important;
}

/* ── Footer ── */
.cabai-footer {
    background:#062C1B; color:#aaa; padding:24px 48px;
    font-size:12px; text-align:center;
}

/* ── Override Streamlit native widgets ── */
[data-testid="stFileUploader"] { background:transparent !important; border:none !important; }
[data-testid="stFileUploader"] > div { border:none !important; }

/* Override Streamlit columns background */
[data-testid="stColumn"] {
    background: transparent !important;
}

/* ── Equal-height columns ── */
/* The horizontal block that wraps st.columns */
[data-testid="stHorizontalBlock"] {
    background-color: transparent !important;
    display: flex !important;
    align-items: flex-start !important;
}
/* Each column container must stretch to fill */
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
    display: flex !important;
    flex-direction: column !important;
    height: auto !important;
}
/* The vertical block inside each column must also stretch */
[data-testid="stColumn"] > [data-testid="stVerticalBlock"] {
    flex: 1 !important;
    display: flex !important;
    flex-direction: column !important;
}
/* Make each markdown element inside the column stretch equally */
[data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] {
    flex: 1 !important;
    display: flex !important;
    flex-direction: column !important;
}
[data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] > .stMarkdown {
    flex: 1 !important;
    display: flex !important;
    flex-direction: column !important;
}
[data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] > .stMarkdown > div {
    flex: 1 !important;
}

/* Vertical blocks general */
[data-testid="stVerticalBlock"] {
    background-color: transparent !important;
}

/* Force Streamlit spinner/info/warning to use light bg */
.stAlert { background-color: #FFFFFF !important; }

/* File uploader styling */
[data-testid="stFileUploaderDropzone"] {
    background: #FFFFFF !important;
    border: 2px dashed #ccc !important;
    border-radius: 12px !important;
}

.chat-container-inner [data-testid="stVerticalBlock"] {
    padding: 0 !important;
}

[data-testid="stChatInput"], 
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] div[class*="st-"] {
    background-color: #0d5f3a !important;
    border-radius: 15px !important;
    border: 1px solid #0d5f3a !important;
    box-shadow: none !important;
}

[data-testid="stChatInput"]:focus-within,
[data-testid="stChatInput"] textarea:focus,
[data-testid="stChatInput"] div:focus-within,
[data-testid="stChatInput"] [role="textbox"]:focus {
    border-color: #0d5f3a !important;
    box-shadow: none !important;
    outline: none !important;
}

[data-testid="stChatInput"] textarea {
    background-color: transparent !important; 
    color: white !important;
    caret-color: white !important; 
    -webkit-text-fill-color: white !important;
    border: none !important;
    padding: 10px !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: rgba(255, 255, 255, 0.6) !important;
    -webkit-text-fill-color: rgba(255, 255, 255, 0.6) !important;
}

[data-testid="stChatInput"] button {
    border: none !important;
    background: transparent !important;
}

[data-testid="stChatInput"] button svg {
    fill: white !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# Device Model 
@st.cache_resource
def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    elif torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

@st.cache_resource(show_spinner="⏳ Memuat model AI...")
def load_model():
    ckpt = CHECKPOINTS_DIR / 'efficientnet_b0_demo.pt'
    model = create_model(model_name="efficientnet_b0", num_classes=5, pretrained=not ckpt.exists())
    if ckpt.exists():
        ck = torch.load(ckpt, map_location=get_device())
        model.load_state_dict(ck['model_state_dict'])
        trained = True
    else:
        trained = False
    model.to(get_device()).eval()
    return model, trained

device = get_device()
model, is_trained = load_model()
transform = get_transforms(train=False)

# Gemini key 
try:
    gemini_key = st.secrets.get("GEMINI_API_KEY", "")
except Exception:
    gemini_key = ""

gemini_key = gemini_key or os.environ.get("GEMINI_API_KEY", "")

if not gemini_key and os.path.exists(".env"):
    with open(".env", "r") as f:
        for line in f:
            if "GEMINI_API_KEY" in line:
                gemini_key = line.split("=", 1)[1].strip()
                break

# Routing using query params
page = st.query_params.get("page", "beranda")

# Navbar 
nav_html = f"""
<nav class="cabai-nav">
  <div class="logo"><img src="data:image/png;base64,{logo_base64}" style="width:120px;"></div>
  <div class="links">
    <a href="?page=beranda" target="_self" class="{'active' if page == 'beranda' else ''}">Beranda</a>
    <a href="?page=tentang" target="_self" class="{'active' if page == 'tentang' else ''}">Tentang Kami</a>
    <a href="?page=dokumentasi" target="_self" class="{'active' if page == 'dokumentasi' else ''}">Dokumentasi</a>
  </div>
  <a class="nav-cta" href="?page=beranda#upload" target="_self">MULAI SCREENING</a>
</nav>
"""
st.markdown(nav_html, unsafe_allow_html=True)

if page == "tentang":
    render_html("""
    <style>
    .tentang-container { max-width: 1000px; margin: 0 auto; padding: 40px 20px 80px; }
    .misi-label { font-size: 12px; font-weight: 700; color: #C82121 !important; text-transform: uppercase; letter-spacing: 2px; margin-top: 24px !important;}
    .tentang-title { font-size: clamp(32px, 4vw, 42px); font-weight: 800; color: #062C1B; line-height: 1.2; margin: 0 0 24px; max-width: 800px; }
    .tentang-subtitle { font-size: 16px; color: #555; line-height: 1.8; max-width: 800px; margin-bottom: 40px; }
    .tentang-hero-img { width: 100% !important; height: 200px; object-fit: cover !important; object-position: center 30%; border-radius: 24px; margin-bottom: 80px; box-shadow: 0 8px 32px rgba(0,0,0,0.1); }
    .section-header { font-size: 24px; font-weight: 800; color: #062C1B; margin-bottom: 24px; position: relative; padding-bottom: 12px; display: inline-block;}
    .section-header::after { content: ''; position: absolute; left: 0; bottom: 0; width: 60%; height: 4px; background: #C82121; border-radius: 2px; }
    .kisah-text { font-size: 16px; color: #222; line-height: 1.8; font-weight: 500; }
    .value-card { padding: 32px; border-radius: 20px; height: 100%; display: flex; flex-direction: column; justify-content: center; }
    .value-card.light { background: #e9ecef; color: #1A1A2E; }
    .value-card.dark { background: #062C1B; color: #ffffff !important; }
    .value-card.dark h3 {color: #80E2A7 !important; font-size: 18px; font-weight: 800; margin: 16px 0 8px;}
    .value-card.dark p {color: rgba(255,255,255,0.9) !important;}
    .value-card.red { background: #C82121; color: #ffffff !important; }
    .value-card h3 { font-size: 16px; font-weight: 800; margin: 16px 0 8px; }
    .value-card p { font-size: 13px; opacity: 0.9; margin: 0; line-height: 1.6; }
    .value-icon { font-size: 32px; margin-bottom: 8px; }
    .tech-card { background: #ffffff; border-radius: 20px; overflow: hidden; box-shadow: 0 4px 24px rgba(0,0,0,0.06); height: 100%; display:flex; flex-direction:column; }
    .tech-card-content { padding: 32px; flex-grow: 1; }
    .tech-card-content h3 { font-size: 22px; font-weight: 800; color: #062C1B; margin: 0 0 12px; }
    .tech-card-content p { font-size: 14px; color: #444; line-height: 1.7; margin: 0; }
    .tech-card-dark { background: #062C1B; color: white; border-radius: 20px; padding: 32px; height: 100%; display: flex; flex-direction: column; justify-content: center; }
    .tech-card-dark h3 { font-size: 24px; font-weight: 800; color: #80E2A7 !important; margin: 0 0 12px; }
    .tech-card-dark p { font-size: 14px; color: rgba(255,255,255,0.9) !important; line-height: 1.7; margin:0;}
    .css-grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 40px; align-items: start; margin-bottom: 80px; }
    .css-grid-inner { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .css-grid-tech { display: grid; grid-template-columns: 1.5fr 1fr; gap: 24px; }
    @media (max-width: 768px) {
        .css-grid-2, .css-grid-tech { grid-template-columns: 1fr; }
        .tentang-hero-img { height: 250px; }
    }
    </style>

    <div class="tentang-container">
        <div class="misi-label">MISI KAMI</div>
        <h1 class="tentang-title">Memberdayakan Petani Cabai Melalui Kecerdasan Buatan</h1>
        <p class="tentang-subtitle">Kami hadir untuk menjembatani kearifan lokal pertanian dengan inovasi teknologi terdepan, memastikan setiap bibit cabai tumbuh sehat dan produktif melalui deteksi dini berbasis AI.</p>

        <img src="https://images.unsplash.com/photo-1546860255-95536c19724e?q=80&w=708&auto=format&fit=crop" class="tentang-hero-img" alt="Pertanian Cabai">

        <div class="css-grid-2">
            <div>
                <h2 class="section-header">Kisah Perjalanan Kami</h2>
                <p class="kisah-text">CabAI lahir dari kegelisahan melihat tantangan nyata yang dihadapi petani cabai di tingkat lokal. Penyakit tanaman seringkali terlambat dideteksi, mengakibatkan gagal panen yang merugikan ekonomi rumah tangga petani.</p>
                <p class="kisah-text">Dimulai sebagai proyek riset sederhana di tahun 2023, tim kami mengumpulkan ribuan dataset citra daun cabai langsung dari lahan pertanian. Melalui kolaborasi antara agronomis dan pengembang AI, kami membangun solusi yang praktis namun presisi tinggi.</p>
            </div>
            <div class="css-grid-inner">
                <div class="value-card light" style="text-align: center;">
                    <div class="value-icon"><i class="bi bi-geo-alt-fill" style="color: #C82121;"></i></div>
                    <h3>Akar Lokal</h3>
                    <p>Dibangun khusus untuk ekosistem pertanian tropis.</p>
                </div>
                <div class="value-card dark" style="text-align: center;">
                    <div class="value-icon"><i class="bi bi-people-fill" style="color: #B8FFCB;"></i></div>
                    <div style="color: #80E2A7 !important; font-size: 18px; font-weight: 800; margin: 16px 0 8px;">Kolaborasi</div>
                    <div style="color: rgba(255,255,255,0.9) !important; font-size: 13px; margin: 0; line-height: 1.6;">Menghubungkan ahli botani dengan data scientist.</div>
                </div>
                <div class="value-card red" style="grid-column: 1 / -1; display: flex; flex-direction: row; align-items: center; gap: 20px; padding: 24px 32px;">
                    <div class="value-icon" style="margin:0; font-size: 32px;"><i class="bi bi-rocket-fill" style="color: #fff;"></i></div>
                    <div>
                        <div style="margin: 0 0 4px; font-size: 15px; color: #fff !important; font-weight: 800;">Visi 2030</div>
                        <div style="font-size: 13px; font-weight: 500; color: rgba(255,255,255,0.9) !important;">Mewujudkan kedaulatan pangan nasional melalui digitalisasi lahan pertanian Indonesia.</div>
                    </div>
                </div>
            </div>
        </div>

        <div style="text-align: center; margin-bottom: 40px;">
            <h2 class="section-header" style="display: inline-block;">Teknologi Di Balik Layar</h2>
        </div>

        <div class="css-grid-tech">
            <div class="tech-card" style="flex-direction: row; align-items: stretch;">
                <div class="tech-card-content" style="flex: 1; display:flex; flex-direction: column; justify-content: center;">
                    <h3>Akurasi 96.4%</h3>
                    <p>Model computer vision kami telah dilatih dengan lebih dari 50.000 sampel data untuk mengenali berbagai jenis hama dan penyakit daun cabai secara instan.</p>
                    <div><span style="display: inline-block; margin-top: 20px; padding: 6px 16px; background: #062C1B; color: #fff; border-radius: 99px; font-size: 11px; font-weight: 700;"><i class="bi bi-patch-check-fill" style="margin-right: 5px;"></i> Diagnosa Real-Time</span></div>
                </div>
                <div style="flex: 0.8; background: #111;">
                    <img src="https://images.unsplash.com/photo-1555949963-ff9fe0c870eb?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&q=80" style="width: 100%; height: 100%; object-fit: cover; opacity: 0.8;" alt="AI Tech">
                </div>
            </div>
            <div class="tech-card-dark">
                <div style="font-size: 32px; margin-bottom: 16px;"><i class="bi bi-eye-fill" style="color: #B8FFCB;"></i></div>
                <div style="color: #fff !important; font-size: 24px; font-weight: 800; margin: 0 0 12px;">Computer Vision</div>
                <div style="color: rgba(255,255,255,0.9) !important; font-size: 14px; line-height: 1.7; margin: 0;">Memproses citra digital dengan algoritma deep learning untuk identifikasi pola penyakit terkecil yang seringkali luput dari mata telanjang.</div>
            </div>
        </div>
    </div>
    """)
    st.stop()


if page == "dokumentasi":
    import os
    _gradcam_path = os.path.join("static", "grad_cam_contoh.png")
    if os.path.exists(_gradcam_path):
        gradcam_b64 = get_base64_of_bin_file(_gradcam_path)
        gradcam_src = f"data:image/png;base64,{gradcam_b64}"
    else:
        gradcam_src = "https://images.unsplash.com/photo-1590682680695-43b964a3ae17?w=400&auto=format&fit=crop"
    render_html(f"""
    <style>
    .dok-container {{ max-width: 1000px; margin: 0 auto; padding: 40px 20px 80px; }}
    .dok-label {{ font-size: 12px; font-weight: 700; color: #C82121 !important; text-transform: uppercase; letter-spacing: 2px; margin-top: 24px; }}
    .dok-title {{ font-size: clamp(28px, 3.5vw, 38px); font-weight: 800; color: #062C1B; line-height: 1.2; margin: 0 0 12px; }}
    .dok-subtitle {{ font-size: 15px; color: #555; line-height: 1.7; margin-bottom: 40px; max-width: 700px; }}
    .dok-section-title {{ font-size: 22px; font-weight: 800; color: #062C1B; margin: 60px 0 8px; position: relative; padding-bottom: 12px; display: inline-block; }}
    .dok-section-title::after {{ content: ''; position: absolute; left: 0; bottom: 0; width: 60%; height: 4px; background: #C82121; border-radius: 2px; }}
    .dok-section-sub {{ font-size: 14px; color: #666; margin: 8px 0 28px; }}
    .dok-step-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 60px; }}
    .dok-step-card {{ background: #fff; border: 1px solid #e8e8e8; border-radius: 16px; padding: 28px; box-shadow: 0 2px 12px rgba(0,0,0,0.05); }}
    .dok-step-num {{ width: 32px; height: 32px; background: #062C1B; color: #fff !important; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 14px; font-weight: 800; margin-bottom: 16px; }}
    .dok-step-title {{ font-size: 16px; font-weight: 800; color: #1A1A2E; margin: 0 0 10px; }}
    .dok-step-desc {{ font-size: 13px; color: #666; line-height: 1.7; margin: 0 0 20px; }}
    .dok-step-img {{ width: 100% !important; border-radius: 12px; object-fit: cover !important; height: 180px !important; }}
    .dok-step-upload-box {{ border: 2px dashed #ccc; border-radius: 12px; padding: 36px 20px; text-align: center; color: #aaa; font-size: 13px; margin-top: 16px; }}
    .dok-result-container {{ background: #f5f7f5; border-radius: 20px; padding: 32px; margin-bottom: 60px; }}
    .dok-result-grid {{ display: grid; grid-template-columns: 1fr 1.2fr; gap: 24px; align-items: start; }}
    .dok-preview-box {{ background: #fff; border-radius: 14px; padding: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); margin-bottom: 16px; }}
    .dok-preview-label {{ font-size: 11px; font-weight: 700; color: #555; text-transform: uppercase; letter-spacing: 1px; margin-top: 10px; }}
    .dok-preview-meta {{ font-size: 11px; color: #aaa; }}
    .dok-pred-card {{ background: #fff; border-radius: 14px; padding: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
    .dok-badge-high {{ background: #C82121; color: #fff; font-size: 10px; font-weight: 800; padding: 3px 10px; border-radius: 99px; letter-spacing: 0.5px; }}
    .dok-pred-disease {{ font-size: 22px; font-weight: 800; color: #062C1B; margin: 10px 0 6px; }}
    .dok-pred-desc {{ font-size: 12px; color: #777; line-height: 1.6; margin-bottom: 16px; }}
    .dok-score-bar-bg {{ background: #eee; border-radius: 99px; height: 6px; margin-bottom: 4px; }}
    .dok-score-bar {{ background: #C82121; height: 6px; border-radius: 99px; width: 82.5%; }}
    .dok-stat-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 16px; }}
    .dok-stat-box {{ background: #f5f7f5; border-radius: 10px; padding: 12px 16px; }}
    .dok-stat-label {{ font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; color: #999; font-weight: 700; }}
    .dok-stat-val {{ font-size: 20px; font-weight: 800; color: #062C1B; margin-top: 2px; }}
    .dok-stat-val.red {{ color: #C82121; }}
    .dok-ai-box {{ background: #f0f4f0; border-radius: 10px; padding: 14px 16px; font-size: 12px; color: #444; line-height: 1.7; }}
    .dok-ai-label {{ font-size: 10px; font-weight: 700; color: #062C1B; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }}
    .dok-chat-section {{ background: #f5f7f5; border-radius: 20px; padding: 32px; margin-bottom: 60px; }}
    .dok-chat-card {{ background: #fff; border-radius: 14px; padding: 20px 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
    .dok-chat-title {{ font-size: 15px; font-weight: 800; color: #1A1A2E; margin: 0 0 12px; }}
    .dok-chat-warning {{ background: #fff8e1; border-left: 3px solid #f59e0b; padding: 10px 14px; border-radius: 0 8px 8px 0; font-size: 12px; color: #92400e; margin-bottom: 14px; }}
    .dok-chat-input {{ background: #0d5f3a; border-radius: 10px; padding: 14px 50px 14px 16px; color: rgba(255,255,255,0.5) !important; font-size: 13px; position: relative; }}
    @media (max-width: 768px) {{ .dok-step-grid, .dok-result-grid {{ grid-template-columns: 1fr; }} }}
    </style>
    <div class="dok-container">
    <div class="dok-label">PANDUAN PENGGUNAAN</div>
    <h1 class="dok-title">Panduan Memulai</h1>
    <p class="dok-subtitle">Pelajari cara menggunakan teknologi AI kami untuk mendiagnosis kesehatan tanaman cabai Anda dalam hitungan detik melalui unggahan foto yang sederhana.</p>
    <div class="dok-step-grid">
    <div class="dok-step-card">
    <div class="dok-step-num">1</div>
    <div class="dok-step-title">Ambil Foto Daun</div>
    <div class="dok-step-desc">Pastikan daun berada di tengah bingkai dengan pencahayaan yang cukup. AI kami bekerja paling baik dengan latar belakang yang netral.</div>
    <img src="https://www.shutterstock.com/image-photo/sick-leaves-on-peach-tree-260nw-2465477217.jpg" class="dok-step-img" alt="Foto Daun">
    </div>
    <div class="dok-step-card">
    <div class="dok-step-num">2</div>
    <div class="dok-step-title">Unggah ke CabAI</div>
    <div class="dok-step-desc">Gunakan zona unggah seret-dan-lepas di dashboard utama. AI akan memproses gambar Anda secara real-time menggunakan model deep learning.</div>
    <div class="dok-step-upload-box">
    <div style="font-size: 32px; margin-bottom: 8px;"><i class="bi bi-cloud-upload"></i></div>
    <div style="font-size: 13px; font-weight: 600; color: #444;">Klik untuk Unggah</div>
    <div style="font-size: 11px; color: #aaa; margin-top: 4px;">Format JPG, PNG (Maks. 5MB)</div>
    </div>
    </div>
    </div>
    <div class="dok-section-title">Memahami Hasil Diagnosis</div>
    <p class="dok-section-sub">Penjelasan tentang tampilan hasil skrining kesehatan tanaman Anda.</p>
    <div class="dok-result-container">
    <div class="dok-result-grid">
    <div>
    <div class="dok-preview-box">
    <img src="https://www.shutterstock.com/image-photo/sick-leaves-on-peach-tree-260nw-2465477217.jpg" style="width:100%; border-radius: 8px; height: 160px; object-fit: cover;">
    <div class="dok-preview-label">Preview Pemindaian</div>
    <div class="dok-preview-meta">Resolusi: 240x196px · Format: JPG</div>
    </div>
    <div class="dok-preview-box">
    <img src="{gradcam_src}" style="width:100%; border-radius: 8px; height: 140px; object-fit: cover;">
    <div class="dok-preview-label">Peta Perhatian AI (Grad-CAM)</div>
    <div class="dok-preview-meta">Area yang digunakan model saat membuat prediksi</div>
    </div>
    </div>
    <div>
    <div class="dok-pred-card">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
    <div style="font-size:10px; font-weight:700; color:#999; text-transform:uppercase; letter-spacing:1px;">HASIL PREDIKSI</div>
    <div class="dok-badge-high">Peringatan Tinggi</div>
    </div>
    <div class="dok-pred-disease">Whitefly / Kutu Kebul</div>
    <div class="dok-pred-desc">Gejala terkait serangan whitefly atau kutu kebul pada daun cabai.</div>
    <div style="font-size:11px; color:#999; margin-bottom:4px; display:flex; justify-content:space-between;"><span>Confidence Score</span><span style="font-weight:700; color:#1A1A2E;">82.5%</span></div>
    <div class="dok-score-bar-bg"><div class="dok-score-bar"></div></div>
    <div style="height:12px;"></div>
    <div class="dok-stat-row">
    <div class="dok-stat-box">
    <div class="dok-stat-label">AREA TERDAMPAK</div>
    <div class="dok-stat-val">37.4%</div>
    </div>
    <div class="dok-stat-box">
    <div class="dok-stat-label">TINGKAT RISIKO</div>
    <div class="dok-stat-val red">Tinggi</div>
    </div>
    </div>
    <div class="dok-ai-label"><i class="bi bi-robot"></i> MENGAPA AI MENYIMPULKAN INI</div>
    <div class="dok-ai-box">Model mendeteksi "Whitefly / kutu kebul" dengan keyakinan 83%. Area perhatian utama model berada di bagian tengah dan sisi kiri. Serangan kutu kebul menyebabkan bintik-bintik kuning kecil dan daun menjadi layu, serangga kecil putih sering ditemukan di bagian bawah daun.</div>
    </div>
    </div>
    </div>
    </div>
    <div class="dok-section-title">Interaksi Lanjutan</div>
    <p class="dok-section-sub">Gunakan asisten AI untuk bertanya lebih detail tentang diagnosis dan solusi spesifik untuk lahan Anda.</p>
    <div class="dok-chat-section">
    <div class="dok-chat-card">
    <div class="dok-chat-title"><i class="bi bi-chat-dots-fill"></i> Tanya Lebih Lanjut tentang Whitefly / Kutu Kebul</div>
    <div class="dok-chat-warning"><i class="bi bi-exclamation-triangle-fill" style="color: #FF9800; font-size: 18px;"></i> Jawaban chatbot AI bersifat informatif dan dapat mengandung ketidakakuratan. Selalu konsultasikan ke ahli agronomi untuk keputusan penting.</div>
    <div class="dok-chat-input">Tanya seputar whitefly / kutu kebul pada cabai kamu…</div>
    </div>
    </div>
    </div>
    """)
    st.stop()

# Hero
st.markdown("""
<section class="hero">
  <h1>Identifikasi Penyakit<br><span class="red">Tanaman Cabai</span> Sekejap.</h1>
  <p>Gunakan teknologi visi komputer tercanggih untuk mendeteksi hama
  dan penyakit pada daun cabai Anda dengan akurasi hingga 96.4%.</p>
</section>
""", unsafe_allow_html=True)


if 'show_upload' not in st.session_state:
    st.session_state.show_upload = False

uploaded_file = None

#  Upload Section 
st.markdown("""
    <style>

    div.stButton > button {
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    div.stButton > button[kind="primary"] {
        background-color: #09452b !important;
        border: none !important;
    }

    div.stButton > button[kind="primary"] p,
    div.stButton > button[kind="primary"] span,
    div.stButton > button[kind="primary"] div {
        color: white !important;
    }
    
    div.stButton > button[kind="secondary"] {
        background-color: white !important;
        color: #09452b !important;
        border: 2px solid #09452b !important;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #062C1B !important; 
        color: white !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    div.stButton > button[kind="secondary"]:hover {
        background-color: #f0fdf4 !important; 
        transform: translateY(-2px);
    }
    </style>
""", unsafe_allow_html=True)

st.write("")
st.write("")
col_btn = st.columns([1, 2, 1])[1] 

with col_btn:
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Unggah Foto", type="primary", icon=":material/upload:", use_container_width=True):
            st.session_state.show_upload = True
    with c2:
        if st.button("Pelajari Demo", type="secondary", icon=":material/lightbulb:", use_container_width=True):
            st.query_params["page"] = "dokumentasi"
            st.rerun()

if st.session_state.show_upload:
    with st.container():
        _, content_col, _ = st.columns([1, 6, 1])
        with content_col:
            st.markdown('<div class="upload-card">', unsafe_allow_html=True)
            st.markdown('<p style="font-weight:700; color:#1A1A2E; margin-bottom:10px;"><strong>Upload foto daun cabai</strong></p>', unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Upload", 
                type=["jpg", "jpeg", "png", "webp"], 
                label_visibility="collapsed"
            )
            
            if st.button("Tutup", key="btn_tutup"):
                st.session_state.show_upload = False
                st.rerun()

# Inference & Results
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    w, h = image.size
    fmt = uploaded_file.name.split(".")[-1].upper()

    with st.spinner("🔍 Menganalisis gambar..."):
        inp = transform(image).unsqueeze(0).to(device)
        rgb = np.float32(image.resize((224,224))) / 255.0
        with torch.no_grad():
            probs = torch.nn.functional.softmax(model(inp)[0], dim=0)
        conf, idx = torch.max(probs, 0)
        conf, idx = conf.item(), idx.item()
        pred = CLASS_NAMES[idx]
        dname = DISPLAY_NAMES.get(pred, pred)

        cam_img, grayscale_cam = generate_gradcam(model, inp, rgb, target_category=idx)
        cam_pil = Image.fromarray(cam_img)

        explanation = (
            generate_explanation_gemini(pred, conf, grayscale_cam, gemini_key)
            if gemini_key else _fallback_explanation(pred, conf, grayscale_cam)
        )
        rec = get_recommendation(pred)

    # Severity logic
    if pred == "healthy":
        sev_cls, sev_label, risk_cls = "sev-none", "Sehat", "ok"
        risk_label = "Rendah"
    elif conf >= 0.80:
        sev_cls, sev_label, risk_cls = "sev-high", "Peringatan Tinggi", "danger"
        risk_label = "Tinggi"
    elif conf >= 0.55:
        sev_cls, sev_label, risk_cls = "sev-medium", "Peringatan Sedang", "warning"
        risk_label = "Medium"
    else:
        sev_cls, sev_label, risk_cls = "sev-low", "Peringatan Rendah", "ok"
        risk_label = "Rendah"

    area_pct = float(np.mean(grayscale_cam > 0.5) * 100)
    conf_pct = conf * 100

    # --- Encode images for HTML display ---
    def pil_to_b64(img):
        import io
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    img_b64 = pil_to_b64(image)
    cam_b64 = pil_to_b64(cam_pil)

    st.markdown('<div class="results-wrapper">', unsafe_allow_html=True)

    _, main_col, _ = st.columns([0.5, 10, 0.5])
    
    with main_col:
        col1, col2 = st.columns([1.1, 1], gap="large")

        with col1:
            st.markdown(f"""
            <div class="img-card">
            <img src="data:image/png;base64,{img_b64}" style="width:100%;display:block;max-height:360px;object-fit:cover;" />
            <div class="img-card-footer">
            <div class="meta">
            <strong>Preview Pemindaian</strong>
            Resolusi: {w}×{h}px &nbsp;·&nbsp; Format: {fmt}
            </div>
            </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="margin-top:16px;" class="img-card">
            <img src="data:image/png;base64,{cam_b64}" style="width:100%;display:block;max-height:280px;object-fit:cover;" />
            <div class="img-card-footer">
            <div class="meta"><strong>Peta Perhatian AI (Grad-CAM)</strong>
            Area yang difokuskan model saat membuat prediksi</div>
            </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="pred-card">
            <div class="pred-card-header">
            <div class="section-label">HASIL PREDIKSI</div>
            <span class="severity-badge {sev_cls}">{sev_label}</span>
            </div>
            <h2 class="disease-name">{dname.title()}</h2>
            <p class="description">{rec.get('deskripsi','')}</p>
            <div class="conf-label">Confidence Score</div>
            <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{conf_pct:.1f}%"></div></div>
            <div class="conf-pct">{conf_pct:.1f}%</div>
            <div class="stats-row">
            <div class="stat-box"><div class="stat-label">Area Terdampak</div><div class="stat-value">{area_pct:.1f}%</div></div>
            <div class="stat-box"><div class="stat-label">Tingkat Risiko</div><div class="stat-value {risk_cls}">{risk_label}</div></div>
            </div>
            <div class="explain-box">
            <div class="section-label"><i class="bi bi-robot"></i> MENGAPA AI MENYIMPULKAN INI</div>
            <p>{explanation.replace(chr(10), '<br>')}</p>
            </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Recommendations
    steps = rec.get("tindakan_awal", []) + rec.get("pencegahan", [])
    icons = ["🧴","🌿","🌱","💧","🔍","🏥"]
    cards_html = ""

    for i, step in enumerate(steps[:6]):
        if "." in step:
            title, desc = step.split(".", 1)
            title = title.strip()
            desc = desc.strip()
        else:
            title = step
            desc = step

        cards_html += f"""
        <div class="rec-card">
            <div class="rec-card-img"><span>{icons[i % len(icons)]}</span></div>
            <div class="rec-card-body">
                <div class="rec-card-num">0{i+1}</div>
                <div class="rec-card-content">
                    <h4>{title}</h4>
                    <p>{desc}</p>
                </div>
            </div>
        </div>"""

    st.markdown(f"""
    <div class="rec-section">
      <div class="section-label">SOLUSI &amp; REKOMENDASI</div>
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <h2 class="section-title" style="margin-bottom:0;">Langkah Penanganan</h2>
        <a href="#" style="font-size:13px;color:#E63946;font-weight:600;text-decoration:none;">Selengkapnya →</a>
      </div>
      <div class="rec-cards">{cards_html}</div>
    </div>
    """, unsafe_allow_html=True)

    # Chatbot 
    st.markdown('<div class="results-wrapper" style="background: transparent; padding-top: 0; margin-top: -60px;">', unsafe_allow_html=True)
    _, chat_col, _ = st.columns([0.5, 10, 0.5]) 

    with chat_col:
        st.markdown(f"""
        <h3 class="chat-title" style="display: flex; align-items: center;">
            <i class="bi bi-chat-dots-fill" style   ="color: #062C1B;"></i>
            Tanya Lebih Lanjut tentang {dname.title()}
        </h3>
        <div class="chat-warning" style="display: flex; align-items: center; gap: 12px; background: #fff8e1; border: 1px solid #ffe082; padding: 12px 16px; border-radius: 10px; font-size: 13px; color: #795548; margin-bottom: 20px;">
            <i class="bi bi-exclamation-triangle-fill" style="color: #FF9800; font-size: 18px;"></i>
            <div>
                Jawaban chatbot dihasilkan AI (Gemini) dan dapat mengandung ketidakakuratan.
                Selalu konsultasikan ke ahli agronomi untuk keputusan penting.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if not gemini_key:
            st.info("Chatbot tidak tersedia — API key Gemini belum dikonfigurasi.")
        else:
            q = st.chat_input(f"Tanya seputar {dname} pada cabai kamu...")
            if q:
                with st.chat_message("user"): st.markdown(q)
                with st.chat_message("assistant"):
                    with st.spinner("Memproses..."):
                        ans = answer_followup_question(q, pred, conf, rec, gemini_key)
                    st.markdown(ans)

else:
    st.markdown("""
    <div style="text-align:center;padding:40px 48px 60px;">
      <p style="color:#999;font-size:14px;margin-bottom:0;">
        <i class="bi bi-cloud-upload"></i> Unggah foto daun cabai di atas untuk memulai analisis
      </p>
    </div>
    """, unsafe_allow_html=True)

# Footer 
st.markdown("""
<div class="cabai-footer" style="color:white">
  © 2025 CabAI · Tugas Besar II4012 – Inteligensi Artifisial untuk Bisnis · ITB<br>
  <span style="font-size:11px; color:#FFFFFF;">Peringatan: Sistem ini adalah alat bantu identifikasi awal, bukan pengganti ahli agronomi.</span>
</div>
""", unsafe_allow_html=True)

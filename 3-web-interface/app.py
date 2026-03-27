"""
Streamlit Frontend Premium - License Plate Detection
Features:
    - Button-based sidebar navigation
    - Glassmorphism & Neon Design System
    - Unified PySpark Inference Pipeline
    - Run-based history grouping
"""

import streamlit as st
import requests
import pandas as pd
import base64
import datetime
import csv
import random
import uuid
import ast
from io import BytesIO
from pathlib import Path
from PIL import Image
from typing import List, Dict, Optional
from functions_stremlit import migrate_old_history_csv


# ============================================================================
# CONFIG & PAGE SETUP
# ============================================================================
API_URL = "http://localhost:8000"
HISTORY_CSV = Path(__file__).parent / "history.csv"
#TEST_IMAGES_DIR = Path("/Users/zazak/Documents/Studies/M1 - ESGI/Trimestre 2/Projet Spark Core/license-plate-detection-dataset-10125-images/test/images")
TEST_IMAGES_DIR = Path(__file__).parent.parent / "license-plate-detection-dataset-10125-images" / "test" / "images"

st.set_page_config(
    page_title="Vision Plates | AI Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# MODERN DATA HELPERS
# ============================================================================
@st.cache_data(ttl=5)
def get_api_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200: return r.json()
    except: pass
    return {"online": False, "model_loaded": False}

@st.cache_data(ttl=30)
def get_api_stats():
    try:
        r = requests.get(f"{API_URL}/api/stats", timeout=3)
        if r.status_code == 200: return r.json()
    except: pass
    return None

def load_history_csv(limit: int = 50) -> List[Dict]:
    if not HISTORY_CSV.exists(): return []
    try:
        migrate_old_history_csv()
        entries = []
        with open(HISTORY_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                entries.append({
                    "datetime": row.get('timestamp', ''),
                    "filename": row.get('image_name', ''),
                    "nb_plates": int(row.get('nb_plates', 0)),
                    "status": row.get('status', 'unknown'),
                    "detections": row.get('detections', '[]'),
                    "original_image_path": row.get('original_image_path', ''),
                    "annotated_image_path": row.get('annotated_image_path', '')
                })
        return list(reversed(entries))[:limit]
    except: return []

def get_history_stats():
    history = load_history_csv(limit=1000)
    return {
        "total_predictions": len(history),
        "total_plates": sum(h.get('nb_plates', 0) for h in history)
    }

def load_image_from_path(relative_path: str) -> Optional[Image.Image]:
    if not relative_path: return None
    try:
        abs_path = HISTORY_CSV.parent / relative_path
        if abs_path.exists(): return Image.open(abs_path).convert('RGB')
    except: pass
    return None

def predict_batch_images(uploaded_files: List) -> Optional[Dict]:
    try:
        files = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]
        r = requests.post(f"{API_URL}/predict", files=files, timeout=300)
        if r.status_code == 200: return r.json()
        st.error(f"API Error: {r.text}")
    except Exception as e: st.error(f"Request failed: {e}")
    return None

def get_dataset_samples(n=24):
    if not TEST_IMAGES_DIR.exists(): return []
    all_images = list(TEST_IMAGES_DIR.glob("*.jpg")) + list(TEST_IMAGES_DIR.glob("*.png"))
    return random.sample(all_images, min(n, len(all_images)))

# ============================================================================
# PREMIUM GLASSMORPHISM CSS
# ============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;600;800&family=Inter:wght@400;500;700&display=swap');

/* Main Theme Styling */
html, body, [class*="css"] { 
    font-family: 'Inter', sans-serif;
}
h1, h2, h3, .main-title {
    font-family: 'Outfit', sans-serif;
}

.stApp {
    background: radial-gradient(circle at top left, rgba(99,102,241,0.1) 0%, transparent 40%),
                radial-gradient(circle at bottom right, rgba(34,211,238,0.1) 0%, transparent 40%),
                linear-gradient(135deg, #0f172a 0%, #020617 100%);
    color: #f1f5f9;
}

/* Glassmorphism Cards */
.glass-card {
    background: rgba(30, 41, 59, 0.4);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
}

.main-title {
    font-size: 3.5rem; 
    font-weight: 800; 
    letter-spacing: -0.05em;
    margin-bottom: 0.2rem;
    background: linear-gradient(135deg, #818cf8 0%, #22d3ee 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Spark & Status Badges */
.spark-badge {
    background: rgba(245, 158, 11, 0.15);
    color: #fbbf24;
    padding: 4px 12px;
    border-radius: 100px;
    font-size: 0.75rem;
    font-weight: 600;
    border: 1px solid rgba(245, 158, 11, 0.3);
}

/* Custom Selection Grid */
.selection-img-container img {
    height: 180px !important;
    object-fit: cover !important;
    border-radius: 16px !important;
    transition: transform 0.3s ease;
    border: 2px solid rgba(255,255,255,0.05);
}
.selection-img-container img:hover {
    transform: scale(1.03);
    border-color: #818cf8;
}

/* Sidebar Customization */
.stSidebar {
    background-color: rgba(2, 6, 23, 0.8) !important;
    backdrop-filter: blur(10px);
}
[data-testid="stSidebarNav"] {display: none;} /* Hide default radio if any left */

/* Expander Styling */
.streamlit-expanderHeader {
    background-color: rgba(255,255,255,0.03) !important;
    border-radius: 12px !important;
    border: none !important;
}

/* Metric Polish */
[data-testid="stMetricValue"] {
    font-weight: 700 !important;
    font-family: 'Outfit', sans-serif;
    color: #22d3ee !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# NAVIGATION ENGINE (BUTTON BASED)
# ============================================================================
if "page" not in st.session_state:
    st.session_state.page = "📥 Upload Analysis"

def set_page(p):
    st.session_state.page = p

# Sidebar Content
with st.sidebar:
    st.markdown("<h2 style='color:#818cf8;'>⚡ Plate Analytics</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Navigation Buttons
    nav_options = ["📥 Upload Analysis", "📂 Sample Dataset", "🕘 Prediction History"]
    for opt in nav_options:
        is_active = st.session_state.page == opt
        if st.button(
            opt, 
            use_container_width=True, 
            type="primary" if is_active else "secondary",
            key=f"nav_{opt}"
        ):
            st.session_state.page = opt
            st.rerun()
            
    st.markdown("---")
    
    # Mini stats in sidebar
    stats = get_api_stats()
    api_state = get_api_health()
    
    st.markdown("### 📊 Dataset Overview")
    c1, c2 = st.columns(2)
    c1.metric("Images", stats.get("total_images", 0) if stats else 0)
    c2.metric("Labels", stats.get("total_boxes", 0) if stats else 0)
    
    st.markdown("---")
    status_text = "🟢 System Online" if api_state.get("online") else "🔴 Offline"
    st.info(f"**Backend:** {status_text}\n\n**Model:** {'✅ Operational' if api_state.get('model_loaded') else '❌ Not Loaded'}")

# ============================================================================
# MAIN INTERFACE
# ============================================================================
page = st.session_state.page

# Shared Header
st.markdown('<div class="main-title">Vision Plates</div>', unsafe_allow_html=True)
st.markdown(f"Advanced Distributed Inference powered by <span class='spark-badge'>PySpark Cluster</span>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

if page == "📥 Upload Analysis":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📥 Intelligent Batch Upload")
    st.write("Drag and drop multiple images for high-speed concurrent analysis.")
    
    uploaded_files = st.file_uploader("Sélectionnez vos images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, label_visibility="collapsed")
    
    if st.button("🚀 Lancer l'analyse intensive", use_container_width=True) and uploaded_files:
        with st.spinner("Processing on Spark Cluster..."):
            result = predict_batch_images(uploaded_files)
            if result and result.get("success"):
                st.success(f"Execution successful: {result['total_plates']} detections across {len(result['results'])} images.")
                
                for i, res in enumerate(result['results']):
                    with st.expander(f"🔮 Result: {res['filename']} ({res['nb_plates']} plates)", expanded=(i==0)):
                        col1, col2 = st.columns(2)
                        with col1:
                            if res.get("original_image_path"):
                                img = load_image_from_path(res["original_image_path"])
                                if img: st.image(img, caption="Original Stream")
                            else:
                                raw_img = next(f for f in uploaded_files if f.name == res['filename'])
                                st.image(Image.open(BytesIO(raw_img.getvalue())), caption="Original Input")
                        with col2:
                            if res.get("annotated_image_path"):
                                ann = load_image_from_path(res["annotated_image_path"])
                                if ann: st.image(ann, caption="AI Detection Layer")
                        
                        # Data view below
                        st.markdown(f"**Intelligence Summary:** Found {res['nb_plates']} license plates.")
                        if res.get("detections"):
                            st.dataframe(pd.DataFrame(res["detections"]), hide_index=True, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "📂 Sample Dataset":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📂 Interactive Model Validation")
    st.write("Stress test the model using historical data from the curated test set.")
    
    if st.button("🔄 Refresh Data Samples") or "sample_paths" not in st.session_state:
        st.session_state["sample_paths"] = get_dataset_samples(24)
        st.rerun()
    
    selected_payload = []
    st.markdown('<div class="selection-img-container">', unsafe_allow_html=True)
    cols = st.columns(4)
    for i, path in enumerate(st.session_state["sample_paths"]):
        with cols[i % 4]:
            st.image(Image.open(path), use_container_width=True)
            if st.checkbox("Select", key=f"v_sel_{path.name}"):
                selected_payload.append(path)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("🔍 Run Selection Analysis", disabled=not selected_payload, use_container_width=True):
        with st.spinner("Analyzing selected segments..."):
            class MockUpload:
                def __init__(self, p):
                    self.name = p.name
                    with open(p, "rb") as f: self.data = f.read()
                    self.type = "image/jpeg"
                def getvalue(self): return self.data
            
            results = predict_batch_images([MockUpload(p) for p in selected_payload])
            if results and results.get("success"):
                for res in results['results']:
                    with st.expander(f"📉 Inference: {res['filename']}", expanded=True):
                        c1, c2 = st.columns(2)
                        with c1:
                            l_img = load_image_from_path(res.get("original_image_path"))
                            if not l_img: l_img = Image.open(next(p for p in selected_payload if p.name==res['filename']))
                            st.image(l_img, caption="Calibration Ref")
                        with c2:
                            l_ann = load_image_from_path(res.get("annotated_image_path"))
                            if l_ann: st.image(l_ann, caption="Neural Mapping")
                        
                        st.markdown(f"**Metrics:** Detected {res['nb_plates']} plates.")
                        if res.get("detections"):
                            st.dataframe(pd.DataFrame(res["detections"]), hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "🕘 Prediction History":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🕘 Continuous Audit Log")
    st.write("Access full history of remote processing runs and detection metadata.")
    
    hist_data = load_history_csv(100)
    h_stats = get_history_stats()
    
    k1, k2, k3 = st.columns(3)
    k1.metric("Analyses Logged", h_stats["total_predictions"])
    k2.metric("Total Plates Found", h_stats["total_plates"])
    with k3:
        if st.button("🗑️ Clear Prediction Log", use_container_width=True):
            requests.delete(f"{API_URL}/history")
            st.rerun()
            
    st.markdown("---")
    
    if not hist_data:
        st.info("No audit logs found. Predictions will appear here once processed.")
    else:
        for item in hist_data:
            with st.expander(f"📜 {item['filename']} — {item['datetime'][:19]}", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    h_img = load_image_from_path(item["original_image_path"])
                    if h_img: st.image(h_img, caption="Archived Original")
                with c2:
                    h_ann = load_image_from_path(item["annotated_image_path"])
                    if h_ann: st.image(h_ann, caption="Processed Layer")
                
                st.markdown(f"**Metadata:** Status `{item['status']}` | Total Plates: `{item['nb_plates']}`")
                if item.get("detections") and item["detections"] != "[]":
                    try:
                        d_parsed = ast.literal_eval(item["detections"])
                        st.dataframe(pd.DataFrame(d_parsed), hide_index=True, use_container_width=True)
                    except:
                        st.code(item["detections"])
    st.markdown('</div>', unsafe_allow_html=True)

# Footer Styling
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center; color:#64748b; font-size:0.85rem; letter-spacing: 0.05em; opacity: 0.7;">'
    'ENGINEERED BY <strong>NEURAL PLATES LAB</strong> &bull; POWERED BY <strong>PYSPARK ENGINE</strong>'
    '</div>',
    unsafe_allow_html=True
)
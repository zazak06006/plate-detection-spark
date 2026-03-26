"""
<<<<<<< HEAD
Microservice 3 — Interface Web (VERSION AMÉLIORÉE)
=================================================
✔ Historique avec images miniatures
✔ Encodage base64
✔ UI plus visuelle
"""

import os
import io
import base64
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
=======
Streamlit Frontend pour la détection de plaques d'immatriculation.

Features:
    - Upload d'une ou plusieurs images
    - Prédiction via FastAPI
    - Traitement batch avec PySpark (automatique si > 3 images)
    - Historique persistant en CSV
    - Interface moderne

Usage:
    streamlit run app.py
"""

import streamlit as st
import requests
import pandas as pd
import base64
import datetime
import csv
from io import BytesIO
from pathlib import Path
from PIL import Image
from typing import List, Dict, Optional
>>>>>>> b4f034a88036f2e72f5b14562e2d27c46768f651

# ============================================================================
# CONFIG
# ============================================================================
API_URL = "http://localhost:8000"  # FastAPI (port 8000)
HISTORY_CSV = Path(__file__).parent / "history.csv"

<<<<<<< HEAD
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
=======
st.set_page_config(
    page_title="License Plate Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Cache pour éviter les appels répétés
@st.cache_data(ttl=5)
def get_api_health():
    """Vérifie l'état de l'API"""
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {"online": False, "model_loaded": False, "data_loaded": False}
>>>>>>> b4f034a88036f2e72f5b14562e2d27c46768f651


<<<<<<< HEAD
# ─────────────────────────────────────────────
# GLOBALS
# ─────────────────────────────────────────────
df = None
model = None
history = []   # ✅ HISTORIQUE GLOBAL

# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
def load_data():
    global df
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        print(f"✅ CSV chargé : {df.shape}")

def load_model():
    global model
    if MODEL_PATH.exists():
        model = YOLO(str(MODEL_PATH))
        print("✅ Modèle chargé")
=======
@st.cache_data(ttl=30)
def get_api_stats():
    """Récupère les statistiques"""
    try:
        r = requests.get(f"{API_URL}/api/stats", timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


# ============================================================================
# CSS MODERNE
# ============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
>>>>>>> b4f034a88036f2e72f5b14562e2d27c46768f651

#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

<<<<<<< HEAD
# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def index():
    stats = {}
    if df is not None:
        stats = {
            "total_images": int(df["image_name"].nunique()),
            "total_boxes": len(df),
        }
    return render_template("index.html", stats=stats)


# =====================================================
# 🔥 PREDICT (AVEC HISTORIQUE IMAGE)
# =====================================================
@app.route("/predict", methods=["GET", "POST"])
def predict():
    global history

    prediction = None
    result_img = None
    error = None

    if request.method == "POST":
        if "file" not in request.files:
            error = "Aucun fichier"
        else:
            file = request.files["file"]

            if file.filename == "":
                error = "Fichier vide"
            else:
                img_path = UPLOAD_DIR / file.filename
                file.save(str(img_path))

                # ✅ Encode image originale
                with open(img_path, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode()

                if model is None:
                    error = "Modèle non chargé"
                else:
                    try:
                        results = model.predict(str(img_path), conf=0.25, verbose=False)
                        result = results[0]

                        # Image annotée
                        img_annotated = result.plot()[:, :, ::-1]
                        pil_img = Image.fromarray(img_annotated)
                        buf = io.BytesIO()
                        pil_img.save(buf, format="PNG")
                        buf.seek(0)
                        result_img = base64.b64encode(buf.read()).decode()

                        detections = []
                        for box in result.boxes:
                            detections.append({
                                "confidence": round(float(box.conf[0]), 3),
                                "x_min": round(float(box.xyxy[0][0]), 1),
                                "y_min": round(float(box.xyxy[0][1]), 1),
                                "x_max": round(float(box.xyxy[0][2]), 1),
                                "y_max": round(float(box.xyxy[0][3]), 1),
                            })

                        prediction = {
                            "filename": file.filename,
                            "image_b64": img_b64,   # ✅ IMAGE MINIATURE
                            "nb_plates": len(detections),
                            "detections": detections,
                        }

                        # ✅ AJOUT HISTORIQUE
                        history.append(prediction)

                        # Limite historique (20 images)
                        history = history[-20:]

                    except Exception as e:
                        error = str(e)

    return render_template(
        "predict.html",
        prediction=prediction,
        result_img=result_img,
        error=error,
        history=history[::-1]  # récent en premier
    )


# =====================================================
# API STATS
# =====================================================
@app.route("/api/stats")
def api_stats():
    if df is None:
        return jsonify({"error": "CSV non chargé"}), 404

    return jsonify({
        "total_images": int(df["image_name"].nunique()),
        "total_boxes": len(df),
    })


# =====================================================
# HEALTH
# =====================================================
@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "data_loaded": df is not None
    })


# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    app.run(debug=True, port=5000)
=======
html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(99,102,241,0.18), transparent 28%),
        radial-gradient(circle at top right, rgba(6,182,212,0.16), transparent 25%),
        linear-gradient(135deg, #0b1120 0%, #111827 45%, #0f172a 100%);
    color: #e5e7eb;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

.main-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #22d3ee, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}

.subtitle {
    color: #94a3b8;
    font-size: 1rem;
    margin-bottom: 1.2rem;
}

.card {
    background: rgba(15, 23, 42, 0.72);
    border: 1px solid rgba(148, 163, 184, 0.16);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 22px;
    padding: 1.2rem 1.2rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.20);
    animation: fadeUp 0.55s ease;
    margin-bottom: 1rem;
}

.kpi-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.20), rgba(6,182,212,0.10));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 1rem 1.1rem;
    min-height: 115px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.18);
    animation: fadeUp 0.6s ease;
}

.kpi-title {
    color: #94a3b8;
    font-size: 0.92rem;
    font-weight: 600;
    margin-bottom: 0.35rem;
}

.kpi-value {
    font-size: 1.7rem;
    font-weight: 800;
    color: #f8fafc;
}

.kpi-sub {
    color: #cbd5e1;
    font-size: 0.85rem;
    margin-top: 0.3rem;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 0.65rem 1rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.95rem;
    border: 1px solid rgba(255,255,255,0.10);
}

.status-on {
    background: rgba(16, 185, 129, 0.12);
    color: #34d399;
}

.status-off {
    background: rgba(239, 68, 68, 0.12);
    color: #f87171;
}

.section-title {
    font-size: 1.08rem;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 0.8rem;
}

.small-muted {
    color: #94a3b8;
    font-size: 0.92rem;
}

.prediction-box {
    background: linear-gradient(135deg, rgba(99,102,241,0.16), rgba(34,211,238,0.08));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 0.9rem 1rem;
    margin-top: 0.6rem;
}

.prediction-chip {
    display: inline-block;
    background: rgba(34, 211, 238, 0.12);
    color: #67e8f9;
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 700;
    margin-right: 0.45rem;
    margin-bottom: 0.45rem;
    border: 1px solid rgba(103,232,249,0.18);
}

.history-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 0.9rem;
    margin-bottom: 0.75rem;
    animation: fadeUp 0.55s ease;
}

.history-title {
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 0.2rem;
}

.history-meta {
    color: #94a3b8;
    font-size: 0.84rem;
}

hr.custom {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(148,163,184,0.25), transparent);
    margin: 1rem 0 0.7rem 0;
}

.stButton > button {
    width: 100%;
    border: none;
    border-radius: 14px;
    padding: 0.85rem 1rem;
    font-weight: 800;
    color: white;
    background: linear-gradient(90deg, #6366f1, #06b6d4);
    box-shadow: 0 8px 24px rgba(37, 99, 235, 0.28);
    transition: all 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-1px);
    filter: brightness(1.06);
}

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03);
    border: 1px dashed rgba(148,163,184,0.26);
    border-radius: 16px;
    padding: 0.6rem;
}

div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    padding: 12px;
    border-radius: 16px;
}

.spark-badge {
    display: inline-block;
    background: rgba(255, 165, 0, 0.15);
    color: #ffa500;
    padding: 0.25rem 0.6rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    margin-left: 0.5rem;
}

@keyframes fadeUp {
    from {
        opacity: 0;
        transform: translateY(12px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# HISTORY FUNCTIONS (CSV)
# ============================================================================
def load_history_csv(limit: int = 30) -> List[Dict]:
    """Charge l'historique depuis le CSV"""
    if not HISTORY_CSV.exists():
        return []

    try:
        entries = []
        with open(HISTORY_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                entries.append({
                    "datetime": row.get('timestamp', ''),
                    "filename": row.get('image_name', ''),
                    "nb_plates": int(row.get('nb_plates', 0)),
                    "status": row.get('status', 'unknown'),
                    "detections": row.get('detections', '[]')
                })
        return list(reversed(entries[-limit:]))
    except Exception as e:
        st.error(f"Error loading history: {e}")
        return []


def get_history_stats() -> Dict:
    """Calcule les stats depuis l'historique"""
    history = load_history_csv(limit=1000)
    total_predictions = len(history)
    total_plates = sum(h.get('nb_plates', 0) for h in history)
    return {
        "total_predictions": total_predictions,
        "total_plates": total_plates
    }


# ============================================================================
# API HELPERS
# ============================================================================
def decode_base64_image(base64_string: str) -> Image.Image:
    """Décode une image base64"""
    image_bytes = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_bytes))


def predict_single_image(uploaded_file) -> Optional[Dict]:
    """Prédiction sur une seule image"""
    try:
        files = {
            "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        }
        r = requests.post(f"{API_URL}/predict", files=files, timeout=120)
        if r.status_code == 200:
            return r.json()
        else:
            st.error(f"API error: {r.text}")
            return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


def predict_batch_images(uploaded_files: List) -> Optional[Dict]:
    """Prédiction batch sur plusieurs images"""
    try:
        files = [
            ("files", (f.name, f.getvalue(), f.type))
            for f in uploaded_files
        ]
        r = requests.post(
            f"{API_URL}/predict_batch",
            files=files,
            params={"use_spark": True},
            timeout=300
        )
        if r.status_code == 200:
            return r.json()
        else:
            st.error(f"Batch API error: {r.text}")
            return None
    except Exception as e:
        st.error(f"Batch request failed: {e}")
        return None


# ============================================================================
# HEADER
# ============================================================================
api_state = get_api_health()
stats = get_api_stats()

left_header, right_header = st.columns([4, 1.2])

with left_header:
    st.markdown('<div class="main-title">🚗 License Plate Detection</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Upload images for license plate detection. '
        '<span class="spark-badge">⚡ PySpark</span> enabled for batch processing.</div>',
        unsafe_allow_html=True
    )

with right_header:
    is_online = api_state.get("online", False)
    status_class = "status-on" if is_online else "status-off"
    status_icon = "🟢" if is_online else "🔴"
    status_text = "API Online" if is_online else "API Offline"

    st.markdown(
        f"""
        <div class="card" style="text-align:center;">
            <div class="status-pill {status_class}">{status_icon} {status_text}</div>
            <div style="height:10px;"></div>
            <div class="small-muted">Model: {"✅ loaded" if api_state.get("model_loaded") else "❌ not loaded"}</div>
            <div class="small-muted">Device: {api_state.get("device", "cpu")}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================================
# KPI CARDS
# ============================================================================
k1, k2, k3, k4 = st.columns(4)

total_images = stats.get("total_images", 0) if stats else 0
total_boxes = stats.get("total_boxes", 0) if stats else 0
avg_ratio = stats.get("avg_aspect_ratio", 0) if stats else 0
history_stats = get_history_stats()

with k1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">📊 Dataset Images</div>
        <div class="kpi-value">{total_images:,}</div>
        <div class="kpi-sub">Training dataset size</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">📦 Bounding Boxes</div>
        <div class="kpi-value">{total_boxes:,}</div>
        <div class="kpi-sub">Dataset annotations</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">🔍 Predictions</div>
        <div class="kpi-value">{history_stats['total_predictions']}</div>
        <div class="kpi-sub">Total predictions made</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">🚘 Plates Detected</div>
        <div class="kpi-value">{history_stats['total_plates']}</div>
        <div class="kpi-sub">All-time detections</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)

# ============================================================================
# MAIN CONTENT
# ============================================================================
left_col, right_col = st.columns([2.1, 1])

with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📤 Upload & Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="small-muted">Upload one or more images. '
        'Batch mode with PySpark is automatically enabled for 3+ images.</div>',
        unsafe_allow_html=True
    )

    uploaded_files = st.file_uploader(
        "Choose images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        run_prediction = st.button("🔍 Run Prediction")
    with col_btn2:
        use_spark = st.checkbox("Force PySpark", value=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Process prediction
    if run_prediction:
        if not api_state.get("online"):
            st.error("❌ FastAPI is not reachable. Start it with: `python api.py`")
        elif not api_state.get("model_loaded"):
            st.error("❌ Model is not loaded on the API side.")
        elif not uploaded_files:
            st.warning("⚠️ Please upload at least one image.")
        else:
            num_files = len(uploaded_files)

            # Batch mode (PySpark) si >= 3 images
            if num_files >= 3 and use_spark:
                with st.spinner(f"🚀 Processing {num_files} images with PySpark..."):
                    result = predict_batch_images(uploaded_files)

                if result and result.get("success"):
                    st.success(
                        f"✅ Batch processed! {result['total_images']} images, "
                        f"{result['total_plates']} plates detected "
                        f"({'⚡ Spark' if result.get('used_spark') else '📷 Simple'}) "
                        f"in {result['processing_time_ms']:.0f}ms"
                    )

                    # Afficher les résultats
                    for i, res in enumerate(result.get("results", [])):
                        with st.expander(f"🖼️ {res['filename']} - {res['nb_plates']} plates", expanded=(i == 0)):
                            col1, col2 = st.columns(2)

                            with col1:
                                # Image originale
                                orig_file = next((f for f in uploaded_files if f.name == res['filename']), None)
                                if orig_file:
                                    st.image(Image.open(BytesIO(orig_file.getvalue())), caption="Original")

                            with col2:
                                # Image annotée
                                if res.get("annotated_image"):
                                    st.image(decode_base64_image(res["annotated_image"]), caption="Annotated")

                            # Détails
                            if res.get("detections"):
                                st.dataframe(
                                    pd.DataFrame(res["detections"]),
                                    use_container_width=True
                                )
            else:
                # Mode single image
                for file in uploaded_files:
                    with st.spinner(f"🔍 Analyzing {file.name}..."):
                        result = predict_single_image(file)

                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown(f'<div class="section-title">🖼️ Result — {file.name}</div>', unsafe_allow_html=True)

                    if result and result.get("success"):
                        col_img1, col_img2 = st.columns(2)

                        with col_img1:
                            st.image(Image.open(BytesIO(file.getvalue())), caption="Original", use_container_width=True)

                        with col_img2:
                            if result.get("annotated_image"):
                                st.image(
                                    decode_base64_image(result["annotated_image"]),
                                    caption="Annotated",
                                    use_container_width=True
                                )

                        nb_plates = result.get("nb_plates", 0)
                        detections = result.get("detections", [])

                        st.markdown(f"""
                        <div class="prediction-box">
                            <div style="font-size:1rem; font-weight:800; color:#f8fafc;">
                                🚘 Detected plates: {nb_plates}
                            </div>
                            <div class="small-muted" style="margin-top:6px;">
                                Processing time: {result.get('processing_time_ms', 0):.0f}ms
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        if detections:
                            st.markdown("**Detection details:**")
                            st.dataframe(pd.DataFrame(detections), use_container_width=True)
                        else:
                            st.info("No license plate detected in this image.")
                    else:
                        st.error("Prediction failed for this image.")

                    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# HISTORY SIDEBAR
# ============================================================================
with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🕘 Prediction History</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-muted">Latest predictions (from CSV).</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Load history from CSV
    history = load_history_csv(limit=30)

    if history:
        hist_stats = get_history_stats()
        a, b = st.columns(2)
        with a:
            st.metric("Predictions", hist_stats["total_predictions"])
        with b:
            st.metric("Plates Found", hist_stats["total_plates"])

        for item in history[:15]:  # Afficher les 15 dernières
            # Formater le timestamp
            try:
                dt = datetime.datetime.fromisoformat(item["datetime"])
                formatted_dt = dt.strftime("%d/%m/%Y %H:%M")
            except Exception:
                formatted_dt = item["datetime"][:16] if item["datetime"] else "Unknown"

            st.markdown(f"""
            <div class="history-card">
                <div class="history-title">{item["filename"]}</div>
                <div class="history-meta">{formatted_dt}</div>
                <hr class="custom">
                <div style="display:flex; justify-content:space-between;">
                    <div>Plates detected</div>
                    <div><strong>{item["nb_plates"]}</strong></div>
                </div>
                <div class="small-muted">{item["status"]}</div>
            </div>
            """, unsafe_allow_html=True)

        # Download button
        if history:
            df_history = pd.DataFrame(history)
            csv_data = df_history.to_csv(index=False)
            st.download_button(
                "⬇️ Download History CSV",
                data=csv_data,
                file_name="prediction_history.csv",
                mime="text/csv"
            )
    else:
        st.markdown("""
        <div class="history-card">
            <div class="history-title">No predictions yet</div>
            <div class="history-meta">History will appear here after first analysis.</div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
    '<div style="text-align:center; color:#64748b; font-size:0.85rem;">'
    '🚀 Powered by <strong>SSD</strong> + <strong>FastAPI</strong> + <strong>PySpark</strong> + <strong>Streamlit</strong>'
    '</div>',
    unsafe_allow_html=True
)
>>>>>>> b4f034a88036f2e72f5b14562e2d27c46768f651

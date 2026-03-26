import streamlit as st
import requests
import pandas as pd
import base64
import datetime
from io import BytesIO
from PIL import Image
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)
# =========================================================
# CONFIG
# =========================================================
API_URL = "http://localhost:5000"

st.set_page_config(
    page_title="Détection de plaques",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# CSS MODERNE
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

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

# =========================================================
# STATE
# =========================================================
if "history" not in st.session_state:
    st.session_state.history = []

# =========================================================
# HELPERS
# =========================================================
def check_api_health():
    try:
        r = requests.get(f"{API_URL}/api/health", timeout=2)
        if r.status_code == 200:
            data = r.json()
            return {
                "online": True,
                "model_loaded": data.get("model_loaded", False),
                "data_loaded": data.get("data_loaded", False)
            }
    except Exception:
        pass
    return {
        "online": False,
        "model_loaded": False,
        "data_loaded": False
    }

def get_stats():
    try:
        r = requests.get(f"{API_URL}/api/stats", timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def decode_base64_image(base64_string):
    image_bytes = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_bytes))

def predict_image(uploaded_file):
    try:
        files = {
            "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        }
        r = requests.post(f"{API_URL}/api/predict", files=files, timeout=120)
        return r
    except Exception as e:
        return None

def add_to_history(filename, nb_plates, status, detections=None):
    st.session_state.history.insert(0, {
        "datetime": datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "filename": filename,
        "nb_plates": nb_plates,
        "status": status,
        "detections": detections if detections else []
    })

# =========================================================
# HEADER
# =========================================================
api_state = check_api_health()
stats = get_stats()

left_header, right_header = st.columns([4, 1.2])

with left_header:
    st.markdown('<div class="main-title">🚗 Détection intelligente de plaques</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Interface Streamlit pour la prédiction à partir d’une ou plusieurs images.</div>',
        unsafe_allow_html=True
    )

with right_header:
    status_class = "status-on" if api_state["online"] else "status-off"
    status_icon = "🟢" if api_state["online"] else "🔴"
    status_text = "API en ligne" if api_state["online"] else "API hors ligne"

    st.markdown(
        f"""
        <div class="card" style="text-align:center;">
            <div class="status-pill {status_class}">{status_icon} {status_text}</div>
            <div style="height:10px;"></div>
            <div class="small-muted">Modèle : {"chargé" if api_state["model_loaded"] else "non chargé"}</div>
            <div class="small-muted">Données : {"chargées" if api_state["data_loaded"] else "non chargées"}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================================================
# KPI
# =========================================================
k1, k2, k3, k4 = st.columns(4)

total_images = stats.get("total_images", 0) if stats else 0
total_boxes = stats.get("total_boxes", 0) if stats else 0
avg_ratio = stats.get("avg_aspect_ratio", 0) if stats else 0
avg_area = stats.get("avg_bbox_area_norm", 0) if stats else 0

with k1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Images dataset</div>
        <div class="kpi-value">{total_images}</div>
        <div class="kpi-sub">Total des images disponibles</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Bounding boxes</div>
        <div class="kpi-value">{total_boxes}</div>
        <div class="kpi-sub">Annotations du dataset</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Aspect ratio moyen</div>
        <div class="kpi-value">{avg_ratio}</div>
        <div class="kpi-sub">Moyenne largeur / hauteur</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Aire bbox moyenne</div>
        <div class="kpi-value">{avg_area}</div>
        <div class="kpi-sub">BBox normalisée moyenne</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)

# =========================================================
# LAYOUT PRINCIPAL
# =========================================================
left_col, right_col = st.columns([2.1, 1])

# =========================================================
# COLONNE GAUCHE - PREDICTION
# =========================================================
with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📤 Upload & prédiction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="small-muted">Charge une ou plusieurs images puis lance la détection des plaques.</div>',
        unsafe_allow_html=True
    )

    uploaded_files = st.file_uploader(
        "Choisir une ou plusieurs images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    run_prediction = st.button("🔍 Lancer la prédiction")

    st.markdown('</div>', unsafe_allow_html=True)

    if run_prediction:
        if not api_state["online"]:
            st.error("L’API Flask n’est pas accessible. Vérifie que le microservice tourne sur http://localhost:5000")
        elif not api_state["model_loaded"]:
            st.error("Le modèle YOLO n’est pas chargé côté Flask.")
        elif not uploaded_files:
            st.warning("Ajoute au moins une image.")
        else:
            for file in uploaded_files:
                with st.spinner(f"Analyse de {file.name}..."):
                    response = predict_image(file)

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f'<div class="section-title">🖼️ Résultat — {file.name}</div>', unsafe_allow_html=True)

                if response is None:
                    st.error("Impossible de contacter l’API.")
                    add_to_history(file.name, 0, "Erreur API")
                elif response.status_code != 200:
                    try:
                        err = response.json().get("error", "Erreur inconnue")
                    except Exception:
                        err = response.text
                    st.error(err)
                    add_to_history(file.name, 0, "Erreur")
                else:
                    data = response.json()

                    col_img1, col_img2 = st.columns(2)

                    with col_img1:
                        original_img = Image.open(BytesIO(file.getvalue()))
                        st.image(original_img, caption="Image originale", use_container_width=True)

                    with col_img2:
                        if data.get("annotated_image"):
                            annotated = decode_base64_image(data["annotated_image"])
                            st.image(annotated, caption="Image annotée", use_container_width=True)

                    nb_plates = data.get("nb_plates", 0)
                    detections = data.get("detections", [])

                    st.markdown(
                        f"""
                        <div class="prediction-box">
                            <div style="font-size:1rem; font-weight:800; color:#f8fafc;">
                                Plaques détectées : {nb_plates}
                            </div>
                            <div class="small-muted" style="margin-top:6px;">
                                Résultat de la détection pour cette image.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    if detections:
                        st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
                        st.markdown("**Détails des détections**")

                        det_rows = []
                        for i, det in enumerate(detections, start=1):
                            det_rows.append({
                                "Plaque": f"Détection {i}",
                                "Confiance": det["confidence"],
                                "x_min": det["x_min"],
                                "y_min": det["y_min"],
                                "x_max": det["x_max"],
                                "y_max": det["y_max"],
                            })

                        st.dataframe(pd.DataFrame(det_rows), use_container_width=True)

                        st.markdown('<div style="margin-top:8px;">', unsafe_allow_html=True)
                        for i, det in enumerate(detections, start=1):
                            st.markdown(
                                f"""
                                <span class="prediction-chip">
                                    Détection {i} — conf {det["confidence"]}
                                </span>
                                """,
                                unsafe_allow_html=True
                            )
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.info("Aucune plaque détectée sur cette image.")

                    add_to_history(file.name, nb_plates, "Succès", detections)

                st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# COLONNE DROITE - HISTORIQUE
# =========================================================
with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🕘 Historique des prédictions</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="small-muted">Les dernières analyses effectuées dans cette session.</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.history:
        total_preds = len(st.session_state.history)
        total_detected = sum(item["nb_plates"] for item in st.session_state.history)

        a, b = st.columns(2)
        with a:
            st.metric("Prédictions", total_preds)
        with b:
            st.metric("Plaques détectées", total_detected)

        st.markdown('<div style="height: 8px;"></div>', unsafe_allow_html=True)

        for item in st.session_state.history:
            badge_color = "#34d399" if item["status"] == "Succès" else "#f87171"
            st.markdown(
                f"""
                <div class="history-card">
                    <div class="history-title">{item["filename"]}</div>
                    <div class="history-meta">{item["datetime"]}</div>
                    <hr class="custom">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div style="color:#cbd5e1;">Plaques détectées</div>
                        <div style="font-weight:800; color:#f8fafc;">{item["nb_plates"]}</div>
                    </div>
                    <div style="margin-top:8px; color:{badge_color}; font-weight:700;">
                        {item["status"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        hist_df = pd.DataFrame(st.session_state.history)
        st.download_button(
            "⬇️ Télécharger l’historique CSV",
            data=hist_df.to_csv(index=False).encode("utf-8"),
            file_name="historique_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.markdown(
            """
            <div class="history-card">
                <div class="history-title">Aucune prédiction</div>
                <div class="history-meta">L’historique apparaîtra ici après la première analyse.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
"""
Microservice 3 — Interface Web
================================
Flask API servant :
  - /           → Dashboard EDA & visualisations
  - /predict    → Prédiction sur image uploadée (YOLOv8)
  - /api/stats  → JSON des statistiques du dataset
"""

import os
import io
import base64
import json
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_from_directory
from ultralytics import YOLO

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
CSV_PATH    = BASE_DIR / "../1-preprocessing-pyspark/output/dataset_ready.csv"
MODEL_PATH  = BASE_DIR / "model/best.pt"
UPLOAD_DIR  = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload

@app.template_filter("format_number")
def format_number(value):
    try:
        return f"{int(value):,}".replace(",", "\u202f")
    except (ValueError, TypeError):
        return value

sns.set_theme(style="darkgrid")

# ─────────────────────────────────────────────
# Chargement modèle & données
# ─────────────────────────────────────────────
df = None
model = None

def load_data():
    global df
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        print(f"✅ CSV chargé : {df.shape}")
    else:
        print(f"⚠️  CSV introuvable : {CSV_PATH}")

def load_model():
    global model
    if MODEL_PATH.exists():
        model = YOLO(str(MODEL_PATH))
        print("✅ Modèle YOLOv8 chargé")
    else:
        print(f"⚠️  Modèle introuvable : {MODEL_PATH} — entraîne d'abord le Microservice 2")

load_data()
load_model()

# ─────────────────────────────────────────────
# Helpers — génération de graphes en base64
# ─────────────────────────────────────────────
def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="#0f172a", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

DARK_BG  = "#0f172a"
CARD_BG  = "#1e293b"
ACCENT1  = "#6366f1"
ACCENT2  = "#06b6d4"
ACCENT3  = "#10b981"
ACCENT4  = "#f59e0b"
TEXT_CLR = "#e2e8f0"

def make_pie_split():
    if df is None: return ""
    counts = df.groupby("split")["image_name"].nunique()
    fig, ax = plt.subplots(figsize=(5, 4), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=counts.index, autopct="%1.1f%%",
        colors=[ACCENT1, ACCENT2, ACCENT3], textprops={"color": TEXT_CLR}
    )
    for at in autotexts: at.set_color(DARK_BG)
    ax.set_title("Répartition Train / Valid / Test", color=TEXT_CLR, fontsize=13)
    return fig_to_b64(fig)

def make_bbox_size_bar():
    if df is None: return ""
    counts = df["bbox_size_category"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4), facecolor=DARK_BG)
    ax.set_facecolor(CARD_BG)
    bars = ax.bar(counts.index, counts.values,
                  color=[ACCENT1, ACCENT2, ACCENT3], edgecolor="none", width=0.5)
    ax.set_title("Distribution tailles BBox", color=TEXT_CLR, fontsize=13)
    ax.tick_params(colors=TEXT_CLR)
    ax.spines[:].set_visible(False)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f"{int(bar.get_height()):,}", ha="center", color=TEXT_CLR, fontsize=10)
    return fig_to_b64(fig)

def make_aspect_ratio_hist():
    if df is None: return ""
    fig, ax = plt.subplots(figsize=(5, 4), facecolor=DARK_BG)
    ax.set_facecolor(CARD_BG)
    ax.hist(df["aspect_ratio"].dropna(), bins=60, color=ACCENT2, alpha=0.85, edgecolor="none")
    ax.set_title("Distribution Aspect Ratio (w/h)", color=TEXT_CLR, fontsize=13)
    ax.set_xlabel("Ratio", color=TEXT_CLR)
    ax.tick_params(colors=TEXT_CLR)
    ax.spines[:].set_visible(False)
    return fig_to_b64(fig)

def make_heatmap():
    if df is None: return ""
    fig, ax = plt.subplots(figsize=(5, 4), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    h = ax.hist2d(df["cx_norm"].dropna(), df["cy_norm"].dropna(), bins=50, cmap="plasma")
    fig.colorbar(h[3], ax=ax, label="Densité").ax.yaxis.set_tick_params(color=TEXT_CLR)
    ax.set_title("Heatmap positions centres BBox", color=TEXT_CLR, fontsize=13)
    ax.set_xlabel("cx (normalisé)", color=TEXT_CLR)
    ax.set_ylabel("cy (normalisé)", color=TEXT_CLR)
    ax.tick_params(colors=TEXT_CLR)
    ax.spines[:].set_visible(False)
    return fig_to_b64(fig)

def make_bbox_area_boxplot():
    if df is None: return ""
    fig, ax = plt.subplots(figsize=(5, 4), facecolor=DARK_BG)
    ax.set_facecolor(CARD_BG)
    data = [df[df["split"] == s]["bbox_area_norm"].dropna().values for s in ["train", "valid", "test"]]
    bp = ax.boxplot(data, patch_artist=True, labels=["Train", "Valid", "Test"],
                    medianprops=dict(color=ACCENT4, linewidth=2))
    for patch, color in zip(bp["boxes"], [ACCENT1, ACCENT2, ACCENT3]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title("Aire normalisée BBox par split", color=TEXT_CLR, fontsize=13)
    ax.tick_params(colors=TEXT_CLR)
    ax.spines[:].set_visible(False)
    return fig_to_b64(fig)

def make_num_boxes_hist():
    if df is None: return ""
    img_boxes = df.groupby("image_name")["num_boxes"].first()
    fig, ax = plt.subplots(figsize=(5, 4), facecolor=DARK_BG)
    ax.set_facecolor(CARD_BG)
    ax.hist(img_boxes, bins=range(0, img_boxes.max()+2), color=ACCENT3, alpha=0.85, edgecolor="none")
    ax.set_title("Nb de plaques par image", color=TEXT_CLR, fontsize=13)
    ax.set_xlabel("Nombre de bboxes", color=TEXT_CLR)
    ax.tick_params(colors=TEXT_CLR)
    ax.spines[:].set_visible(False)
    return fig_to_b64(fig)

# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/")
def index():
    charts = {
        "pie_split":       make_pie_split(),
        "bbox_size":       make_bbox_size_bar(),
        "aspect_ratio":    make_aspect_ratio_hist(),
        "heatmap":         make_heatmap(),
        "bbox_area":       make_bbox_area_boxplot(),
        "num_boxes":       make_num_boxes_hist(),
    }
    stats = {}
    if df is not None:
        stats = {
            "total_images":  int(df["image_name"].nunique()),
            "total_boxes":   len(df),
            "avg_boxes":     round(df.groupby("image_name")["num_boxes"].first().mean(), 2),
            "mean_area":     round(float(df["bbox_area_norm"].mean()), 4),
            "model_ready":   model is not None,
        }
    return render_template("index.html", charts=charts, stats=stats)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    result_img = None
    error = None

    if request.method == "POST":
        if "file" not in request.files:
            error = "Aucun fichier fourni."
        else:
            file = request.files["file"]
            if file.filename == "":
                error = "Fichier vide."
            else:
                # Sauvegarde temporaire
                img_path = UPLOAD_DIR / file.filename
                file.save(str(img_path))

                if model is None:
                    error = "⚠️ Modèle non chargé. Lance d'abord le Microservice 2."
                else:
                    try:
                        results = model.predict(str(img_path), conf=0.25, verbose=False)
                        result = results[0]

                        # Image annotée en base64
                        img_annotated = result.plot()[:, :, ::-1]
                        pil_img = Image.fromarray(img_annotated)
                        buf = io.BytesIO()
                        pil_img.save(buf, format="PNG")
                        buf.seek(0)
                        result_img = base64.b64encode(buf.read()).decode()

                        # Détections
                        boxes = result.boxes
                        detections = []
                        for box in boxes:
                            detections.append({
                                "confidence": round(float(box.conf[0]), 3),
                                "x_min": round(float(box.xyxy[0][0]), 1),
                                "y_min": round(float(box.xyxy[0][1]), 1),
                                "x_max": round(float(box.xyxy[0][2]), 1),
                                "y_max": round(float(box.xyxy[0][3]), 1),
                            })
                        prediction = {
                            "filename": file.filename,
                            "nb_plates": len(detections),
                            "detections": detections,
                        }
                    except Exception as e:
                        error = f"Erreur lors de la prédiction : {e}"

    return render_template("predict.html",
                           prediction=prediction,
                           result_img=result_img,
                           error=error)


@app.route("/api/stats")
def api_stats():
    if df is None:
        return jsonify({"error": "CSV non chargé"}), 404
    stats = {
        "total_images": int(df["image_name"].nunique()),
        "total_boxes": len(df),
        "splits": df.groupby("split")["image_name"].nunique().to_dict(),
        "bbox_size_dist": df["bbox_size_category"].value_counts().to_dict(),
        "avg_aspect_ratio": round(float(df["aspect_ratio"].mean()), 4),
        "avg_bbox_area_norm": round(float(df["bbox_area_norm"].mean()), 4),
    }
    return jsonify(stats)


if __name__ == "__main__":
    app.run(debug=True, port=5000)

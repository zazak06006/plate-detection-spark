"""
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

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
CSV_PATH    = BASE_DIR / "../1-preprocessing-pyspark/output/dataset_ready.csv"
MODEL_PATH  = BASE_DIR / "model/best.pt"
UPLOAD_DIR  = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

sns.set_theme(style="darkgrid")

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

load_data()
load_model()

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
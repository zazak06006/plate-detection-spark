"""
FastAPI Backend pour la détection de plaques d'immatriculation.

Endpoints:
    - GET  /health         : Vérifie l'état de l'API et du modèle
    - GET  /api/health     : Alias pour compatibilité
    - GET  /api/stats      : Statistiques du dataset
    - POST /predict        : Prédiction sur une image
    - POST /api/predict    : Alias pour compatibilité
    - POST /predict_batch  : Prédiction batch avec PySpark

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
    # ou
    python api.py
"""

import io
import os
import sys
import base64
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import uvicorn

# Import des modules locaux
from inference import (
    get_model, is_model_loaded, get_device,
    predict_single_image, predict_from_bytes,
    annotate_image, save_to_history, load_history
)
from spark_batch import process_images_batch


# ============================================================================
# CONFIG
# ============================================================================
UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Port et host
HOST = os.getenv("API_HOST", "0.0.0.0")
PORT = int(os.getenv("API_PORT", "8000"))


# ============================================================================
# FASTAPI APP
# ============================================================================
app = FastAPI(
    title="License Plate Detection API",
    description="API pour la détection de plaques d'immatriculation avec SSD",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# PYDANTIC MODELS
# ============================================================================
class Detection(BaseModel):
    confidence: float
    x_min: int
    y_min: int
    x_max: int
    y_max: int


class PredictionResponse(BaseModel):
    success: bool
    filename: str
    nb_plates: int
    detections: List[Detection]
    annotated_image: Optional[str] = None  # Base64
    processing_time_ms: float


class BatchPredictionResponse(BaseModel):
    success: bool
    total_images: int
    total_plates: int
    results: List[dict]
    processing_time_ms: float
    used_spark: bool


class HealthResponse(BaseModel):
    status: str
    online: bool
    model_loaded: bool
    data_loaded: bool
    device: str
    timestamp: str


class StatsResponse(BaseModel):
    total_images: int
    total_boxes: int
    avg_aspect_ratio: float
    avg_bbox_area_norm: float


class HistoryEntry(BaseModel):
    timestamp: str
    image_name: str
    nb_plates: int
    status: str


# ============================================================================
# STARTUP EVENT
# ============================================================================
@app.on_event("startup")
async def startup_event():
    """Charge le modèle au démarrage"""
    print("🚀 Starting API server...")
    try:
        _ = get_model()
        print("✅ Model loaded successfully at startup")
    except Exception as e:
        print(f"⚠️ Failed to load model at startup: {e}")


# ============================================================================
# HEALTH ENDPOINTS
# ============================================================================
@app.get("/health", response_model=HealthResponse, tags=["Health"])
@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Vérifie l'état de l'API et du modèle"""
    model_loaded = is_model_loaded()

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        online=True,
        model_loaded=model_loaded,
        data_loaded=True,  # Toujours true pour compatibilité
        device=str(get_device()),
        timestamp=datetime.now().isoformat()
    )


@app.get("/api/stats", response_model=StatsResponse, tags=["Stats"])
async def get_stats():
    """Retourne des statistiques (mock pour compatibilité)"""
    # Ces stats pourraient être calculées depuis le dataset réel
    return StatsResponse(
        total_images=10125,
        total_boxes=12500,
        avg_aspect_ratio=2.8,
        avg_bbox_area_norm=0.045
    )


# ============================================================================
# PREDICTION ENDPOINTS
# ============================================================================
@app.post("/predict", tags=["Prediction"])
@app.post("/api/predict", tags=["Prediction"])
async def predict_single(
    file: UploadFile = File(...),
    score_threshold: float = Query(0.3, ge=0.0, le=1.0),
    nms_threshold: float = Query(0.4, ge=0.0, le=1.0),
    save_upload: bool = Query(False, description="Sauvegarder l'image dans uploads/")
):
    """
    Prédiction sur une seule image.

    - **file**: Image à analyser (JPEG, PNG)
    - **score_threshold**: Seuil de confiance (défaut: 0.3)
    - **nms_threshold**: Seuil NMS (défaut: 0.4)
    - **save_upload**: Sauvegarder l'image uploadée

    Returns:
        - nb_plates: Nombre de plaques détectées
        - detections: Liste des détections avec coordonnées
        - annotated_image: Image annotée en base64
    """
    start_time = datetime.now()

    # Valider le fichier
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Lire le fichier
        contents = await file.read()
        filename = file.filename or "unknown.jpg"

        # Sauvegarder si demandé
        if save_upload:
            upload_path = UPLOAD_DIR / filename
            with open(upload_path, "wb") as f:
                f.write(contents)

        # Prédiction
        predictions, original_image = predict_from_bytes(
            contents,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold
        )

        # Annoter l'image
        annotated = annotate_image(original_image, predictions['detections'])

        # Encoder en base64
        buffer = io.BytesIO()
        annotated.save(buffer, format='JPEG', quality=90)
        annotated_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Sauvegarder dans l'historique
        save_to_history(
            filename,
            predictions['nb_plates'],
            predictions['detections']
        )

        # Temps de traitement
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return {
            "success": True,
            "filename": filename,
            "nb_plates": predictions['nb_plates'],
            "detections": predictions['detections'],
            "annotated_image": annotated_base64,
            "processing_time_ms": processing_time
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict_batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(
    files: List[UploadFile] = File(...),
    score_threshold: float = Query(0.3, ge=0.0, le=1.0),
    nms_threshold: float = Query(0.4, ge=0.0, le=1.0),
    use_spark: bool = Query(True, description="Utiliser PySpark pour le traitement")
):
    """
    Prédiction batch sur plusieurs images.

    - **files**: Liste d'images à analyser
    - **score_threshold**: Seuil de confiance (défaut: 0.3)
    - **nms_threshold**: Seuil NMS (défaut: 0.4)
    - **use_spark**: Utiliser PySpark pour paralléliser

    Returns:
        - total_images: Nombre d'images traitées
        - total_plates: Total de plaques détectées
        - results: Liste des résultats par image
        - used_spark: Si Spark a été utilisé
    """
    start_time = datetime.now()

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    try:
        # Préparer les données
        images_data = []
        for file in files:
            if file.content_type and file.content_type.startswith("image/"):
                contents = await file.read()
                images_data.append((file.filename or "unknown.jpg", contents))

        if not images_data:
            raise HTTPException(status_code=400, detail="No valid images provided")

        # Traitement batch
        results = process_images_batch(
            images_data,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            use_spark=use_spark,
            spark_threshold=3  # Utiliser Spark si >= 3 images
        )

        # Calculer les totaux
        total_plates = sum(r.get('nb_plates', 0) for r in results)
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return BatchPredictionResponse(
            success=True,
            total_images=len(results),
            total_plates=total_plates,
            results=results,
            processing_time_ms=processing_time,
            used_spark=use_spark and len(images_data) >= 3
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


# ============================================================================
# HISTORY ENDPOINTS
# ============================================================================
@app.get("/history", tags=["History"])
async def get_history(limit: int = Query(30, ge=1, le=100)):
    """
    Récupère l'historique des prédictions.

    - **limit**: Nombre max d'entrées (défaut: 30)
    """
    history = load_history(limit=limit)
    return {
        "success": True,
        "count": len(history),
        "entries": history
    }


@app.delete("/history", tags=["History"])
async def clear_history_endpoint():
    """Efface l'historique des prédictions"""
    from inference import clear_history
    clear_history()
    return {"success": True, "message": "History cleared"}


# ============================================================================
# UTILS ENDPOINTS
# ============================================================================
@app.get("/", tags=["Info"])
async def root():
    """Page d'accueil de l'API"""
    return {
        "name": "License Plate Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/device", tags=["Info"])
async def get_device_info():
    """Retourne les informations sur le device utilisé"""
    import torch
    return {
        "device": str(get_device()),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print(f"""
╔══════════════════════════════════════════════════════════╗
║       License Plate Detection API - FastAPI              ║
╠══════════════════════════════════════════════════════════╣
║  Endpoints:                                              ║
║    - GET  /health         : Health check                 ║
║    - GET  /api/stats      : Dataset statistics           ║
║    - POST /predict        : Single image prediction      ║
║    - POST /predict_batch  : Batch prediction (PySpark)   ║
║    - GET  /history        : Prediction history           ║
║    - GET  /docs           : Swagger documentation        ║
╚══════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        "api:app",
        host=HOST,
        port=PORT,
        reload=True,
        log_level="info"
    )

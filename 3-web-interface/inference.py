"""
Module d'inférence pour la détection de plaques d'immatriculation.
Centralise le chargement du modèle et les fonctions de prédiction.

Usage:
    from inference import get_model, predict_single_image, predict_batch_images
"""

import io
import sys
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from functools import lru_cache

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Ajouter le chemin vers le module model
MODEL_TRAINING_PATH = Path(__file__).parent.parent / "2-model-training"
sys.path.insert(0, str(MODEL_TRAINING_PATH))

from model import create_model, predict, decode_predictions, cxcywh_to_xyxy, NUM_CLASSES


# ============================================================================
# CONFIG
# ============================================================================
MODEL_PATH = MODEL_TRAINING_PATH / "checkpoints" / "best_model.pt"
IMG_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HISTORY_CSV_PATH = Path(__file__).parent / "history.csv"

# Transform pour normalisation UNIQUEMENT (le letterbox est fait séparément)
NORMALIZE_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

TRANSFORM = NORMALIZE_TRANSFORM


# ============================================================================
# SINGLETON MODEL LOADER
# ============================================================================
class ModelManager:
    """
    Gestionnaire singleton du modèle.
    Charge le modèle une seule fois et le garde en mémoire.
    """
    _instance = None
    _model = None
    _device = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_model(self, checkpoint_path: str = None, device: str = None) -> torch.nn.Module:
        """Charge le modèle si pas déjà chargé"""
        if self._model is not None:
            return self._model

        checkpoint_path = checkpoint_path or str(MODEL_PATH)
        device = device or DEVICE
        self._device = torch.device(device)

        print(f"🔄 Loading model from {checkpoint_path}...")
        print(f"   Device: {self._device}")

        # Charger le checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self._device, weights_only=False)

        # Créer le modèle
        num_classes = checkpoint.get('num_classes', NUM_CLASSES)
        self._model = create_model(num_classes=num_classes)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._model.to(self._device)
        self._model.eval()

        print(f"✅ Model loaded successfully!")
        return self._model

    @property
    def model(self) -> torch.nn.Module:
        if self._model is None:
            self.load_model()
        return self._model

    @property
    def device(self) -> torch.device:
        if self._device is None:
            self._device = torch.device(DEVICE)
        return self._device

    def is_loaded(self) -> bool:
        return self._model is not None


# Instance globale
_model_manager = ModelManager()


def get_model() -> torch.nn.Module:
    """Retourne le modèle (chargé une seule fois)"""
    return _model_manager.model


def get_device() -> torch.device:
    """Retourne le device utilisé"""
    return _model_manager.device


def is_model_loaded() -> bool:
    """Vérifie si le modèle est chargé"""
    return _model_manager.is_loaded()


# ============================================================================
# PREPROCESSING - LETTERBOX (Cohérent avec Spark)
# ============================================================================
def letterbox_image(image: Image.Image) -> Tuple[Image.Image, float, float, float]:
    """
    Applique un letterbox resize identique à celui de Spark.
    Redimensionne en conservant le ratio, padding noir centré.

    Args:
        image: Image PIL (RGB)

    Returns:
        (letterboxed_image, scale, pad_x_normalized, pad_y_normalized)
    """
    orig_w, orig_h = image.size

    # Calculer le scale pour garder le ratio
    scale = min(IMG_SIZE / orig_w, IMG_SIZE / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    # Redimensionner avec le ratio préservé
    img_resized = image.resize((new_w, new_h), Image.BILINEAR)

    # Créer une nouvelle image noire et coller l'image centrée
    img_letterbox = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
    pad_x = (IMG_SIZE - new_w) // 2
    pad_y = (IMG_SIZE - new_h) // 2
    img_letterbox.paste(img_resized, (pad_x, pad_y))

    # Retourner les infos normalisées
    pad_x_norm = float(pad_x) / IMG_SIZE
    pad_y_norm = float(pad_y) / IMG_SIZE

    return img_letterbox, scale, pad_x_norm, pad_y_norm


def preprocess_image(image: Image.Image) -> Tuple[torch.Tensor, float, float]:
    """
    Prétraite une image PIL pour l'inférence avec letterbox.

    Args:
        image: Image PIL (RGB)

    Returns:
        (tensor, pad_x_norm, pad_y_norm)
        - tensor: [1, 3, 256, 256] normalisé
        - pad_x_norm, pad_y_norm: padding appliqué (pour inverse transform)
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Letterbox resize
    img_letterbox, scale, pad_x_norm, pad_y_norm = letterbox_image(image)

    # Normalisation ImageNet
    tensor = NORMALIZE_TRANSFORM(img_letterbox)

    return tensor.unsqueeze(0), pad_x_norm, pad_y_norm


def preprocess_bytes(image_bytes: bytes) -> Tuple[torch.Tensor, Image.Image, float, float]:
    """
    Prétraite des bytes d'image.

    Args:
        image_bytes: Bytes de l'image

    Returns:
        (tensor, original_image, pad_x_norm, pad_y_norm)
    """
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor, pad_x_norm, pad_y_norm = preprocess_image(image)
    return tensor, image, pad_x_norm, pad_y_norm


def inverse_letterbox_coords(
    boxes_normalized: np.ndarray,
    pad_x_norm: float,
    pad_y_norm: float,
    original_size: Tuple[int, int]
) -> np.ndarray:
    """
    Inverse la transformation letterbox pour obtenir les coordonnées
    dans l'espace de l'image originale.

    Args:
        boxes_normalized: [N, 4] coordonnées (x1, y1, x2, y2) normalisées [0,1] dans l'espace letterbox
        pad_x_norm, pad_y_norm: padding appliqué lors du letterbox
        original_size: (width, height) de l'image originale

    Returns:
        boxes_pixels: [N, 4] coordonnées en pixels dans l'image originale
    """
    if len(boxes_normalized) == 0:
        return boxes_normalized

    # Région de l'image dans le letterbox
    img_region_w = 1.0 - 2.0 * pad_x_norm
    img_region_h = 1.0 - 2.0 * pad_y_norm

    # Inverse transform: coords_orig = (coords_letterbox - pad) / img_region
    boxes_orig = boxes_normalized.copy()
    boxes_orig[:, [0, 2]] = (boxes_normalized[:, [0, 2]] - pad_x_norm) / max(img_region_w, 1e-6)
    boxes_orig[:, [1, 3]] = (boxes_normalized[:, [1, 3]] - pad_y_norm) / max(img_region_h, 1e-6)

    # Clamp to [0, 1]
    boxes_orig = np.clip(boxes_orig, 0, 1)

    # Convertir en pixels
    orig_w, orig_h = original_size
    boxes_pixels = boxes_orig.copy()
    boxes_pixels[:, [0, 2]] *= orig_w
    boxes_pixels[:, [1, 3]] *= orig_h

    return boxes_pixels.astype(int)


# ============================================================================
# INFERENCE SINGLE IMAGE
# ============================================================================
@torch.no_grad()
def predict_single_image(
    image: Image.Image,
    score_threshold: float = 0.3,
    nms_threshold: float = 0.4
) -> Dict:
    """
    Prédit sur une seule image avec letterbox preprocessing.

    Args:
        image: Image PIL
        score_threshold: Seuil de confiance
        nms_threshold: Seuil NMS

    Returns:
        Dict avec:
            - nb_plates: int
            - detections: List[Dict] avec confidence, x_min, y_min, x_max, y_max
            - boxes_normalized: List[List[float]] coordonnées normalisées [0-1] dans l'image originale
    """
    model = get_model()
    device = get_device()

    # Prétraitement avec letterbox
    original_size = image.size  # (width, height)
    tensor, pad_x_norm, pad_y_norm = preprocess_image(image)
    tensor = tensor.to(device)

    # Inférence
    results = predict(model, tensor, score_threshold=score_threshold, nms_threshold=nms_threshold)
    det = results[0]

    # Formater les résultats
    boxes_letterbox = det['boxes'].cpu().numpy()  # Coordonnées dans l'espace letterbox
    scores = det['scores'].cpu().numpy()

    detections = []
    boxes_normalized = []

    if len(boxes_letterbox) > 0:
        # Inverser le letterbox pour obtenir les coords dans l'image originale
        boxes_pixels = inverse_letterbox_coords(
            boxes_letterbox, pad_x_norm, pad_y_norm, original_size
        )

        for i, (box_px, box_lb, score) in enumerate(zip(boxes_pixels, boxes_letterbox, scores)):
            x_min, y_min, x_max, y_max = box_px

            detections.append({
                "confidence": float(score),
                "x_min": int(x_min),
                "y_min": int(y_min),
                "x_max": int(x_max),
                "y_max": int(y_max)
            })

            # Coords normalisées dans l'espace original
            orig_w, orig_h = original_size
            boxes_normalized.append([
                float(x_min) / orig_w,
                float(y_min) / orig_h,
                float(x_max) / orig_w,
                float(y_max) / orig_h
            ])

    return {
        "nb_plates": len(detections),
        "detections": detections,
        "boxes_normalized": boxes_normalized,
        "scores": [float(s) for s in scores]
    }


def predict_from_bytes(
    image_bytes: bytes,
    score_threshold: float = 0.3,
    nms_threshold: float = 0.4
) -> Tuple[Dict, Image.Image]:
    """
    Prédit depuis des bytes d'image.

    Returns:
        (predictions_dict, original_image)
    """
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    predictions = predict_single_image(image, score_threshold, nms_threshold)
    return predictions, image


# ============================================================================
# ANNOTATION
# ============================================================================
def annotate_image(
    image: Image.Image,
    detections: List[Dict],
    box_color: str = "#00FF00",
    text_color: str = "#FFFFFF",
    box_width: int = 3
) -> Image.Image:
    """
    Annote une image avec les bounding boxes détectées.

    Args:
        image: Image PIL originale
        detections: Liste des détections
        box_color: Couleur des boxes
        text_color: Couleur du texte
        box_width: Épaisseur des lignes

    Returns:
        Image annotée
    """
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except (IOError, OSError):
        font = ImageFont.load_default()

    for i, det in enumerate(detections):
        x_min = det['x_min']
        y_min = det['y_min']
        x_max = det['x_max']
        y_max = det['y_max']
        confidence = det['confidence']

        # Dessiner la box
        draw.rectangle([x_min, y_min, x_max, y_max], outline=box_color, width=box_width)

        # Label
        label = f"Plate {i+1}: {confidence:.2f}"

        # Background du texte
        text_bbox = draw.textbbox((x_min, y_min - 20), label, font=font)
        draw.rectangle(text_bbox, fill=box_color)
        draw.text((x_min, y_min - 20), label, fill=text_color, font=font)

    return annotated


# ============================================================================
# BATCH INFERENCE (sans Spark - pour petits batches)
# ============================================================================
@torch.no_grad()
def predict_batch_simple(
    images: List[Image.Image],
    score_threshold: float = 0.3,
    nms_threshold: float = 0.4
) -> List[Dict]:
    """
    Prédiction batch simple avec letterbox (sans Spark).
    Utilisé pour des petits batches (< 10 images).

    Args:
        images: Liste d'images PIL
        score_threshold: Seuil de confiance
        nms_threshold: Seuil NMS

    Returns:
        Liste de résultats pour chaque image
    """
    model = get_model()
    device = get_device()

    results = []

    # Prétraiter toutes les images avec letterbox
    tensors = []
    original_sizes = []
    letterbox_params = []  # (pad_x_norm, pad_y_norm)

    for img in images:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        original_sizes.append(img.size)

        # Letterbox + normalisation
        img_letterbox, scale, pad_x_norm, pad_y_norm = letterbox_image(img)
        tensors.append(NORMALIZE_TRANSFORM(img_letterbox))
        letterbox_params.append((pad_x_norm, pad_y_norm))

    # Batch inference
    batch_tensor = torch.stack(tensors).to(device)
    batch_results = predict(model, batch_tensor, score_threshold=score_threshold, nms_threshold=nms_threshold)

    # Formater les résultats avec inverse letterbox
    for i, (det, orig_size, lb_params) in enumerate(zip(batch_results, original_sizes, letterbox_params)):
        boxes_letterbox = det['boxes'].cpu().numpy()
        scores = det['scores'].cpu().numpy()
        pad_x_norm, pad_y_norm = lb_params

        detections = []
        if len(boxes_letterbox) > 0:
            # Inverser le letterbox
            boxes_pixels = inverse_letterbox_coords(
                boxes_letterbox, pad_x_norm, pad_y_norm, orig_size
            )

            for box_px, score in zip(boxes_pixels, scores):
                x_min, y_min, x_max, y_max = box_px
                detections.append({
                    "confidence": float(score),
                    "x_min": int(x_min),
                    "y_min": int(y_min),
                    "x_max": int(x_max),
                    "y_max": int(y_max)
                })

        results.append({
            "nb_plates": len(detections),
            "detections": detections,
            "scores": [float(s) for s in scores]
        })

    return results


# ============================================================================
# HISTORY MANAGEMENT (CSV)
# ============================================================================
def save_to_history(
    image_name: str,
    nb_plates: int,
    detections: List[Dict],
    status: str = "success"
):
    """
    Sauvegarde une prédiction dans l'historique CSV.

    Args:
        image_name: Nom du fichier image
        nb_plates: Nombre de plaques détectées
        detections: Liste des détections
        status: Statut de la prédiction
    """
    history_path = HISTORY_CSV_PATH

    # Créer le fichier avec header si inexistant
    if not history_path.exists():
        with open(history_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'image_name', 'nb_plates', 'detections', 'status'])

    # Ajouter l'entrée
    with open(history_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            image_name,
            nb_plates,
            str(detections),
            status
        ])


def load_history(limit: int = 30) -> List[Dict]:
    """
    Charge l'historique depuis le CSV.

    Args:
        limit: Nombre max d'entrées à charger

    Returns:
        Liste des entrées (les plus récentes en premier)
    """
    history_path = HISTORY_CSV_PATH

    if not history_path.exists():
        return []

    entries = []
    with open(history_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append({
                "timestamp": row['timestamp'],
                "image_name": row['image_name'],
                "nb_plates": int(row['nb_plates']),
                "detections": row['detections'],
                "status": row['status']
            })

    # Retourner les plus récentes en premier
    return list(reversed(entries[-limit:]))


def clear_history():
    """Efface l'historique"""
    if HISTORY_CSV_PATH.exists():
        HISTORY_CSV_PATH.unlink()


# ============================================================================
# TEST
# ============================================================================
if __name__ == "__main__":
    print("🧪 Testing inference module...")

    # Test model loading
    model = get_model()
    print(f"✅ Model loaded: {is_model_loaded()}")

    # Test avec une image factice
    test_image = Image.new('RGB', (640, 480), color='white')
    result = predict_single_image(test_image)
    print(f"✅ Prediction result: {result}")

    # Test historique
    save_to_history("test.jpg", 0, [], "test")
    history = load_history()
    print(f"✅ History entries: {len(history)}")

    print("\n✅ All tests passed!")

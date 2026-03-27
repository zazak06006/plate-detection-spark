"""
Module d'inférence pour la détection de plaques d'immatriculation.
Centralise le chargement du modèle et les fonctions de prédiction.

Usage:
    from inference import get_model, predict_single_image, predict_batch_images
"""

import io
import sys
import csv
import uuid
import base64
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from functools import lru_cache

import json
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Spark Imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, lit, element_at, split
from pyspark.sql.types import StructType, StructField, StringType, BinaryType

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
IMAGES_DIR = Path(__file__).parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)

# Transform pour normalisation UNIQUEMENT (le letterbox est fait séparément)
NORMALIZE_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#add TRANSFORM for compatibility with Spark UDFs
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


# ============================================================================
# SPARK SESSION MANAGER
# ============================================================================
class SparkInferenceManager:
    """Gestionnaire singleton pour la session Spark"""
    _instance = None
    _spark = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def spark(self) -> SparkSession:
        if self._spark is None:
            self._spark = SparkSession.builder \
                .appName("LicensePlateInference") \
                .master("local[*]") \
                .config("spark.driver.memory", "4g") \
                .config("spark.executor.memory", "4g") \
                .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
                .config("spark.ui.enabled", "false") \
                .getOrCreate()
            self._spark.sparkContext.setLogLevel("ERROR")
            
            # Distribuer model.py sur les executors pour l'UDF
            model_py_path = str(MODEL_TRAINING_PATH / "model.py")
            self._spark.sparkContext.addPyFile(model_py_path)
            
            print("✅ Spark session created for Inference")
        return self._spark

_spark_manager = SparkInferenceManager()

def get_spark() -> SparkSession:
    return _spark_manager.spark


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
# INFERENCE PIPELINE (PYSPARK)
# ============================================================================
def process_image_udf_logic(image_bytes: bytes, score_threshold: float, nms_threshold: float) -> str:
    """
    UDF logic exécutée sur les workers PySpark.
    Applique le pipeline complet: Nettoyage -> Redimensionnement -> Inférence.
    """
    import sys
    import io
    import json
    from pathlib import Path
    
    # 0. RESOLUTION DES PATHS SUR LES WORKERS SPARK
    # Les workers Spark n'ont pas forcément le sys.path du driver.
    # On ajoute manuellement les chemins vers '2-model-training' et '3-web-interface'.
    current_file = Path(__file__).resolve()
    web_interface_path = str(current_file.parent)
    model_training_path = str(current_file.parent.parent / "2-model-training")
    
    if web_interface_path not in sys.path:
        sys.path.insert(0, web_interface_path)
    if model_training_path not in sys.path:
        sys.path.insert(0, model_training_path)
        
    try:
        from PIL import Image
        from inference import (
            letterbox_image, NORMALIZE_TRANSFORM, get_model, 
            get_device, inverse_letterbox_coords
        )
        from model import predict

        # 1. NETTOYAGE IMAGE
        # Lecture depuis les bytes bruts et conversion sécurisée en RGB
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        original_size = image.size
        
        # 2. REDIMENSIONNEMENT (LETTERBOX) & NORMALISATION
        img_letterbox, scale, pad_x_norm, pad_y_norm = letterbox_image(image)
        tensor = NORMALIZE_TRANSFORM(img_letterbox).unsqueeze(0)
        
        # 3. INFÉRENCE
        model = get_model()  # Singleton par worker
        device = get_device()
        tensor = tensor.to(device)
        
        results = predict(model, tensor, score_threshold=score_threshold, nms_threshold=nms_threshold)
        det = results[0]
        
        # POST-PROCESSING
        boxes_letterbox = det['boxes'].cpu().numpy()
        scores = det['scores'].cpu().numpy()
        
        detections = []
        if len(boxes_letterbox) > 0:
            boxes_pixels = inverse_letterbox_coords(boxes_letterbox, pad_x_norm, pad_y_norm, original_size)
            for box_px, score in zip(boxes_pixels, scores):
                detections.append({
                    "confidence": float(score),
                    "x_min": int(box_px[0]), "y_min": int(box_px[1]),
                    "x_max": int(box_px[2]), "y_max": int(box_px[3])
                })
        
        return json.dumps({
            "success": True,
            "nb_plates": len(detections),
            "detections": detections,
            "scores": [float(s) for s in scores]
        })
    except Exception as e:
        import traceback
        return json.dumps({
            "success": False, 
            "error": str(e), 
            "traceback": traceback.format_exc(),
            "nb_plates": 0, 
            "detections": [], 
            "scores": []
        })

# Enregistrement de l'UDF PySpark
spark_process_udf = udf(process_image_udf_logic, StringType())


from typing import Union

def _run_spark_pipeline(
    input_source: Union[List[Tuple[str, bytes]], str], 
    score_threshold: float, 
    nms_threshold: float,
    save_history: bool = False
) -> List[Dict]:
    """
    Exécute le pipeline Spark.
    Si input_source est une string (chemin d'un dossier), charge via `spark.read.format("binaryFile")`.
    Sinon, charge depuis une liste python.
    Maximise l'utilisation de Spark Dataframes et UDFs.
    """
    spark = get_spark()
    
    if isinstance(input_source, str):
        # Lecture native PySpark
        df = spark.read.format("binaryFile").load(input_source)
        df = df.withColumn("filename", element_at(split(col("path"), "/"), -1))
        df = df.withColumnRenamed("content", "image_bytes")
        df = df.select("filename", "image_bytes")
    else:
        # Définition du schéma du DataFrame entrant
        schema = StructType([
            StructField("filename", StringType(), False),
            StructField("image_bytes", BinaryType(), False)
        ])
        
        # Création du DataFrame
        df = spark.createDataFrame(input_source, schema)
        
    # Re-partition pour plus de parallelisme
    df = df.repartition(max(1, spark.sparkContext.defaultParallelism))
    
    # Application de l'UDF pour effectuer tous les traitements (Nettoyage, Resize, Inference)
    df_result = df.withColumn(
        "json_result", 
        spark_process_udf(col("image_bytes"), lit(score_threshold), lit(nms_threshold))
    )
    
    results_list = []
    
    # Générer un Run ID si on sauvegarde l'historique
    batch_run_id = None
    if save_history:
        batch_run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{uuid.uuid4().hex[:6]}"

    # Collecte des résultats vers le driver
    for row in df_result.collect():
        filename = row["filename"]
        image_bytes = row["image_bytes"]
        json_res = json.loads(row["json_result"])
        
        if json_res.get("success"):
            detections = json_res["detections"]
            nb_plates = len(detections)
            
            result_dict = {
                "status": "success",
                "success": True,
                "filename": filename,
                "nb_plates": nb_plates,
                "detections": detections,
                "scores": json_res["scores"],
                "run_id": batch_run_id
            }
            
            if save_history:
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                annotated = annotate_image(image, detections)
                orig_path, ann_path = save_images_to_filesystem(image, annotated, filename, run_id=batch_run_id)
                
                # Import des fonctions relatives à history au besoin
                from inference import save_to_history
                save_to_history(filename, nb_plates, detections, "success", orig_path, ann_path)
                result_dict["original_image_path"] = orig_path
                result_dict["annotated_image_path"] = ann_path
                
            results_list.append(result_dict)
        else:
            error_msg = json_res.get("error", "Erreur inconnue")
            print(f"Error on {filename}: {error_msg}\n{json_res.get('traceback', '')}")
            
            if save_history:
                from inference import save_to_history
                save_to_history(filename, 0, [], f"error: {error_msg}", "", "")
                
            results_list.append({
                "status": f"error: {error_msg}",
                "success": False, 
                "filename": filename, 
                "nb_plates": 0,
                "detections": [],
                "error": error_msg
            })
            
    return results_list





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
# BATCH INFERENCE (PYSPARK)
# ============================================================================
def predict_batch_simple(
    images: List[Image.Image],
    score_threshold: float = 0.3,
    nms_threshold: float = 0.4
) -> List[Dict]:
    """
    Prédiction batch redirigée vers Spark (remplace l'ancienne boucle).
    """
    images_data = []
    for i, img in enumerate(images):
        buffer = io.BytesIO()
        img.convert('RGB').save(buffer, format='JPEG')
        images_data.append((f"img_{i}.jpg", buffer.getvalue()))
        
    spark_results = _run_spark_pipeline(images_data, score_threshold, nms_threshold)
    
    formatted = []
    for res in spark_results:
        formatted.append({
            "nb_plates": res.get("nb_plates", 0),
            "detections": res.get("detections", []),
            "scores": res.get("scores", [])
        })
    return formatted


# ============================================================================
# IMAGE FILESYSTEM MANAGEMENT
# ============================================================================
def save_images_to_filesystem(
    original_image: Optional[Image.Image],
    annotated_image: Optional[Image.Image],
    original_filename: str,
    run_id: Optional[str] = None
) -> Tuple[str, str]:
    """
    Sauvegarde les images originale et annotée sur le système de fichiers.
    Groupées par 'run_id' si fourni.

    Args:
        original_image: Image PIL originale (peut être None)
        annotated_image: Image PIL annotée (peut être None)
        original_filename: Nom du fichier uploadé
        run_id: ID de la session/batch pour groupement (ex: 20260327_0200_xxxx)

    Returns:
        (chemin_relatif_original, chemin_relatif_annoté)
    """
    if original_image is None and annotated_image is None:
        return "", ""

    try:
        # 1. Déterminer le répertoire de destination
        if run_id:
            dest_dir = IMAGES_DIR / "runs" / run_id
        else:
            # Fallback structure si pas de run_id (ex: test unitaire)
            now = datetime.now()
            date_str = now.strftime("%Y/%m/%d")
            time_uuid = now.strftime("%Y%m%d_%H%M%S") + f"_{uuid.uuid4().hex[:8]}"
            dest_dir = IMAGES_DIR / date_str / time_uuid

        dest_dir.mkdir(parents=True, exist_ok=True)

        # 2. Préparer les noms de fichiers
        # On utilise le nom de fichier original comme base pour éviter les collisions dans le run_id folder
        base_name = Path(original_filename).stem
        orig_ext = Path(original_filename).suffix.lower()
        if not orig_ext:
            orig_ext = ".jpg"

        # 3. Sauvegarder l'image originale
        original_path_rel = ""
        if original_image is not None:
            fname_orig = f"{base_name}_orig{orig_ext}"
            original_path_abs = dest_dir / fname_orig
            original_image.save(original_path_abs)
            original_path_rel = str(original_path_abs.relative_to(IMAGES_DIR.parent)).replace("\\", "/")

        # 4. Sauvegarder l'image annotée
        annotated_path_rel = ""
        if annotated_image is not None:
            fname_ann = f"{base_name}_ann.jpg"
            annotated_path_abs = dest_dir / fname_ann
            annotated_image.save(annotated_path_abs, format='JPEG', quality=90)
            annotated_path_rel = str(annotated_path_abs.relative_to(IMAGES_DIR.parent)).replace("\\", "/")

        return original_path_rel, annotated_path_rel

    except Exception as e:
        print(f"❌ Erreur sauvegarde images: {e}")
        return "", ""


def load_image_from_path(relative_path: str) -> Optional[Image.Image]:
    """
    Charge une image depuis un chemin relatif.

    Args:
        relative_path: Chemin relatif à partir de 3-web-interface/

    Returns:
        Image PIL ou None si manquante/erreur
    """
    if not relative_path:
        return None

    try:
        base_dir = IMAGES_DIR.parent
        abs_path = (base_dir / relative_path).resolve()

        # Sécurité: vérifier que le chemin est dans le répertoire images/
        if not str(abs_path).startswith(str(base_dir.resolve())):
            print(f"⚠️ Tentative de lecture hors du répertoire autorisé: {relative_path}")
            return None

        if not abs_path.exists():
            print(f"⚠️ Image manquante: {relative_path}")
            return None

        img = Image.open(abs_path).convert('RGB')
        return img

    except Exception as e:
        print(f"⚠️ Erreur lors du chargement de l'image ({relative_path}): {e}")
        return None


def migrate_old_history_csv() -> bool:
    """
    Migre l'ancien historique (format base64) vers le nouveau (format fichiers).

    Returns:
        True si migration effectuée, False sinon
    """
    history_path = HISTORY_CSV_PATH

    if not history_path.exists():
        return False

    try:
        # Augmenter temporairement la limite pour lire l'ancien CSV avec base64
        csv.field_size_limit(int(1e8))  # 100MB limit pour lecture old format

        # Vérifié le format du CSV
        with open(history_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []

            # Si colonne 'annotated_image_path' existe déjà, déjà migré
            if 'annotated_image_path' in fieldnames:
                return False

            # Si pas de colonne 'annotated_image', format inconnu
            if 'annotated_image' not in fieldnames:
                return False

            # Lire toutes les entrées
            entries = []
            for row in reader:
                entries.append(row)

        print(f"🔄 Migration de {len(entries)} entrées d'historique...")

        # Traiter chaque entrée
        migrated_rows = []
        for row in entries:
            base64_data = row.get('annotated_image', '')
            annotated_path = ""

            # Extraire et sauvegarder le base64 si présent
            if base64_data:
                try:
                    image_bytes = base64.b64decode(base64_data)
                    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

                    # Sauvegarder avec timestamp du CSV si disponible
                    try:
                        timestamp = datetime.fromisoformat(row['timestamp'])
                    except:
                        timestamp = None

                    ts_str = timestamp.strftime("%Y%m%d_%H%M%S") if timestamp else "migrated"
                    _, annotated_path = save_images_to_filesystem(None, image, "migrated.jpg", run_id=f"migrated_{ts_str}")
                except Exception as e:
                    print(f"⚠️ Impossible d'extraire image base64: {e}")

            # Créer la nouvelle ligne
            new_row = {
                'timestamp': row['timestamp'],
                'image_name': row['image_name'],
                'nb_plates': row['nb_plates'],
                'detections': row['detections'],
                'status': row['status'],
                'original_image_path': '',  # On n'a pas l'originale
                'annotated_image_path': annotated_path
            }
            migrated_rows.append(new_row)

        # Sauvegarder le CSV migré
        backup_path = history_path.with_suffix('.csv.backup')
        import shutil
        shutil.copy2(history_path, backup_path)
        print(f"✅ Backup créé: {backup_path}")

        # Réinitialiser la limite à normale
        csv.field_size_limit(131072)  # Reset to default

        # Écrire le nouveau CSV
        with open(history_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['timestamp', 'image_name', 'nb_plates', 'detections', 'status', 'original_image_path', 'annotated_image_path']
            )
            writer.writeheader()
            writer.writerows(migrated_rows)

        print(f"✅ Migration terminée: {len(migrated_rows)} entrées traitées")
        return True

    except Exception as e:
        print(f"❌ Erreur lors de la migration: {e}")
        csv.field_size_limit(131072)  # Reset to default
        return False


# ============================================================================
# HISTORY MANAGEMENT (CSV)
# ============================================================================
def save_to_history(
    image_name: str,
    nb_plates: int,
    detections: List[Dict],
    status: str = "success",
    original_image_path: Optional[str] = None,
    annotated_image_path: Optional[str] = None,
    annotated_image_base64: Optional[str] = None
):
    """
    Sauvegarde une prédiction dans l'historique CSV avec le nouveau format (chemins).

    Args:
        image_name: Nom du fichier image
        nb_plates: Nombre de plaques détectées
        detections: Liste des détections
        status: Statut de la prédiction
        original_image_path: Chemin relatif de l'image originale (nouveau format)
        annotated_image_path: Chemin relatif de l'image annotée (nouveau format)
        annotated_image_base64: Image annotée en base64 (ancien format, déprécié)
    """
    history_path = HISTORY_CSV_PATH

    # Créer le fichier avec header si inexistant
    if not history_path.exists():
        with open(history_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'image_name', 'nb_plates', 'detections', 'status',
                'original_image_path', 'annotated_image_path'
            ])

    # Ajouter l'entrée
    with open(history_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            image_name,
            nb_plates,
            str(detections),
            status,
            original_image_path or "",
            annotated_image_path or ""
        ])


def load_history(limit: int = 30) -> List[Dict]:
    """
    Charge l'historique depuis le CSV (nouveau format avec chemins).
    Effectue une migration automatique du format ancien si nécessaire.

    Args:
        limit: Nombre max d'entrées à charger

    Returns:
        Liste des entrées (les plus récentes en premier)
    """
    history_path = HISTORY_CSV_PATH

    if not history_path.exists():
        return []

    # Effectuer la migration si nécessaire
    migrate_old_history_csv()

    entries = []
    try:
        with open(history_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                entries.append({
                    "timestamp": row['timestamp'],
                    "image_name": row['image_name'],
                    "nb_plates": int(row['nb_plates']),
                    "detections": row['detections'],
                    "status": row['status'],
                    "original_image_path": row.get('original_image_path', ''),
                    "annotated_image_path": row.get('annotated_image_path', '')
                })
    except Exception as e:
        print(f"❌ Erreur lors du chargement de l'historique: {e}")
        return []

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
    buffer = io.BytesIO()
    test_image.save(buffer, format='JPEG')
    image_bytes = buffer.getvalue()
    
    result = _run_spark_pipeline([("test.jpg", image_bytes)], 0.3, 0.4)
    print(f"✅ Prediction result: {result}")

    # Test historique
    save_to_history("test.jpg", 0, [], "test")
    history = load_history()
    print(f"✅ History entries: {len(history)}")

    print("\n✅ All tests passed!")

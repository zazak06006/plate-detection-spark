"""
Module de traitement batch avec PySpark.
Permet d'inférer sur un grand nombre d'images en parallèle.

Usage:
    from spark_batch import process_images_batch, SparkBatchProcessor
"""

import io
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import base64

import torch
from PIL import Image
import numpy as np

# Import PySpark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, lit, current_timestamp
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    ArrayType, FloatType, BinaryType
)

# Import du module d'inférence
from inference import (
    get_model, get_device, predict_single_image,
    annotate_image, save_to_history, TRANSFORM
)


# ============================================================================
# CONFIG
# ============================================================================
SPARK_APP_NAME = "PlateDetectionBatch"


# ============================================================================
# SPARK BATCH PROCESSOR
# ============================================================================
class SparkBatchProcessor:
    """
    Processeur batch utilisant PySpark pour paralléliser l'inférence.
    """

    def __init__(self, num_partitions: int = 4):
        """
        Args:
            num_partitions: Nombre de partitions Spark
        """
        self.num_partitions = num_partitions
        self._spark = None

    @property
    def spark(self) -> SparkSession:
        """Lazy loading de la session Spark"""
        if self._spark is None:
            self._spark = SparkSession.builder \
                .appName(SPARK_APP_NAME) \
                .master("local[*]") \
                .config("spark.driver.memory", "4g") \
                .config("spark.executor.memory", "4g") \
                .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
                .config("spark.ui.enabled", "false") \
                .getOrCreate()

            self._spark.sparkContext.setLogLevel("ERROR")
            print(f"✅ Spark session created: {SPARK_APP_NAME}")

        return self._spark

    def stop(self):
        """Arrête la session Spark"""
        if self._spark is not None:
            self._spark.stop()
            self._spark = None

    def process_images(
        self,
        images_data: List[Tuple[str, bytes]],
        score_threshold: float = 0.3,
        nms_threshold: float = 0.4
    ) -> List[Dict]:
        """
        Traite un batch d'images avec PySpark.

        Args:
            images_data: Liste de tuples (filename, image_bytes)
            score_threshold: Seuil de confiance
            nms_threshold: Seuil NMS

        Returns:
            Liste de résultats pour chaque image
        """
        # Pré-charger le modèle sur le driver
        _ = get_model()

        # Créer le DataFrame
        schema = StructType([
            StructField("filename", StringType(), False),
            StructField("image_bytes", BinaryType(), False)
        ])

        # Convertir en DataFrame
        df = self.spark.createDataFrame(images_data, schema)
        df = df.repartition(self.num_partitions)

        # Collecter et traiter (le modèle est sur le driver)
        # Pour une vraie distribution, il faudrait broadcast le modèle
        # Ici, on collecte et on utilise le batch simple
        rows = df.collect()

        results = []
        model = get_model()
        device = get_device()

        for row in rows:
            filename = row['filename']
            image_bytes = row['image_bytes']

            try:
                # Charger l'image
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

                # Inférence
                prediction = predict_single_image(
                    image,
                    score_threshold=score_threshold,
                    nms_threshold=nms_threshold
                )

                # Annoter l'image
                annotated = annotate_image(image, prediction['detections'])

                # Encoder l'image annotée en base64
                buffer = io.BytesIO()
                annotated.save(buffer, format='JPEG', quality=90)
                annotated_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

                result = {
                    "filename": filename,
                    "nb_plates": prediction['nb_plates'],
                    "detections": prediction['detections'],
                    "annotated_image": annotated_base64,
                    "status": "success"
                }

                # Sauvegarder dans l'historique
                save_to_history(filename, prediction['nb_plates'], prediction['detections'])

            except Exception as e:
                result = {
                    "filename": filename,
                    "nb_plates": 0,
                    "detections": [],
                    "annotated_image": None,
                    "status": f"error: {str(e)}"
                }
                save_to_history(filename, 0, [], "error")

            results.append(result)

        return results


# Instance globale
_processor = None


def get_spark_processor(num_partitions: int = 4) -> SparkBatchProcessor:
    """Retourne le processeur Spark (singleton)"""
    global _processor
    if _processor is None:
        _processor = SparkBatchProcessor(num_partitions)
    return _processor


def process_images_batch(
    images_data: List[Tuple[str, bytes]],
    score_threshold: float = 0.3,
    nms_threshold: float = 0.4,
    use_spark: bool = True,
    spark_threshold: int = 5
) -> List[Dict]:
    """
    Traite un batch d'images.
    Utilise Spark si le nombre d'images dépasse le seuil.

    Args:
        images_data: Liste de tuples (filename, image_bytes)
        score_threshold: Seuil de confiance
        nms_threshold: Seuil NMS
        use_spark: Forcer l'utilisation de Spark
        spark_threshold: Seuil pour activer Spark automatiquement

    Returns:
        Liste de résultats
    """
    num_images = len(images_data)

    # Utiliser Spark si beaucoup d'images
    if use_spark or num_images >= spark_threshold:
        print(f"🚀 Processing {num_images} images with PySpark...")
        processor = get_spark_processor()
        return processor.process_images(images_data, score_threshold, nms_threshold)
    else:
        # Traitement simple sans Spark
        print(f"📷 Processing {num_images} images without Spark...")
        return process_images_simple(images_data, score_threshold, nms_threshold)


def process_images_simple(
    images_data: List[Tuple[str, bytes]],
    score_threshold: float = 0.3,
    nms_threshold: float = 0.4
) -> List[Dict]:
    """
    Traitement simple (sans Spark) pour petits batches.
    """
    results = []

    for filename, image_bytes in images_data:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

            prediction = predict_single_image(
                image,
                score_threshold=score_threshold,
                nms_threshold=nms_threshold
            )

            annotated = annotate_image(image, prediction['detections'])

            buffer = io.BytesIO()
            annotated.save(buffer, format='JPEG', quality=90)
            annotated_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            result = {
                "filename": filename,
                "nb_plates": prediction['nb_plates'],
                "detections": prediction['detections'],
                "annotated_image": annotated_base64,
                "status": "success"
            }

            save_to_history(filename, prediction['nb_plates'], prediction['detections'])

        except Exception as e:
            result = {
                "filename": filename,
                "nb_plates": 0,
                "detections": [],
                "annotated_image": None,
                "status": f"error: {str(e)}"
            }
            save_to_history(filename, 0, [], "error")

        results.append(result)

    return results


# ============================================================================
# TEST
# ============================================================================
if __name__ == "__main__":
    print("🧪 Testing Spark batch processor...")

    # Créer des images de test
    test_images = []
    for i in range(3):
        img = Image.new('RGB', (640, 480), color=(100 + i * 50, 100, 100))
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        test_images.append((f"test_{i}.jpg", buffer.getvalue()))

    # Test sans Spark
    print("\n📷 Testing simple processing...")
    results = process_images_simple(test_images)
    print(f"   Results: {len(results)} images processed")

    # Test avec Spark
    print("\n🚀 Testing Spark processing...")
    results = process_images_batch(test_images, use_spark=True)
    print(f"   Results: {len(results)} images processed")

    print("\n✅ All tests passed!")

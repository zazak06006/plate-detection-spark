"""
Microservice 1 — Prétraitement orienté SSD avec Apache Spark
===============================================================

Approche SSD (Single Shot MultiBox Detector) :
  - Contrairement à un CNN simple, un SSD analyse l'image ENTIÈRE.
  - On ne fait donc PLUS de crops (ni positifs, ni négatifs).
  - Les images sont lues en binaire, redimensionnées à 300x300 (standard SSD typique),
    puis encodées en base64 pour être stockées dans un CSV.
  - Les bounding boxes (cx, cy, w, h) associées à chaque image sont regroupées dans un tableau (encodé en JSON).
  - Le résultat final est exporté en format CSV.

Format du fichier Parquet produit :
  image_name | split_id | images | cls_targets | reg_targets | pos_mask
  - images      : Image 256x256 en format binaire (JPEG).
  - cls_targets : Liste (Array) des numéros de classes.
  - reg_targets : Liste (Array) des bboxes [cx, cy, w, h].
  - pos_mask    : Liste (Array) des masques d'objet (1.0).
"""

import io
import shutil
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, FloatType, BinaryType
from PIL import Image

# ─────────────────────────────────────────────────────────────────
# PARAMÈTRES GLOBAUX
# ─────────────────────────────────────────────────────────────────
DATASET    = "../license-plate-detection-dataset-10125-images"
IMG_SIZE   = 256   # Résolution du modèle SSD-CNN-256

OUTPUT_DIR = Path("data/processed") # Enregistrement dans le dossier courant
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# ─────────────────────────────────────────────────────────────────
# Démarrage Spark
# ─────────────────────────────────────────────────────────────────
spark = (
    SparkSession.builder
    .appName("LicensePlate-SSD-Preprocessing")
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    # Augmenter la taille limite si des images sont très grosses
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
print("SparkSession démarrée (Mode SSD, Export Parquet)\n")

# ─────────────────────────────────────────────────────────────────
# 1. Lecture des images au format binaire
# ─────────────────────────────────────────────────────────────────
print("Lecture des images avec binaryFile...")
df_images = (
    spark.read.format("binaryFile")
    .load([
        f"{DATASET}/train/images/*.jpg",
        f"{DATASET}/valid/images/*.jpg",
        f"{DATASET}/test/images/*.jpg"
    ])
    .withColumn("split_name", F.regexp_extract("path", r"/(train|valid|test)/", 1))
    .withColumn("image_name", F.regexp_extract("path", r"/([^/]+)\.jpg$", 1))
    .withColumn("split_id", 
                F.when(F.col("split_name") == "train", 0)
                 .when(F.col("split_name") == "valid", 1)
                 .otherwise(2))
    .select("image_name", "split_id", "split_name", F.col("content").alias("raw_content"))
)

# ─────────────────────────────────────────────────────────────────
# 2. Lecture, Parsing et Regroupement des Labels YOLO
# ─────────────────────────────────────────────────────────────────
print("Lecture et regroupement des labels YOLO...")
df_labels_raw = (
    spark.read.text([
        f"{DATASET}/train/labels/*.txt",
        f"{DATASET}/valid/labels/*.txt",
        f"{DATASET}/test/labels/*.txt",
    ])
    .withColumn("label_path", F.input_file_name())
    .withColumn("image_name", F.regexp_extract("label_path", r"/([^/]+)\.txt$", 1))
    .filter(F.col("value") != "")
)

df_labels_parsed = (
    df_labels_raw
    .withColumn("parts", F.split(F.col("value"), " "))
    .withColumn("class_id", F.col("parts")[0].cast(FloatType())) # On cast tout en float pour avoir un tableau homogène
    .withColumn("cx_norm",  F.col("parts")[1].cast(FloatType()))
    .withColumn("cy_norm",  F.col("parts")[2].cast(FloatType()))
    .withColumn("w_norm",   F.col("parts")[3].cast(FloatType()))
    .withColumn("h_norm",   F.col("parts")[4].cast(FloatType()))
    .drop("value", "parts", "label_path")
    .dropna()
)

# On groupe par image : séparation en cls_targets, reg_targets et pos_mask selon le format attendu
df_bboxes = (
    df_labels_parsed
    .withColumn("reg_target", F.array("cx_norm", "cy_norm", "w_norm", "h_norm"))
    .withColumn("pos_mask", F.lit(1.0).cast(FloatType()))
    .groupBy("image_name")
    .agg(
        F.collect_list("class_id").alias("cls_targets"),
        F.collect_list("reg_target").alias("reg_targets"),
        F.collect_list("pos_mask").alias("pos_mask")
    )
)

# ─────────────────────────────────────────────────────────────────
# 3. Jointure Images & Labels
# ─────────────────────────────────────────────────────────────────
print("Jointure entre images et annotations...")
df_joined = df_images.join(df_bboxes, on="image_name", how="left")

# Remplir les images sans labels (uniquement du fond) avec une liste vide
# Un SSD gère très bien les images sans objets (elles servent d'exemples négatifs)
df_joined = (
    df_joined
    .withColumn("cls_targets", F.when(F.col("cls_targets").isNull(), F.array()).otherwise(F.col("cls_targets")))
    .withColumn("reg_targets", F.when(F.col("reg_targets").isNull(), F.array()).otherwise(F.col("reg_targets")))
    .withColumn("pos_mask", F.when(F.col("pos_mask").isNull(), F.array()).otherwise(F.col("pos_mask")))
)

# ─────────────────────────────────────────────────────────────────
# 4. UDF : Redimensionnement Image à 300x300
# ─────────────────────────────────────────────────────────────────
@F.udf(BinaryType())
def resize_image(content):
    """
    UDF Spark : Prend le jpeg d'origine, le redimensionne en 300x300 standard, 
    et le renvoie en octets JPEG. L'avantage du format SSD est que les coordonnées 
    YOLO normalisées restent vraies même si le ratio change légèrement.
    """
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        out = io.BytesIO()
        img.save(out, format="JPEG")
        return out.getvalue()
    except Exception:
        return None

print(f"Redimensionnement des images en {IMG_SIZE}x{IMG_SIZE}...")
df_final = (
    df_joined
    .withColumn("images", resize_image("raw_content"))
    .drop("raw_content")
    .filter(F.col("images").isNotNull())
)

# ─────────────────────────────────────────────────────────────────
# 5. Export des données par split (Format Parquet)
# ─────────────────────────────────────────────────────────────────
print("Export du dataset SSD en Parquet (coalesce pour un fichier par split)...")

splits = {"train": 0, "valid": 1, "test": 2}
stats = {}

for split_name, split_id in splits.items():
    print(f"   Export de la partition : {split_name}...")
    df_split = df_final.filter(F.col("split_id") == split_id).drop("split_name", "split_id")
    
    tmp_dir = OUTPUT_DIR / f"tmp_{split_name}"
    output_file = OUTPUT_DIR / f"{split_name}.parquet"
    
    # Écriture parquet
    df_split.coalesce(1).write.mode("overwrite").parquet(str(tmp_dir))
    
    # Renommage du part-*.parquet → train.parquet / valid.parquet / test.parquet
    part_file = next(tmp_dir.glob("part-*.parquet"))
    shutil.move(str(part_file), str(output_file))
    shutil.rmtree(str(tmp_dir))
    
    # On force un décompte action() pour avoir les stats
    nb = spark.read.parquet(str(output_file)).count()
    stats[split_name] = nb
    print(f"   ✅ {split_name}.parquet  → {nb} images")

print(f"\n{'=' * 55}")
print(f"Prétraitement SSD terminé !")
print(f"{'=' * 55}")
print(f"   Dossier         : {OUTPUT_DIR.resolve()}/")
print(f"   Train images    : {stats['train']}")
print(f"   Valid images    : {stats['valid']}")
print(f"   Test images     : {stats['test']}")
print(f"   Format d'image  : {IMG_SIZE}×{IMG_SIZE} encodées en binaire (JPEG)")
print(f"   Format de table : [image_name | images | cls_targets | reg_targets | pos_mask]")
print(f"{'=' * 55}")

spark.stop()
print("\nSparkSession arrêtée.")
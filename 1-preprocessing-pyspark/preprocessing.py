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

Format du CSV produit :
  image_name | split_id | bboxes_json | image_base64
  - bboxes_json  : Chaine JSON contenant la liste des bboxes. Chaque bbox est [class_id, cx_norm, cy_norm, w_norm, h_norm].
  - image_base64 : Image 300x300 encodée en JPEG, puis en chaine Base64.
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
IMG_SIZE   = 300   # Résolution standard pour beaucoup de modèles SSD (ex: 300x300)

OUTPUT_DIR = Path("output_ssd_csv") # Nouveau dossier pour ne pas écraser l'ancien
OUTPUT_DIR.mkdir(exist_ok=True)

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
print("🚀 SparkSession démarrée (Mode SSD, Export CSV)\n")

# ─────────────────────────────────────────────────────────────────
# 1. Lecture des images au format binaire
# ─────────────────────────────────────────────────────────────────
print("📸 Lecture des images avec binaryFile...")
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
print("🏷️ Lecture et regroupement des labels YOLO...")
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

# On groupe par image : une image contient une liste (Array) de bounding boxes
df_bboxes = (
    df_labels_parsed
    .withColumn("bbox", F.array("class_id", "cx_norm", "cy_norm", "w_norm", "h_norm"))
    .groupBy("image_name")
    .agg(F.collect_list("bbox").alias("bboxes"))
)

# ─────────────────────────────────────────────────────────────────
# 3. Jointure Images & Labels
# ─────────────────────────────────────────────────────────────────
print("🔗 Jointure entre images et annotations...")
df_joined = df_images.join(df_bboxes, on="image_name", how="left")

# Remplir les images sans labels (uniquement du fond) avec une liste vide
# Un SSD gère très bien les images sans objets (elles servent d'exemples négatifs)
df_joined = df_joined.withColumn(
    "bboxes", 
    F.when(F.col("bboxes").isNull(), F.array()).otherwise(F.col("bboxes"))
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

print(f"🖼️ Redimensionnement des images en {IMG_SIZE}x{IMG_SIZE}...")
df_final = (
    df_joined
    .withColumn("image_bytes", resize_image("raw_content"))
    # Encodage en Base64 pour la compatibilité CSV et JSON pour les bboxes
    .withColumn("image_base64", F.base64(F.col("image_bytes")))
    .withColumn("bboxes_json", F.to_json(F.col("bboxes")))
    .drop("raw_content", "image_bytes", "bboxes")
    .filter(F.col("image_base64").isNotNull())
)

# ─────────────────────────────────────────────────────────────────
# 5. Export des données par split (Format CSV)
# ─────────────────────────────────────────────────────────────────
print("💾 Export du dataset SSD en CSV (coalesce pour n'avoir qu'un fichier)...")

splits = {"train": 0, "valid": 1, "test": 2}
stats = {}

for split_name, split_id in splits.items():
    print(f"   Export de la partition : {split_name}...")
    df_split = df_final.filter(F.col("split_id") == split_id).drop("split_name", "split_id")
    
    tmp_dir = OUTPUT_DIR / f"tmp_{split_name}"
    output_csv = OUTPUT_DIR / f"{split_name}.csv"
    
    # Écriture csv
    # L'option escape permet de gérer proprement les quotes du JSON dans le CSV
    df_split.coalesce(1).write.mode("overwrite").option("escape", '"').csv(str(tmp_dir), header=True)
    
    # Renommage du part-*.csv → train.csv / valid.csv / test.csv
    part_file = next(tmp_dir.glob("part-*.csv"))
    shutil.move(str(part_file), str(output_csv))
    shutil.rmtree(str(tmp_dir))
    
    # On force un décompte action() pour avoir les stats
    nb = spark.read.option("escape", '"').csv(str(output_csv), header=True).count()
    stats[split_name] = nb
    print(f"   ✅ {split_name}.csv  → {nb} images")

print(f"\n{'=' * 55}")
print(f"✅ Prétraitement SSD terminé !")
print(f"{'=' * 55}")
print(f"   Dossier         : {OUTPUT_DIR.resolve()}/")
print(f"   Train images    : {stats['train']}")
print(f"   Valid images    : {stats['valid']}")
print(f"   Test images     : {stats['test']}")
print(f"   Format d'image  : {IMG_SIZE}×{IMG_SIZE} encodées en JPEG Base64")
print(f"   Format de table : [image_name | bboxes_json | image_base64]")
print(f"{'=' * 55}")

spark.stop()
print("\n🏁 SparkSession arrêtée.")
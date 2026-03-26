"""
Microservice 1 — Prétraitement SSD avec Apache Spark (Version FIX Windows)
"""

import os
import io
import glob
import shutil
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, BinaryType
from PIL import Image

# ─────────────────────────────────────────────────────────────────
# FIX WINDOWS (évite crash Hadoop)
# ─────────────────────────────────────────────────────────────────
os.environ["HADOOP_OPTS"] = "-Djava.library.path="

# ─────────────────────────────────────────────────────────────────
# PARAMÈTRES
# ─────────────────────────────────────────────────────────────────
DATASET = "../license-plate-detection-dataset-10125-images"
IMG_SIZE = 256

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# ─────────────────────────────────────────────────────────────────
# SPARK SESSION
# ─────────────────────────────────────────────────────────────────
spark = (
    SparkSession.builder
    .appName("LicensePlate-SSD-Preprocessing")
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")
print("✅ SparkSession démarrée\n")

# ─────────────────────────────────────────────────────────────────
# 1. LECTURE IMAGES (FIX glob Windows)
# ─────────────────────────────────────────────────────────────────
print("📷 Lecture des images...")

image_files = (
    glob.glob(f"{DATASET}/train/images/*.jpg") +
    glob.glob(f"{DATASET}/valid/images/*.jpg") +
    glob.glob(f"{DATASET}/test/images/*.jpg")
)

df_images = (
    spark.read.format("binaryFile")
    .load(image_files)
    .withColumn("split_name", F.regexp_extract("path", r"/(train|valid|test)/", 1))
    .withColumn("image_name", F.regexp_extract("path", r"/([^/]+)\.jpg$", 1))
    .withColumn("split_id",
        F.when(F.col("split_name") == "train", 0)
         .when(F.col("split_name") == "valid", 1)
         .otherwise(2))
    .select("image_name", "split_id", "split_name", F.col("content").alias("raw_content"))
)

print(f"➡️ {df_images.count()} images chargées")

# ─────────────────────────────────────────────────────────────────
# 2. LECTURE LABELS (FIX glob)
# ─────────────────────────────────────────────────────────────────
print("📝 Lecture des labels...")

label_files = (
    glob.glob(f"{DATASET}/train/labels/*.txt") +
    glob.glob(f"{DATASET}/valid/labels/*.txt") +
    glob.glob(f"{DATASET}/test/labels/*.txt")
)

df_labels_raw = (
    spark.read.text(label_files)
    .withColumn("label_path", F.input_file_name())
    .withColumn("image_name", F.regexp_extract("label_path", r"/([^/]+)\.txt$", 1))
    .filter(F.col("value") != "")
)

df_labels_parsed = (
    df_labels_raw
    .withColumn("parts", F.split("value", " "))
    .withColumn("class_id", F.col("parts")[0].cast(FloatType()))
    .withColumn("cx", F.col("parts")[1].cast(FloatType()))
    .withColumn("cy", F.col("parts")[2].cast(FloatType()))
    .withColumn("w", F.col("parts")[3].cast(FloatType()))
    .withColumn("h", F.col("parts")[4].cast(FloatType()))
    .drop("value", "parts", "label_path")
    .dropna()
)

df_bboxes = (
    df_labels_parsed
    .withColumn("reg_target", F.array("cx", "cy", "w", "h"))
    .withColumn("pos_mask", F.lit(1.0))
    .groupBy("image_name")
    .agg(
        F.collect_list("class_id").alias("cls_targets"),
        F.collect_list("reg_target").alias("reg_targets"),
        F.collect_list("pos_mask").alias("pos_mask")
    )
)

# ─────────────────────────────────────────────────────────────────
# 3. JOIN
# ─────────────────────────────────────────────────────────────────
print("🔗 Jointure images + labels...")

df = df_images.join(df_bboxes, on="image_name", how="left")

df = (
    df
    .withColumn("cls_targets", F.when(F.col("cls_targets").isNull(), F.array()).otherwise(F.col("cls_targets")))
    .withColumn("reg_targets", F.when(F.col("reg_targets").isNull(), F.array()).otherwise(F.col("reg_targets")))
    .withColumn("pos_mask", F.when(F.col("pos_mask").isNull(), F.array()).otherwise(F.col("pos_mask")))
)

# ─────────────────────────────────────────────────────────────────
# 4. UDF Resize
# ─────────────────────────────────────────────────────────────────
@F.udf(BinaryType())
def resize_image(content):
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        out = io.BytesIO()
        img.save(out, format="JPEG")
        return out.getvalue()
    except:
        return None

print("🧠 Redimensionnement...")

df = (
    df
    .withColumn("images", resize_image("raw_content"))
    .drop("raw_content")
    .filter(F.col("images").isNotNull())
)

# ─────────────────────────────────────────────────────────────────
# 5. EXPORT PARQUET
# ─────────────────────────────────────────────────────────────────
print("💾 Export Parquet...")

splits = {"train": 0, "valid": 1, "test": 2}

for name, sid in splits.items():
    print(f"➡️ Export {name}")

    tmp = OUTPUT_DIR / f"tmp_{name}"
    out = OUTPUT_DIR / f"{name}.parquet"

    df_split = df.filter(F.col("split_id") == sid).drop("split_name", "split_id")

    df_split.coalesce(1).write.mode("overwrite").parquet(str(tmp))

    part = next(tmp.glob("part-*.parquet"))
    shutil.move(str(part), str(out))
    shutil.rmtree(tmp)

    print(f"✅ {name}.parquet OK")

print("\n🎉 DONE")

spark.stop()
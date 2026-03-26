"""Data loading from dataset"""
import glob
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
from config import DATASET

def load_images(spark):
    """Load images from dataset"""
    print("Loading images...")

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

    print(f"{df_images.count()} images loaded")
    return df_images

def load_labels(spark):
    """Load and parse labels from dataset"""
    print("Loading labels...")

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

    return df_labels_parsed

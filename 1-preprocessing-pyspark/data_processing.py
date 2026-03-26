"""Data processing and transformations"""
from pyspark.sql import functions as F
from image_utils import resize_image

def build_bboxes(df_labels_parsed):
    """Aggregate bounding boxes by image"""
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
    return df_bboxes

def join_data(df_images, df_bboxes):
    """Join images with bounding boxes"""
    print("Joining images and labels...")

    df = df_images.join(df_bboxes, on="image_name", how="left")

    df = (
        df
        .withColumn("cls_targets", F.when(F.col("cls_targets").isNull(), F.array()).otherwise(F.col("cls_targets")))
        .withColumn("reg_targets", F.when(F.col("reg_targets").isNull(), F.array()).otherwise(F.col("reg_targets")))
        .withColumn("pos_mask", F.when(F.col("pos_mask").isNull(), F.array()).otherwise(F.col("pos_mask")))
    )

    return df

def resize_images(df):
    """Resize images and filter errors"""
    print("Resizing images...")

    df = (
        df
        .withColumn("images", resize_image("raw_content"))
        .drop("raw_content")
        .filter(F.col("images").isNotNull())
    )

    return df

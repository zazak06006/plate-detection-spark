"""Data processing and transformations"""
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, FloatType
from image_utils import letterbox_resize_image


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


# UDF pour transformer les labels après letterbox
@F.udf(ArrayType(ArrayType(FloatType())))
def transform_labels_letterbox(reg_targets, scale, pad_x, pad_y):
    """
    Transforme les coordonnées des bounding boxes pour correspondre
    à l'image letterboxée.

    Args:
        reg_targets: Liste de [cx, cy, w, h] normalisés [0,1] par rapport à l'image originale
        scale: Facteur d'échelle appliqué lors du letterbox (ex: 0.8 si image réduite à 80%)
        pad_x: Padding horizontal normalisé [0,1] (ex: 0.1 si 10% de padding)
        pad_y: Padding vertical normalisé [0,1]

    Returns:
        Liste de [cx_new, cy_new, w_new, h_new] normalisés par rapport à l'image letterboxée 256x256

    Formula (pour coordonnées normalisées):
        - La région de l'image dans le letterbox occupe:
          - largeur: (1 - 2*pad_x) = new_w/IMG_SIZE
          - hauteur: (1 - 2*pad_y) = new_h/IMG_SIZE
        - cx_new = cx_orig * (1 - 2*pad_x) + pad_x
        - cy_new = cy_orig * (1 - 2*pad_y) + pad_y
        - w_new = w_orig * (1 - 2*pad_x)
        - h_new = h_orig * (1 - 2*pad_y)
    """
    if reg_targets is None or scale is None:
        return []

    transformed = []
    # La région de l'image dans le letterbox
    img_region_w = 1.0 - 2.0 * pad_x  # Ex: si pad_x=0.1, img_region_w=0.8
    img_region_h = 1.0 - 2.0 * pad_y

    for bbox in reg_targets:
        if bbox is None or len(bbox) < 4:
            continue

        cx_orig, cy_orig, w_orig, h_orig = bbox[0], bbox[1], bbox[2], bbox[3]

        # Transformer les coordonnées
        cx_new = cx_orig * img_region_w + pad_x
        cy_new = cy_orig * img_region_h + pad_y
        w_new = w_orig * img_region_w
        h_new = h_orig * img_region_h

        transformed.append([float(cx_new), float(cy_new), float(w_new), float(h_new)])

    return transformed


def resize_images(df):
    """
    Letterbox resize images and transform labels accordingly.
    Preserves aspect ratio with black padding.
    """
    print("Letterbox resizing images...")

    # Appliquer le letterbox resize (retourne struct avec image + infos transformation)
    df = df.withColumn("letterbox_result", letterbox_resize_image("raw_content"))

    # Extraire l'image et les paramètres de transformation
    df = (
        df
        .withColumn("images", F.col("letterbox_result.image"))
        .withColumn("lb_scale", F.col("letterbox_result.scale"))
        .withColumn("lb_pad_x", F.col("letterbox_result.pad_x"))
        .withColumn("lb_pad_y", F.col("letterbox_result.pad_y"))
        .drop("raw_content", "letterbox_result")
    )

    # Filtrer les erreurs de resize
    df = df.filter(F.col("images").isNotNull())

    # Transformer les labels pour correspondre au letterbox
    df = df.withColumn(
        "reg_targets",
        transform_labels_letterbox(
            F.col("reg_targets"),
            F.col("lb_scale"),
            F.col("lb_pad_x"),
            F.col("lb_pad_y")
        )
    )

    # Supprimer les colonnes temporaires (optionnel, garder pour debug)
    df = df.drop("lb_scale", "lb_pad_x", "lb_pad_y")

    return df

"""
Microservice 1 — Prétraitement SSD avec Apache Spark (Version FIX Windows)
Script principal d'orchestration
"""

from spark_setup import create_spark_session
from data_loading import load_images, load_labels
from data_processing import build_bboxes, join_data, resize_images
from export import export_splits

def main():
    """Pipeline principal de prétraitement"""

    # 1. Initialiser Spark
    spark = create_spark_session()

    # 2. Charger les données
    df_images = load_images(spark)
    df_labels_parsed = load_labels(spark)

    # 3. Traiter les labels
    df_bboxes = build_bboxes(df_labels_parsed)

    # 4. Joindre images + labels
    df = join_data(df_images, df_bboxes)

    # 5. Redimensionner les images
    df = resize_images(df)

    # 6. Exporter les splits
    export_splits(df)

    print("\n🎉 DONE")

    spark.stop()

if __name__ == "__main__":
    main()

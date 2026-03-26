"""Configuration and parameters"""
from pathlib import Path

DATASET = "../license-plate-detection-dataset-10125-images"
IMG_SIZE = 256

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

SPARK_CONFIG = {
    "appName": "LicensePlate-SSD-Preprocessing",
    "master": "local[*]",
    "driver_memory": "4g"
}

SPLITS = {"train": 0, "valid": 1, "test": 2}

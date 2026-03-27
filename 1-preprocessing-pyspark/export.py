"""Parquet data export"""
import shutil
from pyspark.sql import functions as F
from config import OUTPUT_DIR, SPLITS

def export_splits(df):
    """Export data by split (train/valid/test) as Parquet"""
    print("Exporting to Parquet...")

    for name, sid in SPLITS.items():
        print(f"Exporting {name}...")

        tmp = OUTPUT_DIR / f"tmp_{name}"
        out = OUTPUT_DIR / f"{name}.parquet"

        df_split = df.filter(F.col("split_id") == sid).drop("split_name", "split_id")

        df_split.coalesce(1).write.mode("overwrite").parquet(str(tmp))

        part = next(tmp.glob("part-*.parquet"))
        shutil.move(str(part), str(out))
        shutil.rmtree(tmp)

        print(f"{name}.parquet saved")

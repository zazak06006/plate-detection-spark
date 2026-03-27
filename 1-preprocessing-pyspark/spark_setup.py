"""SparkSession initialization"""
import os
from pyspark.sql import SparkSession
from config import SPARK_CONFIG

os.environ["HADOOP_OPTS"] = "-Djava.library.path="

def create_spark_session():
    """Create and configure SparkSession"""
    spark = (
        SparkSession.builder
        .appName(SPARK_CONFIG["appName"])
        .master(SPARK_CONFIG["master"])
        .config("spark.driver.memory", SPARK_CONFIG["driver_memory"])
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    print("SparkSession started\n")

    return spark

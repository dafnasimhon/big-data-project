
from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession

from config import settings


def load_raw_dataset(spark: SparkSession, path: str | None = None) -> DataFrame:
    return (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .option("multiLine", True)
        .option("escape", '"')
        .csv(path or settings.DATASET_PATH)
    )

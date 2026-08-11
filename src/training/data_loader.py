"""Loads the raw Stack Overflow survey CSV as a Spark DataFrame."""

from pyspark.sql import DataFrame, SparkSession

from config import settings


def load_raw_dataset(spark: SparkSession, path: str | None = None) -> DataFrame:
    """Read the raw survey CSV (quoted, multi-line fields) from DATASET_PATH."""
    return (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .option("multiLine", True)
        .option("escape", '"')
        .csv(path or settings.DATASET_PATH)
    )

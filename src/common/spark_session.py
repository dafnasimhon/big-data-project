"""Builds the project's SparkSession from config/settings.py (PLAN.md rule 5)."""

import pyspark
from pyspark.sql import SparkSession

from config import settings

# Matches the Kafka connector version to the installed PySpark version, since the VM's
# Spark install (not this requirements.txt) determines the actual Spark version.
_KAFKA_PACKAGE = f"org.apache.spark:spark-sql-kafka-0-10_2.12:{pyspark.__version__}"


def get_spark_session(app_name: str | None = None, with_kafka: bool = False) -> SparkSession:
    """Build (or fetch) the SparkSession for SPARK_MASTER_URL (local[*] or a standalone cluster).

    Pass with_kafka=True for jobs that read/write Kafka via Structured Streaming
    (src/streaming/*, src/producers/*); plain batch jobs (exploration, training) can
    leave it False.
    """
    builder = (
        SparkSession.builder.appName(app_name or settings.SPARK_APP_NAME)
        .master(settings.SPARK_MASTER_URL)
    )

    if with_kafka:
        builder = builder.config("spark.jars.packages", _KAFKA_PACKAGE)

    return builder.getOrCreate()

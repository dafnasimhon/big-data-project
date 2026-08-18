"""Real-time salary prediction stream (PLAN.md §16).

Reads `salary_requests` from Kafka, applies the saved `models/best_salary_model`
`PipelineModel`, reverses the log1p target transform, and publishes predictions to
`salary_predictions`. Requests missing a `request_id` (the one field the dashboard is
guaranteed to always send, per §17 - "generate a UUID per request") go to
`salary_dead_letter` instead of being force-predicted on.

Run with:

    python -m src.streaming.prediction_stream

Ported from `notebooks/04_Spark_Streaming_Prediction.ipynb` (PLAN.md §23), which proved
the mechanics work end-to-end on the VM but had gaps this version fixes:
  - Its request schema (`LanguageCount`/`DatabaseCount`/`PlatformCount` scalar counts,
    `Employment` as a plain single string) matched the notebook's own superseded feature
    design, not the model this repo actually trains now —
    `src/training/feature_pipeline.py` expects raw `LanguageHaveWorkedWith`/
    `DatabaseHaveWorkedWith`/`PlatformHaveWorkedWith` strings and treats `Employment` as
    multi-value too (§23 Known Issue #14, the real-data Employment-cardinality fix).
    This version uses `src/common/schemas.py: SALARY_REQUEST_SCHEMA`, which already
    matches `CANDIDATE_INPUT_FEATURES` - the same columns the trained pipeline expects.
  - Hardcoded `localhost:9092` and a Spark-generated temp checkpoint directory -> both
    now come from `config/settings.py`, consistent with the rest of the project.
  - No dead-letter handling for malformed/incomplete requests (§16 step 8) -> added.
  - `model_name`/`model_version` were hardcoded strings -> now read from the actual
    saved `model_metadata.json` (`selected_model`, `trained_at`), so this doesn't go
    stale the next time a different model wins.
  - Non-required fields (anything but `request_id`) missing from a request are
    `'Unknown'`-filled the same way training data handles missing values (§10.6),
    rather than being silently dropped or crashing `RegexTokenizer` on a null input.
"""

from __future__ import annotations

import json
import signal

from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.streaming import StreamingQuery

from config import settings
from src.common.feature_config import MULTI_VALUE_COLUMNS, SINGLE_VALUE_CATEGORICAL_COLUMNS
from src.common.logging_config import get_logger
from src.common.schemas import SALARY_REQUEST_SCHEMA
from src.common.spark_session import get_spark_session
from src.common.spark_utils import reverse_log1p_predictions, to_kafka_rows

logger = get_logger(__name__)

STRING_FEATURE_COLUMNS = SINGLE_VALUE_CATEGORICAL_COLUMNS + MULTI_VALUE_COLUMNS


def load_model_metadata(path: str | None = None) -> dict:
    with open(path or settings.MODEL_METADATA_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def parse_requests(raw_stream: DataFrame) -> DataFrame:
    """§16 step 3: parse Kafka records against the explicit request schema."""
    return (
        raw_stream.select(
            F.col("value").cast("string").alias("raw_value"),
            F.col("timestamp").alias("received_at"),
        )
        .withColumn("request", F.from_json(F.col("raw_value"), SALARY_REQUEST_SCHEMA))
        .select("raw_value", "received_at", "request.*")
    )


def split_valid_and_dead_letters(parsed: DataFrame) -> tuple[DataFrame, DataFrame]:
    """§16 step 4: validate required fields, preserve `request_id`.

    `request_id` is the only field treated as strictly required (missing/unparseable ->
    dead letter). Every other feature field is `'Unknown'`-filled if missing, mirroring
    how `data_cleaning.py` step 6 handles missing training data, rather than rejecting an
    otherwise-usable request or crashing `RegexTokenizer` on a null string column.
    """
    valid = parsed.filter(F.col("request_id").isNotNull())
    valid = valid.fillna("Unknown", subset=STRING_FEATURE_COLUMNS)
    valid = valid.withColumnRenamed("YearsCodePro", "YearsCodeProNumeric")

    dead_letters = parsed.filter(F.col("request_id").isNull()).select(
        F.lit(settings.KAFKA_REQUEST_TOPIC).alias("source_topic"),
        F.col("raw_value"),
        F.lit("missing or unparseable request_id").alias("error_reason"),
        F.col("received_at").cast("string").alias("received_at"),
    )
    return valid, dead_letters


def build_predictions(
    valid_requests: DataFrame, model: PipelineModel, model_name: str, model_version: str
) -> DataFrame:
    """§16 steps 5-7: apply the fitted pipeline, reverse log1p, build the response."""
    scored = model.transform(valid_requests)
    scored = reverse_log1p_predictions(scored, log_prediction_col="log_prediction", output_col="prediction")

    return scored.select(
        F.col("request_id"),
        F.round(F.col("prediction"), 2).alias("prediction"),
        F.lit("annual salary").alias("target_unit"),
        F.lit(model_name).alias("model_name"),
        F.lit(model_version).alias("model_version"),
        F.date_format(F.current_timestamp(), "yyyy-MM-dd'T'HH:mm:ss'Z'").alias("processed_at"),
        F.lit("success").alias("status"),
    )


def build_streams(spark: SparkSession) -> tuple[StreamingQuery, StreamingQuery]:
    """Builds and starts both streaming queries (predictions + dead letters) and
    returns them immediately, without blocking. Shared by the blocking CLI entry point
    (`run_prediction_stream`, below) and interactive/notebook use, where the caller
    wants to start the streams, do other things (publish test requests, inspect
    results), and explicitly `.stop()` them when done rather than block forever.
    """
    metadata = load_model_metadata()
    model = PipelineModel.load(settings.MODEL_PATH)

    logger.info(
        "Loaded model %s (trained_at=%s) from %s",
        metadata["selected_model"], metadata["trained_at"], settings.MODEL_PATH,
    )

    raw_stream = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", settings.KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", settings.KAFKA_REQUEST_TOPIC)
        .option("startingOffsets", "latest")
        .load()
    )

    parsed = parse_requests(raw_stream)
    valid_requests, dead_letters = split_valid_and_dead_letters(parsed)
    predictions = build_predictions(
        valid_requests, model, metadata["selected_model"], metadata["trained_at"]
    )

    prediction_query = (
        to_kafka_rows(predictions, key_col="request_id")
        .writeStream.format("kafka")
        .option("kafka.bootstrap.servers", settings.KAFKA_BOOTSTRAP_SERVERS)
        .option("topic", settings.KAFKA_PREDICTION_TOPIC)
        .option("checkpointLocation", settings.PREDICTION_CHECKPOINT_PATH)
        .outputMode("append")
        .start()
    )

    dead_letter_query = (
        to_kafka_rows(dead_letters, key_col=None)
        .writeStream.format("kafka")
        .option("kafka.bootstrap.servers", settings.KAFKA_BOOTSTRAP_SERVERS)
        .option("topic", settings.KAFKA_DEAD_LETTER_TOPIC)
        .option("checkpointLocation", settings.DEAD_LETTER_CHECKPOINT_PATH)
        .outputMode("append")
        .start()
    )

    logger.info(
        "Prediction stream running: %s -> %s (dead letters -> %s).",
        settings.KAFKA_REQUEST_TOPIC, settings.KAFKA_PREDICTION_TOPIC, settings.KAFKA_DEAD_LETTER_TOPIC,
    )
    return prediction_query, dead_letter_query


def run_prediction_stream() -> None:
    """Blocking CLI entry point: `python -m src.streaming.prediction_stream`."""
    spark = get_spark_session(app_name="SalaryPredictionStream", with_kafka=True)
    prediction_query, dead_letter_query = build_streams(spark)
    logger.info("Ctrl+C to stop.")

    def _graceful_shutdown(signum, frame):  # noqa: ARG001 - signal handler signature
        logger.info("Received shutdown signal, stopping streaming queries...")
        prediction_query.stop()
        dead_letter_query.stop()

    signal.signal(signal.SIGTERM, _graceful_shutdown)
    signal.signal(signal.SIGINT, _graceful_shutdown)

    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    run_prediction_stream()

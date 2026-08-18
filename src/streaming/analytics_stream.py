"""Real-time developer-events analytics stream (PLAN.md §15).

Reads `developer_events` (published by `src/producers/dataset_producer.py`, Phase 6),
and computes three windowed aggregates, all published as tagged JSON records to
`salary_analytics`:
  - `event_counts` - total events per time window.
  - `salary_breakdown` - avg salary + event count per (window, country, role,
    experience range).
  - `technology_counts` - event count per (window, technology), from exploding the
    `;`-separated `LanguageHaveWorkedWith` field.
Records missing/unparseable `event_id` or `event_time` go to `salary_dead_letter`
instead, the same pattern `src/streaming/prediction_stream.py` uses for `salary_requests`.

Design choices worth documenting:
  - PLAN.md §15 lists "event counts" and "avg salary by country/role/experience range"
    as separate bullets; rather than one aggregation per dimension (country-only,
    role-only, experience-only - which would undercount by double-bucketing the same
    events three different ways), this groups by all three dimensions together in one
    table (`salary_breakdown`), which a downstream consumer can still slice/filter by any
    single dimension. `event_counts` stays a separate, simpler window-only aggregate
    since it answers a different question (raw throughput, not a salary breakdown).
  - `WINDOW_DURATION`/`WATERMARK_DELAY` are short (30s/15s) relative to a typical
    production streaming job - this is a VM demo, not a production deployment, and a
    multi-minute window would mean waiting several minutes of wall-clock time (via
    `dataset_producer`'s `event_time`, which tracks real time) before any aggregate
    result ever gets emitted. `notebooks/run_analytics_stream.ipynb` documents how long to
    keep the producer running for results to actually appear.
  - Kafka's structured-streaming sink only supports `append`/`update` output modes (not
    `complete`); `append` is used here, which Spark only allows for a watermarked
    aggregation - exactly the case here - so each window's result is emitted exactly
    once, after the watermark confirms it's final, rather than repeatedly updated.

Run with:

    python -m src.streaming.analytics_stream
"""

from __future__ import annotations

import os
import signal

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.streaming import StreamingQuery

from config import settings
from src.common.logging_config import get_logger
from src.common.schemas import DEVELOPER_EVENT_SCHEMA
from src.common.spark_session import get_spark_session
from src.common.spark_utils import to_kafka_rows

logger = get_logger(__name__)

EVENT_TIME_FORMAT = "yyyy-MM-dd'T'HH:mm:ss'Z'"
WINDOW_DURATION = "30 seconds"
WATERMARK_DELAY = "15 seconds"


def parse_events(raw_stream: DataFrame) -> DataFrame:
    """§15 step 1-2: parse Kafka records against the explicit event schema, and derive a
    real timestamp column (`event_timestamp`) from the string `event_time` field for
    windowing/watermarking."""
    parsed = (
        raw_stream.select(
            F.col("value").cast("string").alias("raw_value"),
            F.col("timestamp").alias("received_at"),
        )
        .withColumn("event", F.from_json(F.col("raw_value"), DEVELOPER_EVENT_SCHEMA))
        .select("raw_value", "received_at", "event.*")
    )
    return parsed.withColumn(
        "event_timestamp", F.to_timestamp(F.col("event_time"), EVENT_TIME_FORMAT)
    )


def split_valid_and_dead_letters(parsed: DataFrame) -> tuple[DataFrame, DataFrame]:
    """§15 step 3: malformed records (missing/unparseable `event_id` or `event_time`) go
    to `salary_dead_letter` instead of the aggregates below."""
    is_valid = F.col("event_id").isNotNull() & F.col("event_timestamp").isNotNull()

    valid = parsed.filter(is_valid)
    dead_letters = parsed.filter(~is_valid).select(
        F.lit(settings.KAFKA_DATASET_TOPIC).alias("source_topic"),
        F.col("raw_value"),
        F.lit("missing or unparseable event_id/event_time").alias("error_reason"),
        F.col("received_at").cast("string").alias("received_at"),
    )
    return valid, dead_letters


def add_experience_range(df: DataFrame) -> DataFrame:
    """Buckets `YearsCodePro` into coarse ranges for the salary breakdown - a documented
    judgment call (not derived from the data), same spirit as `MAX_PLAUSIBLE_SALARY` in
    `config/settings.py`."""
    return df.withColumn(
        "experience_range",
        F.when(F.col("YearsCodePro").isNull(), "Unknown")
        .when(F.col("YearsCodePro") < 3, "0-2 years")
        .when(F.col("YearsCodePro") < 6, "3-5 years")
        .when(F.col("YearsCodePro") < 11, "6-10 years")
        .otherwise("11+ years"),
    )


def build_event_counts(valid: DataFrame) -> DataFrame:
    """§15: "Compute event counts" - total events per time window."""
    aggregated = (
        valid.withWatermark("event_timestamp", WATERMARK_DELAY)
        .groupBy(F.window("event_timestamp", WINDOW_DURATION))
        .agg(F.count(F.lit(1)).alias("event_count"))
    )
    return aggregated.select(
        F.lit("event_counts").alias("metric"),
        F.col("window.start").cast("string").alias("window_start"),
        F.col("window.end").cast("string").alias("window_end"),
        F.col("event_count"),
    )


def build_salary_breakdown(valid: DataFrame) -> DataFrame:
    """§15: "avg salary by country/role/experience range", combined into one grouped
    table (see module docstring for why).

    `Country`/`DevType` are "Unknown"-filled before grouping (same pattern
    `prediction_stream.py` uses for `salary_requests`) - without this, a null `Country`/
    `DevType` groups events into a null-keyed row, and `to_kafka_rows()`'s `to_json()`
    then *drops that key from the JSON entirely* (Spark's default null-handling), rather
    than writing `null` - which breaks any consumer (e.g. the dashboard) that assumes the
    key is always present. Rows where every event in the group had a missing salary
    (`avg_salary` still null after aggregation) are dropped for the same reason - a
    salary breakdown with no known salary isn't meaningful to show anyway.
    """
    enriched = add_experience_range(valid).fillna("Unknown", subset=["Country", "DevType"])
    aggregated = (
        enriched.withWatermark("event_timestamp", WATERMARK_DELAY)
        .groupBy(
            F.window("event_timestamp", WINDOW_DURATION),
            F.col("Country"),
            F.col("DevType"),
            F.col("experience_range"),
        )
        .agg(
            F.avg("ConvertedCompYearly").alias("avg_salary"),
            F.count(F.lit(1)).alias("event_count"),
        )
        .filter(F.col("avg_salary").isNotNull())
    )
    return aggregated.select(
        F.lit("salary_breakdown").alias("metric"),
        F.col("window.start").cast("string").alias("window_start"),
        F.col("window.end").cast("string").alias("window_end"),
        F.col("Country").alias("country"),
        F.col("DevType").alias("role"),
        F.col("experience_range"),
        F.round(F.col("avg_salary"), 2).alias("avg_salary"),
        F.col("event_count"),
    )


def build_technology_counts(valid: DataFrame) -> DataFrame:
    """§15: "common technologies" - event count *and* avg salary per technology,
    exploding the `;`-separated `LanguageHaveWorkedWith` field. `avg_salary` was added
    (2026-08-18) for the dashboard's "salary by ... language/technology" requirement
    (§17) - same aggregation, just one more column, rather than a separate 4th query."""
    exploded = valid.withColumn(
        "technology", F.explode(F.split(F.col("LanguageHaveWorkedWith"), ";"))
    ).filter((F.col("technology") != "") & (F.col("technology") != "Unknown"))
    aggregated = (
        exploded.withWatermark("event_timestamp", WATERMARK_DELAY)
        .groupBy(F.window("event_timestamp", WINDOW_DURATION), F.col("technology"))
        .agg(
            F.count(F.lit(1)).alias("event_count"),
            F.avg("ConvertedCompYearly").alias("avg_salary"),
        )
    )
    return aggregated.select(
        F.lit("technology_counts").alias("metric"),
        F.col("window.start").cast("string").alias("window_start"),
        F.col("window.end").cast("string").alias("window_end"),
        F.col("technology"),
        F.col("event_count"),
        F.round(F.col("avg_salary"), 2).alias("avg_salary"),
    )


def build_streams(spark: SparkSession) -> tuple[StreamingQuery, StreamingQuery, StreamingQuery, StreamingQuery]:
    """Builds and starts all four streaming queries (dead letters + three aggregates) and
    returns them immediately, without blocking - same non-blocking pattern as
    `src/streaming/prediction_stream.py: build_streams()`, for interactive/notebook use.
    """
    raw_stream = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", settings.KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", settings.KAFKA_DATASET_TOPIC)
        .option("startingOffsets", "latest")
        .load()
    )

    parsed = parse_events(raw_stream)
    valid, dead_letters = split_valid_and_dead_letters(parsed)

    def _start(df: DataFrame, checkpoint_subdir: str) -> StreamingQuery:
        return (
            to_kafka_rows(df, key_col=None)
            .writeStream.format("kafka")
            .option("kafka.bootstrap.servers", settings.KAFKA_BOOTSTRAP_SERVERS)
            .option("topic", settings.KAFKA_ANALYTICS_TOPIC)
            .option(
                "checkpointLocation",
                os.path.join(settings.ANALYTICS_CHECKPOINT_PATH, checkpoint_subdir),
            )
            .outputMode("append")
            .start()
        )

    dead_letter_query = (
        to_kafka_rows(dead_letters, key_col=None)
        .writeStream.format("kafka")
        .option("kafka.bootstrap.servers", settings.KAFKA_BOOTSTRAP_SERVERS)
        .option("topic", settings.KAFKA_DEAD_LETTER_TOPIC)
        .option(
            "checkpointLocation",
            os.path.join(settings.ANALYTICS_CHECKPOINT_PATH, "dead_letters"),
        )
        .outputMode("append")
        .start()
    )
    event_counts_query = _start(build_event_counts(valid), "event_counts")
    salary_breakdown_query = _start(build_salary_breakdown(valid), "salary_breakdown")
    technology_counts_query = _start(build_technology_counts(valid), "technology_counts")

    logger.info(
        "Analytics stream running: %s -> %s (event_counts, salary_breakdown, "
        "technology_counts; dead letters -> %s).",
        settings.KAFKA_DATASET_TOPIC, settings.KAFKA_ANALYTICS_TOPIC, settings.KAFKA_DEAD_LETTER_TOPIC,
    )
    return dead_letter_query, event_counts_query, salary_breakdown_query, technology_counts_query


def run_analytics_stream() -> None:
    """Blocking CLI entry point: `python -m src.streaming.analytics_stream`."""
    spark = get_spark_session(app_name="SalaryAnalyticsStream", with_kafka=True)
    queries = build_streams(spark)
    logger.info("Ctrl+C to stop.")

    def _graceful_shutdown(signum, frame):  # noqa: ARG001 - signal handler signature
        logger.info("Received shutdown signal, stopping streaming queries...")
        for query in queries:
            query.stop()

    signal.signal(signal.SIGTERM, _graceful_shutdown)
    signal.signal(signal.SIGINT, _graceful_shutdown)

    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    run_analytics_stream()

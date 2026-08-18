import json
from datetime import datetime, timezone

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.streaming.analytics_stream import (
    add_experience_range,
    build_event_counts,
    build_salary_breakdown,
    build_technology_counts,
    parse_events,
    split_valid_and_dead_letters,
)


@pytest.fixture(scope="module")
def spark():
    session = SparkSession.builder.master("local[1]").appName("test_analytics_stream").getOrCreate()
    yield session
    session.stop()


def _raw_kafka_df(spark, payloads):
    rows = [(json.dumps(payload).encode("utf-8"),) for payload in payloads]
    df = spark.createDataFrame(rows, ["value"])
    return df.withColumn("timestamp", F.current_timestamp())


def test_parse_events_derives_event_timestamp(spark):
    raw = _raw_kafka_df(
        spark, [{"event_id": "e1", "event_time": "2026-08-18T15:51:37Z", "Country": "Germany"}]
    )
    parsed = parse_events(raw)
    row = parsed.first()

    assert row["event_id"] == "e1"
    assert row["Country"] == "Germany"
    assert row["event_timestamp"] == datetime(2026, 8, 18, 15, 51, 37, tzinfo=timezone.utc).replace(tzinfo=None)


def test_split_routes_missing_event_id_or_time_to_dead_letters(spark):
    raw = _raw_kafka_df(
        spark,
        [
            {"event_id": "e1", "event_time": "2026-08-18T15:51:37Z"},
            {"event_time": "2026-08-18T15:51:37Z"},  # no event_id
            {"event_id": "e3", "event_time": "not-a-timestamp"},  # unparseable time
        ],
    )
    parsed = parse_events(raw)
    valid, dead_letters = split_valid_and_dead_letters(parsed)

    assert valid.count() == 1
    assert valid.first()["event_id"] == "e1"
    assert dead_letters.count() == 2
    assert all(
        row["error_reason"] == "missing or unparseable event_id/event_time"
        for row in dead_letters.collect()
    )


def test_add_experience_range_buckets_correctly(spark):
    df = spark.createDataFrame(
        [(None,), (1.0,), (4.0,), (8.0,), (20.0,)], ["YearsCodePro"]
    )
    result = {row["YearsCodePro"]: row["experience_range"] for row in add_experience_range(df).collect()}

    assert result[None] == "Unknown"
    assert result[1.0] == "0-2 years"
    assert result[4.0] == "3-5 years"
    assert result[8.0] == "6-10 years"
    assert result[20.0] == "11+ years"


def _timestamped(spark, rows, columns):
    df = spark.createDataFrame(rows, columns)
    return df.withColumn("event_timestamp", F.to_timestamp(F.col("event_timestamp")))


def test_build_event_counts_counts_events_in_the_same_window(spark):
    valid = _timestamped(
        spark,
        [("2026-08-18T10:00:01Z",), ("2026-08-18T10:00:05Z",)],
        ["event_timestamp"],
    )
    result = build_event_counts(valid).first()

    assert result["metric"] == "event_counts"
    assert result["event_count"] == 2


def test_build_salary_breakdown_averages_within_group(spark):
    valid = _timestamped(
        spark,
        [
            ("2026-08-18T10:00:01Z", "Germany", "Developer", 4.0, 100000.0),
            ("2026-08-18T10:00:05Z", "Germany", "Developer", 4.0, 120000.0),
        ],
        ["event_timestamp", "Country", "DevType", "YearsCodePro", "ConvertedCompYearly"],
    )
    result = build_salary_breakdown(valid).first()

    assert result["metric"] == "salary_breakdown"
    assert result["country"] == "Germany"
    assert result["role"] == "Developer"
    assert result["experience_range"] == "3-5 years"
    assert result["avg_salary"] == 110000.0
    assert result["event_count"] == 2


def test_build_technology_counts_explodes_and_filters_unknown(spark):
    valid = _timestamped(
        spark,
        [
            ("2026-08-18T10:00:01Z", "Python;SQL"),
            ("2026-08-18T10:00:05Z", "Python"),
            ("2026-08-18T10:00:05Z", "Unknown"),
        ],
        ["event_timestamp", "LanguageHaveWorkedWith"],
    )
    counts = {
        row["technology"]: row["event_count"] for row in build_technology_counts(valid).collect()
    }

    assert counts["Python"] == 2
    assert counts["SQL"] == 1
    assert "Unknown" not in counts

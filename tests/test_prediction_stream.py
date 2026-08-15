import json
import math

import pytest
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.streaming.prediction_stream import (
    build_predictions,
    parse_requests,
    split_valid_and_dead_letters,
    to_kafka_rows,
)
from src.training.feature_pipeline import build_feature_stages

TRAIN_COLUMNS = [
    "Country", "Age", "EdLevel", "RemoteWork", "DevType", "OrgSize", "Industry",
    "Employment", "LanguageHaveWorkedWith", "DatabaseHaveWorkedWith", "PlatformHaveWorkedWith",
    "YearsCodeProNumeric", "log_label",
]


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.master("local[1]").appName("test_prediction_stream").getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture(scope="module")
def fitted_model(spark):
    rows = [
        (
            "USA", "25-34 years old", "Bachelor's", "Remote", "Developer",
            "20 to 99 employees", "Tech", "Employed, full-time", "Python;SQL",
            "PostgreSQL", "AWS", 5.0, 11.5129,
        ),
        (
            "Israel", "35-44 years old", "Master's", "Hybrid", "Developer",
            "100 to 499 employees", "Tech", "Employed, full-time", "Java;Python",
            "MySQL", "GCP", 8.0, 11.4076,
        ),
        (
            "Germany", "45-54 years old", "PhD", "In-person", "Manager",
            "10,000 or more employees", "Finance", "Employed, part-time", "Unknown",
            "Unknown", "Unknown", 15.0, 11.6952,
        ),
    ]
    df = spark.createDataFrame(rows, TRAIN_COLUMNS)
    regressor = LinearRegression(
        featuresCol="features", labelCol="log_label", predictionCol="log_prediction", maxIter=5
    )
    return Pipeline(stages=build_feature_stages() + [regressor]).fit(df)


def _raw_kafka_df(spark, payloads):
    rows = [(json.dumps(payload).encode("utf-8"),) for payload in payloads]
    df = spark.createDataFrame(rows, ["value"])
    return df.withColumn("timestamp", F.current_timestamp())


def test_parse_requests_extracts_schema_fields(spark):
    raw = _raw_kafka_df(
        spark,
        [{"request_id": "abc-123", "Country": "USA", "YearsCodePro": 5}],
    )
    parsed = parse_requests(raw)
    row = parsed.first()

    assert row["request_id"] == "abc-123"
    assert row["Country"] == "USA"
    assert row["YearsCodePro"] == 5.0


def test_parse_requests_handles_malformed_json(spark):
    raw = spark.createDataFrame([("not valid json{{{",)], ["value"]).withColumn(
        "timestamp", F.current_timestamp()
    )
    parsed = parse_requests(raw)
    row = parsed.first()

    assert row["request_id"] is None


def test_split_routes_missing_request_id_to_dead_letters(spark):
    raw = _raw_kafka_df(
        spark,
        [
            {"request_id": "has-id", "Country": "USA"},
            {"Country": "Israel"},  # no request_id
        ],
    )
    parsed = parse_requests(raw)
    valid, dead_letters = split_valid_and_dead_letters(parsed)

    assert valid.count() == 1
    assert valid.first()["request_id"] == "has-id"
    assert dead_letters.count() == 1
    assert dead_letters.first()["error_reason"] == "missing or unparseable request_id"


def test_split_fills_missing_features_with_unknown(spark):
    raw = _raw_kafka_df(spark, [{"request_id": "abc-123", "Country": "USA"}])
    parsed = parse_requests(raw)
    valid, _dead_letters = split_valid_and_dead_letters(parsed)
    result = valid.first()

    assert result["EdLevel"] == "Unknown"
    assert result["Employment"] == "Unknown"
    assert result["LanguageHaveWorkedWith"] == "Unknown"


def test_split_renames_years_code_pro_for_the_pipeline(spark):
    raw = _raw_kafka_df(spark, [{"request_id": "abc-123", "YearsCodePro": 7}])
    parsed = parse_requests(raw)
    valid, _dead_letters = split_valid_and_dead_letters(parsed)

    assert "YearsCodeProNumeric" in valid.columns
    assert "YearsCodePro" not in valid.columns
    assert valid.first()["YearsCodeProNumeric"] == 7.0


def test_build_predictions_reverses_log1p_and_shapes_response(spark, fitted_model):
    raw = _raw_kafka_df(
        spark,
        [{"request_id": "abc-123", "Country": "USA", "YearsCodePro": 5}],
    )
    parsed = parse_requests(raw)
    valid, _dead_letters = split_valid_and_dead_letters(parsed)

    predictions = build_predictions(valid, fitted_model, "LinearRegression", "2026-08-15T00:00:00Z")
    row = predictions.first()

    assert set(predictions.columns) == {
        "request_id", "prediction", "target_unit", "model_name", "model_version",
        "processed_at", "status",
    }
    assert row["request_id"] == "abc-123"
    assert row["prediction"] is not None and row["prediction"] >= 0
    assert row["model_name"] == "LinearRegression"
    assert row["status"] == "success"
    assert row["target_unit"] == "annual salary"


def test_to_kafka_rows_keys_by_given_column(spark):
    df = spark.createDataFrame([("abc-123", 100.0)], ["request_id", "prediction"])
    kafka_rows = to_kafka_rows(df, key_col="request_id")
    row = kafka_rows.first()

    assert row["key"] == "abc-123"
    payload = json.loads(row["value"])
    assert payload["request_id"] == "abc-123"
    assert payload["prediction"] == 100.0


def test_to_kafka_rows_uses_null_key_when_no_key_column(spark):
    df = spark.createDataFrame([("some raw value",)], ["raw_value"])
    kafka_rows = to_kafka_rows(df, key_col=None)
    row = kafka_rows.first()

    assert row["key"] is None

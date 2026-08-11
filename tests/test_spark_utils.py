import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.common.spark_utils import is_missing_text, safe_cast_double


@pytest.fixture(scope="module")
def spark():
    session = SparkSession.builder.master("local[1]").appName("test_spark_utils").getOrCreate()
    yield session
    session.stop()


def test_is_missing_text_catches_null_blank_and_na(spark):
    df = spark.createDataFrame(
        [("USA",), ("NA",), ("",), ("  ",), (None,)], ["Country"]
    )
    result = df.withColumn("missing", is_missing_text(F.col("Country"))).collect()
    flags = {row["Country"]: row["missing"] for row in result}

    assert flags["USA"] is False
    assert flags["NA"] is True
    assert flags[""] is True
    assert flags["  "] is True
    assert flags[None] is True


def test_safe_cast_double_rejects_na(spark):
    df = spark.createDataFrame([("100000",), ("NA",), ("-500",)], ["value"])
    result = df.withColumn("cast_value", safe_cast_double(F.col("value"))).collect()
    values = {row["value"]: row["cast_value"] for row in result}

    assert values["100000"] == 100000.0
    assert values["NA"] is None
    assert values["-500"] == -500.0

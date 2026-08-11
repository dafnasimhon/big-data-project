import pytest
from pyspark.sql import SparkSession

from src.exploration.explore_dataset import check_target_candidates, missing_value_summary

TARGET_COLUMN = "ConvertedCompYearly"


@pytest.fixture(scope="module")
def spark():
    session = SparkSession.builder.master("local[1]").appName("test_explore_dataset").getOrCreate()
    yield session
    session.stop()


def test_missing_value_summary_counts_na_string_as_missing(spark):
    # Regression test for the bug found on the real VM run (2026-08-11): this dataset's
    # missing-value sentinel is the literal string "NA", not null/blank. A naive
    # isNull()-only check reported 0% missing on a column that's actually ~46% missing.
    df = spark.createDataFrame(
        [("USA",), ("NA",), ("Israel",), ("NA",)], ["Country"]
    )
    summary = {row["column"]: row["missing_count"] for row in missing_value_summary(df, ["Country"]).collect()}
    assert summary["Country"] == 2


def test_check_target_candidates_finds_present_and_missing_columns(spark):
    df = spark.createDataFrame([(100000,)], [TARGET_COLUMN])
    result = check_target_candidates(df)

    assert result["present"] == [TARGET_COLUMN]
    assert "CompFreq" in result["missing"]


def test_check_target_candidates_stops_without_target_column(spark):
    df = spark.createDataFrame([("x",)], ["SomeOtherColumn"])
    with pytest.raises(RuntimeError):
        check_target_candidates(df)

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, StructField, StructType

from config import settings
from src.training.data_cleaning import clean_dataset, convert_years_code_pro

RAW_COLUMNS = [
    "Country",
    "Age",
    "EdLevel",
    "Employment",
    "RemoteWork",
    "YearsCodePro",
    "DevType",
    "OrgSize",
    "Industry",
    "LanguageHaveWorkedWith",
    "DatabaseHaveWorkedWith",
    "PlatformHaveWorkedWith",
    "ConvertedCompYearly",
]

# Explicit schema (matches how these columns actually arrive from the real CSV read via
# data_loader.py) so a small sample where a column is null in every row doesn't hit
# PySpark's "CANNOT_DETERMINE_TYPE" schema-inference error.
RAW_SCHEMA = StructType([StructField(name, StringType(), True) for name in RAW_COLUMNS])


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.master("local[1]").appName("test_data_cleaning").getOrCreate()
    )
    yield session
    session.stop()


def test_convert_years_code_pro(spark):
    df = spark.createDataFrame(
        [("Less than 1 year",), ("More than 50 years",), ("7",), (None,)],
        ["YearsCodePro"],
    )
    rows = {row["YearsCodePro"]: row["YearsCodeProNumeric"] for row in convert_years_code_pro(df).collect()}

    assert rows["Less than 1 year"] == 0.5
    assert rows["More than 50 years"] == 51.0
    assert rows["7"] == 7.0
    assert rows[None] is None


def test_clean_dataset_drops_missing_and_nonpositive_target(spark):
    rows = [
        (
            "USA", "25-34 years old", "Bachelor's", "Employed, full-time", "Remote", "5",
            "Developer", "20 to 99 employees", "Other", "Python;SQL", "PostgreSQL", "AWS",
            "100000",
        ),
        (
            "USA", "25-34 years old", "Bachelor's", "Employed, full-time", "Remote", "5",
            "Developer", "20 to 99 employees", "Other", "Python;SQL", "PostgreSQL", "AWS",
            "NA",
        ),
        (
            "USA", "25-34 years old", "Bachelor's", "Employed, full-time", "Remote", "5",
            "Developer", "20 to 99 employees", "Other", "Python;SQL", "PostgreSQL", "AWS",
            "-500",
        ),
    ]
    df = spark.createDataFrame(rows, schema=RAW_SCHEMA)

    cleaned = clean_dataset(df)

    assert cleaned.count() == 1
    result = cleaned.first()
    assert result["label"] == 100000.0
    assert result["log_label"] == pytest.approx(11.5129, rel=1e-3)


def test_clean_dataset_fills_null_categoricals_with_unknown(spark):
    rows = [
        (
            None, "25-34 years old", None, "Employed, full-time", "Remote", "5",
            "Developer", "20 to 99 employees", None, None, None, None,
            "100000",
        ),
    ]
    df = spark.createDataFrame(rows, schema=RAW_SCHEMA)

    cleaned = clean_dataset(df)
    result = cleaned.first()

    assert result["Country"] == "Unknown"
    assert result["EdLevel"] == "Unknown"
    assert result["Industry"] == "Unknown"
    assert result["LanguageHaveWorkedWith"] == "Unknown"


def test_clean_dataset_fills_na_string_categoricals_with_unknown(spark):
    # The real dataset encodes missing values as the literal string "NA", not SQL null or
    # a blank string — this is the case that actually matters (see spark_utils.py:
    # is_missing_text and data_cleaning.py's module docstring for how this was found).
    rows = [
        (
            "NA", "25-34 years old", "NA", "Employed, full-time", "Remote", "5",
            "Developer", "20 to 99 employees", "NA", "NA", "NA", "NA",
            "100000",
        ),
    ]
    df = spark.createDataFrame(rows, schema=RAW_SCHEMA)

    cleaned = clean_dataset(df)
    result = cleaned.first()

    assert result["Country"] == "Unknown"
    assert result["EdLevel"] == "Unknown"
    assert result["Industry"] == "Unknown"
    assert result["LanguageHaveWorkedWith"] == "Unknown"
    assert result["DatabaseHaveWorkedWith"] == "Unknown"
    assert result["PlatformHaveWorkedWith"] == "Unknown"


def test_clean_dataset_drops_implausibly_high_salaries(spark):
    # Regression test for the §10.9 revision (2026-08-11): log1p alone let this dataset's
    # extreme values (max $74,351,432) dominate RMSE in real training. Rows above
    # MAX_PLAUSIBLE_SALARY must now be dropped outright, not just log-transformed.
    base = (
        "USA", "25-34 years old", "Bachelor's", "Employed, full-time", "Remote", "5",
        "Developer", "20 to 99 employees", "Other", "Python;SQL", "PostgreSQL", "AWS",
    )
    rows = [
        base + (str(settings.MAX_PLAUSIBLE_SALARY),),  # at the cap: kept
        base + (str(settings.MAX_PLAUSIBLE_SALARY + 1),),  # just over: dropped
        base + ("74351432",),  # this dataset's real max: dropped
    ]
    df = spark.createDataFrame(rows, schema=RAW_SCHEMA)

    cleaned = clean_dataset(df)

    assert cleaned.count() == 1
    assert cleaned.first()["label"] == float(settings.MAX_PLAUSIBLE_SALARY)


def test_clean_dataset_deduplicates_rows(spark):
    row = (
        "USA", "25-34 years old", "Bachelor's", "Employed, full-time", "Remote", "5",
        "Developer", "20 to 99 employees", "Other", "Python;SQL", "PostgreSQL", "AWS",
        "100000",
    )
    df = spark.createDataFrame([row, row], RAW_COLUMNS)

    cleaned = clean_dataset(df)

    assert cleaned.count() == 1


def test_clean_dataset_leaves_years_code_pro_unimputed(spark):
    rows = [
        (
            "USA", "25-34 years old", "Bachelor's", "Employed, full-time", "Remote", None,
            "Developer", "20 to 99 employees", "Other", "Python;SQL", "PostgreSQL", "AWS",
            "100000",
        ),
    ]
    df = spark.createDataFrame(rows, schema=RAW_SCHEMA)

    cleaned = clean_dataset(df)
    result = cleaned.first()

    # Imputation happens inside the Spark ML Pipeline (feature_pipeline.py), fit only on
    # the training fold, so it must NOT be filled in here — see data_cleaning.py docstring.
    assert result["YearsCodeProNumeric"] is None

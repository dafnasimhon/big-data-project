import json
import os

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from config import settings
from src.common.feature_config import CANDIDATE_INPUT_FEATURES, TARGET_COLUMN
from src.common.logging_config import get_logger
from src.common.spark_session import get_spark_session
from src.common.spark_utils import is_missing_text, safe_cast_double
from src.training.data_loader import load_raw_dataset

logger = get_logger(__name__)

# Absolute, not relative to whatever the caller's (or the Spark JVM's - see
# config/settings.py's PROJECT_ROOT comment) working directory happens to be.
REPORT_DIR = os.path.join(settings.PROJECT_ROOT, "data", "processed")
TARGET_CANDIDATE_COLUMNS = ["CompTotal", "Currency", "ConvertedCompYearly", "CompFreq"]


def check_target_candidates(df: DataFrame) -> dict:
    """§9: confirm target-related columns exist; stop rather than proceed silently."""
    present = [column for column in TARGET_CANDIDATE_COLUMNS if column in df.columns]
    missing = [column for column in TARGET_CANDIDATE_COLUMNS if column not in df.columns]

    if TARGET_COLUMN not in present:
        raise RuntimeError(
            f"STOP: target column '{TARGET_COLUMN}' is not present in the dataset "
            f"columns ({sorted(df.columns)}). Do not proceed with training until the "
            f"target decision in PLAN.md §2 is re-verified against this file."
        )

    return {"present": present, "missing": missing}


def missing_value_summary(df: DataFrame, columns: list) -> DataFrame:
    total = df.count()
    rows = []
    for column in columns:
        missing = df.filter(is_missing_text(F.col(column))).count()
        percentage = round(missing / total * 100, 2) if total else 0.0
        rows.append((column, missing, percentage))
    return df.sparkSession.createDataFrame(rows, ["column", "missing_count", "missing_percentage"])


def cardinality_summary(df: DataFrame, columns: list) -> dict:
    return {column: df.select(column).distinct().count() for column in columns}


def top_values(df: DataFrame, column: str, top_n: int, delimiter: str = ";") -> list:
    exploded = (
        df.filter(~is_missing_text(F.col(column)))
        .select(F.explode(F.split(F.col(column), delimiter)).alias("value"))
        .filter(~is_missing_text(F.col("value")))
    )
    return [
        [row["value"], row["count"]]
        for row in exploded.groupBy("value").count().orderBy(F.desc("count")).limit(top_n).collect()
    ]


def salary_outlier_quantiles(df: DataFrame) -> dict:
    numeric = df.withColumn(
        "salary_numeric", safe_cast_double(F.regexp_replace(F.col(TARGET_COLUMN), ",", ""))
    ).filter(F.col("salary_numeric").isNotNull())

    quantiles = numeric.approxQuantile("salary_numeric", [0.01, 0.25, 0.5, 0.75, 0.95, 0.99], 0.0)
    max_value = numeric.agg(F.max("salary_numeric")).first()[0]

    stats = dict(zip(["p1", "p25", "p50", "p75", "p95", "p99"], quantiles))
    stats["max"] = max_value
    return stats


def build_report(spark: SparkSession) -> dict:
    df = load_raw_dataset(spark)

    target_check = check_target_candidates(df)

    row_count = df.count()
    column_count = len(df.columns)

    selected_df = df.select(*CANDIDATE_INPUT_FEATURES, TARGET_COLUMN)
    missing_df = missing_value_summary(selected_df, selected_df.columns)
    missing_df.coalesce(1).write.mode("overwrite").option("header", True).csv(
        os.path.join(REPORT_DIR, "missing_values_summary")
    )

    salary_present = df.filter(~is_missing_text(F.col(TARGET_COLUMN))).count()

    report = {
        "row_count": row_count,
        "column_count": column_count,
        "target_column": TARGET_COLUMN,
        "target_unit": "USD, annualized (Stack Overflow's own currency-normalized figure)",
        "target_candidate_columns": target_check,
        "rows_with_target_present": salary_present,
        "rows_with_target_missing": row_count - salary_present,
        "rows_excluded_reason": (
            "missing/blank ConvertedCompYearly, or non-positive after numeric cast "
            "(finalized during cleaning, see src/training/data_cleaning.py)"
        ),
        "cardinality": cardinality_summary(
            df,
            ["Country", "Age", "EdLevel", "Employment", "RemoteWork", "DevType", "OrgSize", "Industry"],
        ),
        "top_languages": top_values(df, "LanguageHaveWorkedWith", settings.TOP_LANGUAGES),
        "top_databases": top_values(df, "DatabaseHaveWorkedWith", settings.TOP_DATABASES),
        "top_platforms": top_values(df, "PlatformHaveWorkedWith", settings.TOP_PLATFORMS),
        "salary_outlier_quantiles": salary_outlier_quantiles(df),
    }
    return report


def main() -> None:
    spark = get_spark_session(app_name="SalaryDataExploration")
    os.makedirs(REPORT_DIR, exist_ok=True)

    report = build_report(spark)

    report_path = os.path.join(REPORT_DIR, "exploration_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)

    logger.info("Exploration report written to %s", report_path)
    logger.info("Target column: %s (%s)", report["target_column"], report["target_unit"])
    logger.info(
        "Usable rows: %d / %d (%.1f%%)",
        report["rows_with_target_present"],
        report["row_count"],
        100 * report["rows_with_target_present"] / report["row_count"],
    )


if __name__ == "__main__":
    main()

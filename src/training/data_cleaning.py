from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from config import settings
from src.common.feature_config import (
    CANDIDATE_INPUT_FEATURES,
    SINGLE_VALUE_CATEGORICAL_COLUMNS,
    MULTI_VALUE_COLUMNS,
    TARGET_COLUMN,
)
from src.common.logging_config import get_logger
from src.common.spark_utils import is_missing_text, safe_cast_double

logger = get_logger(__name__)

FINAL_COLUMNS = (
    SINGLE_VALUE_CATEGORICAL_COLUMNS + MULTI_VALUE_COLUMNS + ["YearsCodeProNumeric", "label", "log_label"]
)


def _log_row_count(df: DataFrame, step: str) -> DataFrame:
    count = df.count()
    logger.info("Rows after %s: %d", step, count)
    return df


def convert_years_code_pro(df: DataFrame) -> DataFrame:
    return df.withColumn(
        "YearsCodeProNumeric",
        F.when(F.col("YearsCodePro") == "Less than 1 year", F.lit(0.5))
        .when(F.col("YearsCodePro") == "More than 50 years", F.lit(51.0))
        .otherwise(safe_cast_double(F.col("YearsCodePro"))),
    )


def log_outlier_diagnostics(df: DataFrame) -> dict:
    quantiles = df.approxQuantile("label", [0.01, 0.25, 0.5, 0.75, 0.95, 0.99], 0.0)
    max_value = df.agg(F.max("label")).first()[0]
    stats = dict(zip(["p1", "p25", "p50", "p75", "p95", "p99"], quantiles))
    stats["max"] = max_value
    logger.info("label quantiles (diagnostic, no rows dropped): %s", stats)
    return stats


def clean_dataset(raw_df: DataFrame) -> DataFrame:
    df = raw_df.select(*CANDIDATE_INPUT_FEATURES, TARGET_COLUMN)
    df = _log_row_count(df, "selecting required columns (step 1)")

    # Step 2: blank strings / this dataset's "NA" placeholder -> null, drop duplicates.
    for column in df.columns:
        df = df.withColumn(
            column,
            F.when(is_missing_text(F.col(column)), None).otherwise(F.col(column)),
        )
    df = df.dropDuplicates()
    df = _log_row_count(df, "blank/NA-to-null + de-duplication (step 2)")

    # Step 3/4: cast target to numeric, remove rows with missing target.
    df = df.withColumn(
        "label", safe_cast_double(F.regexp_replace(F.col(TARGET_COLUMN), ",", ""))
    )
    df = df.filter(F.col("label").isNotNull())
    df = _log_row_count(df, "removing missing/invalid target (steps 3-4)")

    # Step 7: filter non-positive salaries.
    df = df.filter(F.col("label") > 0)
    df = _log_row_count(df, "filtering non-positive salaries (step 7)")

    # Step 4/5: YearsCodePro -> numeric.
    df = convert_years_code_pro(df)

    # Step 8: outlier investigation (diagnostic, on the data about to be filtered below).
    log_outlier_diagnostics(df)

    # Step 9: drop implausible extreme values (both ends), then log1p the rest (decision
    # documented in the module docstring / PLAN.md §23).
    df = df.filter(
        (F.col("label") >= settings.MIN_PLAUSIBLE_SALARY)
        & (F.col("label") <= settings.MAX_PLAUSIBLE_SALARY)
    )
    df = _log_row_count(
        df,
        f"filtering label outside [{settings.MIN_PLAUSIBLE_SALARY}, "
        f"{settings.MAX_PLAUSIBLE_SALARY}] as implausible (step 9)",
    )
    df = df.withColumn("log_label", F.log1p(F.col("label")))

    # Step 6: fill missing categorical/multi-value values with 'Unknown'.
    df = df.fillna("Unknown", subset=SINGLE_VALUE_CATEGORICAL_COLUMNS + MULTI_VALUE_COLUMNS)
    df = _log_row_count(df, "filling missing categorical/multi-value fields with 'Unknown' (step 6)")

    df = df.select(*FINAL_COLUMNS)
    df = _log_row_count(df, "final cleaned dataset (step 10)")
    return df

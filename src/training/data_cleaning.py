"""Implements PLAN.md §10 cleaning steps, logging row counts at every stage (§10.10).

Ported from the `notebooks/02_Data_Preprocessing.ipynb` prototype, with two corrections
documented in PLAN.md §23 "Known issues":
  - The prototype's outlier check used `approxQuantile(..., relError=0.01)`, which is a
    RANK-based error bound, not a value-based one — on this heavy-tailed salary column it
    silently returned the raw max as the "99th percentile" and filtered nothing. This
    version computes exact quantiles (`relativeError=0.0`, cheap at this row count) purely
    for diagnostics/logging.
  - §10.9 requires choosing EITHER controlled outlier filtering OR a log1p transform of
    the target, and documenting the choice: this module chooses log1p (`log_label`),
    matching what the prototype's own `03B_Model_Improvement.ipynb` empirically validated
    as an improvement over both the untransformed target and a percentile cutoff. No rows
    are dropped for being high earners; only missing/non-positive targets are removed.
  - `YearsCodeProNumeric` is intentionally left un-imputed here (nulls stay null). Median
    imputation happens inside the Spark ML `Imputer` stage in `feature_pipeline.py`, fit
    only on the training fold, so the imputed value never leaks information from
    validation/test rows (PLAN.md rule 7). The prototype imputed it globally before any
    split, which was a leakage shortcut this version deliberately avoids.
  - Numeric casts use a regex-validated helper (`safe_cast_double`) instead of a bare
    `.cast("double")`. Found via the local test suite: with Spark's ANSI SQL mode enabled
    (the default in newer Spark, e.g. the pyspark used to test this locally), casting a
    non-numeric string like `"NA"` *raises an exception* instead of returning null. The
    VM's Spark 3.3.0 happens to have ANSI off by default, which is the only reason the
    notebook prototype's bare `.cast("double")` didn't crash there — this version doesn't
    depend on that setting either way.
  - Step 2's blank-to-null pass also nullifies the literal text `"NA"`, not just empty
    strings. Found running against the real dataset on the VM (2026-08-11): this CSV
    encodes missing values as the literal string `"NA"` (confirmed by inspection — Spark's
    CSV reader's `nullValue` option defaults to `""`, not `"NA"`), so a blank-string-only
    check left `"NA"` sitting in every categorical/skill column, meaning step 6's
    `fillna("Unknown", ...)` never actually caught it (`fillna` only touches true nulls).
    See `src/common/spark_utils.py: is_missing_text()`.
  - `Employment` moved from `SINGLE_VALUE_CATEGORICAL_COLUMNS` to `MULTI_VALUE_COLUMNS`
    (2026-08-11) — it's actually a `;`-separated multi-select field, not single-valued;
    see `src/common/feature_config.py` for why (found via a real OutOfMemoryError
    training `RandomForestRegressor` on the VM). Step 6's "Unknown" fill-in and step 2's
    missing-value handling both already treat it identically to the other multi-value
    columns since it's just in the list now — no special-casing needed here.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

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
    """'Less than 1 year' -> 0.5, 'More than 50 years' -> 51, else numeric cast (§10.5)."""
    return df.withColumn(
        "YearsCodeProNumeric",
        F.when(F.col("YearsCodePro") == "Less than 1 year", F.lit(0.5))
        .when(F.col("YearsCodePro") == "More than 50 years", F.lit(51.0))
        .otherwise(safe_cast_double(F.col("YearsCodePro"))),
    )


def log_outlier_diagnostics(df: DataFrame) -> dict:
    """§10.8: investigate extreme outliers using (exact) quantiles. Diagnostic only."""
    quantiles = df.approxQuantile("label", [0.01, 0.25, 0.5, 0.75, 0.95, 0.99], 0.0)
    max_value = df.agg(F.max("label")).first()[0]
    stats = dict(zip(["p1", "p25", "p50", "p75", "p95", "p99"], quantiles))
    stats["max"] = max_value
    logger.info("label quantiles (diagnostic, no rows dropped): %s", stats)
    return stats


def clean_dataset(raw_df: DataFrame) -> DataFrame:
    """Run PLAN.md §10 steps 1-10 and return the cleaned, model-ready DataFrame."""
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

    # Step 8: outlier investigation (diagnostic; see module docstring for step 9 decision).
    log_outlier_diagnostics(df)

    # Step 9: log1p transform of the target (decision documented above/in PLAN.md §23).
    df = df.withColumn("log_label", F.log1p(F.col("label")))

    # Step 6: fill missing categorical/multi-value values with 'Unknown'.
    df = df.fillna("Unknown", subset=SINGLE_VALUE_CATEGORICAL_COLUMNS + MULTI_VALUE_COLUMNS)
    df = _log_row_count(df, "filling missing categorical/multi-value fields with 'Unknown' (step 6)")

    df = df.select(*FINAL_COLUMNS)
    df = _log_row_count(df, "final cleaned dataset (step 10)")
    return df

"""Small Spark DataFrame helpers shared across exploration/cleaning/training."""

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

_NUMERIC_PATTERN = r"^-?\d+(\.\d+)?$"
_MISSING_TEXT_VALUES = {"", "NA"}


def is_missing_text(column: Column) -> Column:
    """True for null, blank/whitespace-only, or this dataset's literal "NA" placeholder.

    The Stack Overflow survey CSV encodes missing values as the literal 2-character
    string "NA" (confirmed by inspection — Spark's CSV reader has no built-in null-value
    marker for it, `nullValue` defaults to "", not "NA"), not as SQL null or blank string.
    Checking only `isNull()`/blank-string missed this entirely — confirmed on the real VM
    run (2026-08-11): the exploration report claimed 0 missing target values out of
    89,184 rows, when the actual figure (via `safe_cast_double` on the target, which
    happens to reject "NA" as non-numeric) is 41,165 missing. Any column-level
    missing-value check must go through this helper, not a bare null/blank check.
    """
    text_column = F.trim(column.cast("string"))
    return column.isNull() | text_column.isin(_MISSING_TEXT_VALUES)


def safe_cast_double(column: Column) -> Column:
    """Cast to double, returning null for non-numeric text (e.g. 'NA') regardless of the
    session's ANSI SQL mode.

    A bare `.cast("double")` returns null for malformed input under Spark's default
    (non-ANSI) settings, but *raises* under ANSI SQL mode (the default in newer Spark).
    Validating with a regex first makes the behavior consistent either way.
    """
    return F.when(column.rlike(_NUMERIC_PATTERN), column.cast("double")).otherwise(
        F.lit(None).cast("double")
    )


def reverse_log1p_predictions(
    df: DataFrame, log_prediction_col: str = "log_prediction", output_col: str = "prediction"
) -> DataFrame:
    """Reverse a log1p target transform (`expm1`) and clip degenerate results.

    PLAN.md §13: "Reject any pipeline producing NaN/infinite/negative final predictions
    without controlled post-processing" — null/NaN/negative results are clipped to 0.0
    rather than propagated. Truly infinite `expm1` output would require `log_prediction`
    beyond ~709 (double overflow), which isn't reachable from a regressor trained on this
    feature set, so it isn't explicitly guarded here.
    """
    reversed_column = F.expm1(F.col(log_prediction_col))
    safe_column = F.when(
        F.col(log_prediction_col).isNull() | F.isnan(reversed_column) | (reversed_column < 0),
        F.lit(0.0),
    ).otherwise(reversed_column)
    return df.withColumn(output_col, safe_column)

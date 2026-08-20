"""Small Spark DataFrame helpers shared across exploration/cleaning/training."""

from __future__ import annotations

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

_NUMERIC_PATTERN = r"^-?\d+(\.\d+)?$"
_MISSING_TEXT_VALUES = {"", "NA"}


def is_missing_text(column: Column) -> Column:
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


def to_kafka_rows(df: DataFrame, key_col: str | None) -> DataFrame:
    key_expr = F.col(key_col).cast("string") if key_col else F.lit(None).cast("string")
    value_expr = F.to_json(F.struct(*df.columns), {"ignoreNullFields": "false"})
    return df.select(key_expr.alias("key"), value_expr.alias("value"))


def reverse_log1p_predictions(
    df: DataFrame, log_prediction_col: str = "log_prediction", output_col: str = "prediction"
) -> DataFrame:
    reversed_column = F.expm1(F.col(log_prediction_col))
    safe_column = F.when(
        F.col(log_prediction_col).isNull() | F.isnan(reversed_column) | (reversed_column < 0),
        F.lit(0.0),
    ).otherwise(reversed_column)
    return df.withColumn(output_col, safe_column)

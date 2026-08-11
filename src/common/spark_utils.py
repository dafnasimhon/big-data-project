"""Small Spark DataFrame helpers shared across exploration/cleaning/training."""

from pyspark.sql import Column
from pyspark.sql import functions as F

_NUMERIC_PATTERN = r"^-?\d+(\.\d+)?$"


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


from __future__ import annotations

from pyspark.sql import DataFrame

from config import settings

CV_TRAIN_RATIO = 0.6
VALIDATION_RATIO = 0.2
TEST_RATIO = 0.2


def split_dataset(df: DataFrame, seed: int | None = None) -> tuple[DataFrame, DataFrame, DataFrame]:
    resolved_seed = settings.RANDOM_SEED if seed is None else seed
    cv_train_df, validation_df, test_df = df.randomSplit(
        [CV_TRAIN_RATIO, VALIDATION_RATIO, TEST_RATIO], seed=resolved_seed
    )
    return cv_train_df, validation_df, test_df

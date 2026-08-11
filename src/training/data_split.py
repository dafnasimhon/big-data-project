"""Leakage-free train/validation/test split (PLAN.md §12, §23 Known Issue #1).

The notebook prototype reused the same 80/20 split's test half across three separate
decisions (pick the winning model, decide an "improvement" beat it, report final KPIs),
which means that "test" set was never actually an unbiased estimate by the time KPIs were
reported. This module fixes that by carving out three disjoint slices:

  - `cv_train` (60% of the full dataset): the only data `CrossValidator` ever sees, in
    `tune_models.py`. Hyperparameter search and per-fold averaging happen entirely inside
    this slice.
  - `validation` (20%): held out from CV entirely. Used exactly once per candidate model
    (in `tune_models.py`) to compute the real-scale (post-`expm1`) RMSE/MAE/R² that
    `select_best_model.py` actually compares across model families — CrossValidator's own
    `avgMetrics` only gives one metric in log-space, not the three real-scale metrics
    §12's tie-break rule needs.
  - `test` (20%): never touched until `train_final_model.py` evaluates the single already
    -selected winner, exactly once, after refitting on `cv_train + validation` combined.

This mirrors PLAN.md §12 ("remaining 80% for train/validation... CrossValidator or
TrainValidationSplit on the training portion") while additionally splitting that 80% so a
genuine, only-used-once validation slice exists for cross-model comparison.
"""

from pyspark.sql import DataFrame

from config import settings

CV_TRAIN_RATIO = 0.6
VALIDATION_RATIO = 0.2
TEST_RATIO = 0.2


def split_dataset(df: DataFrame, seed: int | None = None) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Return (cv_train_df, validation_df, test_df) — a fixed 60/20/20 split."""
    resolved_seed = settings.RANDOM_SEED if seed is None else seed
    cv_train_df, validation_df, test_df = df.randomSplit(
        [CV_TRAIN_RATIO, VALIDATION_RATIO, TEST_RATIO], seed=resolved_seed
    )
    return cv_train_df, validation_df, test_df

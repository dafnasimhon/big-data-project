"""The 4 required candidate models and their tuning grids (PLAN.md §12).

Every regressor predicts `log_label` (see `data_cleaning.py`'s log1p decision) into
`log_prediction`; reversing to the real salary scale happens downstream in
`tune_models.py`/`train_final_model.py` via `expm1`, per §13's requirement to report
metrics in the original scale.

Grids are deliberately small — "sized for the available machine" per §12. Each model
tunes exactly ONE hyperparameter over 2 values (others fixed at a reasonable default via
the constructor), cut down from 2 tuned hyperparameters/~4 combinations (2026-08-11,
after the VM run was taking too long even after the OOM fix below) — with `NUM_FOLDS=2`
(see `tune_models.py`), that's 2 combos x 2 folds + 1 final refit = 5 pipeline fits per
model, 20 total across all 4 candidates, down from 52. Still genuine cross-validated
tuning per §12, just over a narrower search space; widen both the grids and `NUM_FOLDS`
back up once a full run comfortably completes.

`RandomForestRegressor`'s and `GBTRegressor`'s ceilings (`numTrees`/`maxIter`, `maxDepth`)
were lowered from an initial [20, 50]/[5, 10] after a real `OutOfMemoryError: Java heap
space` on the VM (2026-08-11), inside `RandomForestRegressor`'s split-finding
(`RandomForests.findBestSplits` -> `collectAsMap`) — tree ensembles hold per-(node,
feature, bin) statistics in memory, and this project's one-hot-encoded categorical
features (`Employment` alone has ~107 distinct values in the real data — it's actually a
semicolon-separated multi-select field, not truly single-valued, which inflates its
one-hot width considerably more than a normal category would; worth revisiting as its own
fix later) plus the per-skill `CountVectorizer` columns add up to several hundred feature
dimensions. Fewer/shallower trees reduces peak memory without giving up real tuning.
"""

from __future__ import annotations

from dataclasses import dataclass

from pyspark.ml.regression import (
    DecisionTreeRegressor,
    GBTRegressor,
    LinearRegression,
    RandomForestRegressor,
)
from pyspark.ml.tuning import ParamGridBuilder

from config import settings


@dataclass
class ModelCandidate:
    name: str
    regressor: object
    param_grid: list


def build_candidates() -> list[ModelCandidate]:
    """Fresh regressor instances + param grids — call once per training run, since
    ParamGridBuilder grids are tied to the specific regressor instances they wrap."""
    seed = settings.RANDOM_SEED

    linear_regression = LinearRegression(
        featuresCol="features",
        labelCol="log_label",
        predictionCol="log_prediction",
        maxIter=100,
        elasticNetParam=0.0,
    )
    linear_regression_grid = (
        ParamGridBuilder().addGrid(linear_regression.regParam, [0.01, 0.1]).build()
    )

    decision_tree = DecisionTreeRegressor(
        featuresCol="features",
        labelCol="log_label",
        predictionCol="log_prediction",
        seed=seed,
        minInstancesPerNode=10,
    )
    decision_tree_grid = (
        ParamGridBuilder().addGrid(decision_tree.maxDepth, [5, 10]).build()
    )

    random_forest = RandomForestRegressor(
        featuresCol="features",
        labelCol="log_label",
        predictionCol="log_prediction",
        seed=seed,
        maxDepth=6,
    )
    random_forest_grid = (
        ParamGridBuilder().addGrid(random_forest.numTrees, [10, 20]).build()
    )

    gbt = GBTRegressor(
        featuresCol="features",
        labelCol="log_label",
        predictionCol="log_prediction",
        seed=seed,
        maxDepth=4,
    )
    gbt_grid = ParamGridBuilder().addGrid(gbt.maxIter, [10, 20]).build()

    return [
        ModelCandidate("LinearRegression", linear_regression, linear_regression_grid),
        ModelCandidate("DecisionTreeRegressor", decision_tree, decision_tree_grid),
        ModelCandidate("RandomForestRegressor", random_forest, random_forest_grid),
        ModelCandidate("GBTRegressor", gbt, gbt_grid),
    ]

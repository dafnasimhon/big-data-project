"""The 4 required candidate models and their tuning grids (PLAN.md §12).

Every regressor predicts `log_label` (see `data_cleaning.py`'s log1p decision) into
`log_prediction`; reversing to the real salary scale happens downstream in
`tune_models.py`/`train_final_model.py` via `expm1`, per §13's requirement to report
metrics in the original scale.

Grids were cut hard on 2026-08-11 (each model down to ONE tuned hyperparameter over 2
values, `NUM_FOLDS=2`) after real runtime/memory problems on the VM. Widened back up on
2026-08-15 now that two of the root causes are fixed — `Employment`'s one-hot cardinality
bug (see below) and slow sequential fitting (see `tune_models.py`'s `TUNING_PARALLELISM`,
which now runs fits concurrently across the VM's cores). Each model now tunes 2
hyperparameters over ~3x2 values (~6 combinations); with `NUM_FOLDS=3`, that's
6 combos x 3 folds + 1 final refit = 19 pipeline fits per model, 76 total — nearly 4x the
previous 20, but `TUNING_PARALLELISM=4` means roughly 4 of those run at once rather than
strictly one-at-a-time.

`RandomForestRegressor`'s and `GBTRegressor`'s ceilings (`numTrees`/`maxIter`, `maxDepth`)
are deliberately NOT restored to their original pre-OOM values ([20, 50] / [5, 10]) even
though the `Employment` fix freed up real headroom — concurrent fits (see above) mean
multiple tree ensembles can be mid-training on the driver JVM *simultaneously*, which
increases peak memory pressure rather than reducing it, so the ceilings stay moderate
(`numTrees` up to 30, `maxDepth` up to 8) as a deliberate safety margin. The original
OOM was inside `RandomForestRegressor`'s split-finding (`RandomForests.findBestSplits`
-> `collectAsMap`, which holds per-(node, feature, bin) statistics in memory) — tree
ensembles are the memory-sensitive candidates here, linear/single-tree models are not.
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
    )
    linear_regression_grid = (
        ParamGridBuilder()
        .addGrid(linear_regression.regParam, [0.001, 0.01, 0.1])
        .addGrid(linear_regression.elasticNetParam, [0.0, 0.5])
        .build()
    )

    decision_tree = DecisionTreeRegressor(
        featuresCol="features",
        labelCol="log_label",
        predictionCol="log_prediction",
        seed=seed,
    )
    decision_tree_grid = (
        ParamGridBuilder()
        .addGrid(decision_tree.maxDepth, [5, 10, 15])
        .addGrid(decision_tree.minInstancesPerNode, [1, 10])
        .build()
    )

    random_forest = RandomForestRegressor(
        featuresCol="features",
        labelCol="log_label",
        predictionCol="log_prediction",
        seed=seed,
    )
    random_forest_grid = (
        ParamGridBuilder()
        .addGrid(random_forest.numTrees, [10, 20, 30])
        .addGrid(random_forest.maxDepth, [5, 8])
        .build()
    )

    gbt = GBTRegressor(
        featuresCol="features",
        labelCol="log_label",
        predictionCol="log_prediction",
        seed=seed,
    )
    gbt_grid = (
        ParamGridBuilder()
        .addGrid(gbt.maxIter, [10, 20, 30])
        .addGrid(gbt.maxDepth, [3, 5])
        .build()
    )

    return [
        ModelCandidate("LinearRegression", linear_regression, linear_regression_grid),
        ModelCandidate("DecisionTreeRegressor", decision_tree, decision_tree_grid),
        ModelCandidate("RandomForestRegressor", random_forest, random_forest_grid),
        ModelCandidate("GBTRegressor", gbt, gbt_grid),
    ]

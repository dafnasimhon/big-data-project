
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

"""Refit the selected winner and evaluate it ONCE on the untouched test set (PLAN.md §12).

This is the other half of the §23 Known Issue #1 fix: `tune_models.py`/
`select_best_model.py` never look at `test_df`, so by the time this module runs, the
winning model type + hyperparameters were chosen entirely from `cv_train_df` and
`validation_df`. Refitting on `cv_train_df + validation_df` (all non-test data) before the
single test evaluation is standard practice — more training data for the model that's
actually being shipped, without ever letting the test set influence which model or which
hyperparameters were chosen.
"""

from __future__ import annotations

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.regression import (
    DecisionTreeRegressor,
    GBTRegressor,
    LinearRegression,
    RandomForestRegressor,
)
from pyspark.sql import DataFrame

from config import settings
from src.common.logging_config import get_logger
from src.training.feature_pipeline import build_feature_stages
from src.training.tune_models import evaluate_real_scale

logger = get_logger(__name__)

_REGRESSOR_CLASSES = {
    "LinearRegression": LinearRegression,
    "DecisionTreeRegressor": DecisionTreeRegressor,
    "RandomForestRegressor": RandomForestRegressor,
    "GBTRegressor": GBTRegressor,
}


def build_regressor(model_name: str, best_params: dict, seed: int | None = None):
    """A fresh regressor instance of `model_name`, with `best_params` applied."""
    if model_name not in _REGRESSOR_CLASSES:
        raise ValueError(f"Unknown model_name '{model_name}'; expected one of {sorted(_REGRESSOR_CLASSES)}")

    regressor_cls = _REGRESSOR_CLASSES[model_name]
    kwargs = {"featuresCol": "features", "labelCol": "log_label", "predictionCol": "log_prediction"}
    if model_name != "LinearRegression":
        kwargs["seed"] = settings.RANDOM_SEED if seed is None else seed

    regressor = regressor_cls(**kwargs)
    for name, value in best_params.items():
        # Params.set() mutates in place and returns None, not self - don't reassign here.
        regressor.set(regressor.getParam(name), value)
    return regressor


def train_final_model(
    model_name: str,
    best_params: dict,
    cv_train_df: DataFrame,
    validation_df: DataFrame,
    test_df: DataFrame,
) -> tuple[PipelineModel, dict]:
    """Refit `model_name` (with `best_params`) on cv_train+validation, evaluate once on
    `test_df`. Returns (fitted PipelineModel, {"rmse":, "mae":, "r2":})."""
    train_val_df = cv_train_df.unionByName(validation_df)

    regressor = build_regressor(model_name, best_params)
    pipeline = Pipeline(stages=build_feature_stages() + [regressor])

    logger.info("Refitting %s on train+validation with best params=%s", model_name, best_params)
    final_model = pipeline.fit(train_val_df)

    predictions = final_model.transform(test_df)
    rmse, mae, r2 = evaluate_real_scale(predictions)

    logger.info(
        "FINAL test evaluation for %s (touched once): RMSE=%.2f MAE=%.2f R2=%.4f",
        model_name, rmse, mae, r2,
    )

    return final_model, {"rmse": rmse, "mae": mae, "r2": r2}

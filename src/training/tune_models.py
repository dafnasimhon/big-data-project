"""Hyperparameter tuning + candidate comparison (PLAN.md §12), fixing §23 Known Issues #1/#2.

For each of the 4 required candidates:
  1. Wrap `feature_pipeline` stages + the regressor in one `Pipeline` (indexers/encoders/
     imputer/vocab are fit fresh per candidate, inside the CV folds, so nothing leaks
     across models or out of `cv_train_df` — PLAN.md rule 7).
  2. Run k-fold cross-validation (§12: "CrossValidator or TrainValidationSplit on the
     training portion") on `cv_train_df` only, tuning against log-space RMSE.
  3. Evaluate the resulting best-per-model on `validation_df` — data the CV step never
     saw — reporting real-scale (post-`expm1`) RMSE/MAE/R², since §13 requires metrics in
     the original salary scale even though training targets `log_label`.

`test_df` is never referenced here — it's reserved for `train_final_model.py`'s one-time
final evaluation of the already-selected winner (§23 Known Issue #1 fix).

**Why a hand-rolled k-fold loop instead of `pyspark.ml.tuning.CrossValidator`:** found
running on the actual VM (2026-08-11) — `CrossValidator` fits each fold from a background
thread (even at its default `parallelism=1`), and that thread needs its own fresh py4j
socket connection back to the JVM gateway. In the Jupyter kernel environment this project
was run in, that reconnection intermittently failed under load
(`ConnectionRefusedError`, reproduced twice, including mid-`RandomForestRegressor`)
while every same-thread Spark call worked reliably throughout. `cross_validate()` below
does the identical fold-splitting/fit/evaluate/refit-on-all-data work as
`CrossValidator`, entirely on the calling thread — since `parallelism=1` meant
`CrossValidator` never actually ran folds concurrently anyway, this is not a performance
regression, just a strictly more robust way to get the same result.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from config import settings
from src.common.logging_config import get_logger
from src.common.spark_utils import reverse_log1p_predictions
from src.training.feature_pipeline import build_feature_stages
from src.training.model_candidates import ModelCandidate, build_candidates

logger = get_logger(__name__)

NUM_FOLDS = 2


@dataclass
class TuningResult:
    model_name: str
    best_params: dict
    validation_rmse: float
    validation_mae: float
    validation_r2: float
    training_time_seconds: float
    pipeline_model: object = field(repr=False)


def extract_best_params(pipeline_model, param_grid: list) -> dict:
    """Pull just the grid-tuned hyperparameter values off the fitted regressor stage."""
    regressor_stage = pipeline_model.stages[-1]
    tuned_param_names = {param.name for combo in param_grid for param in combo}
    return {
        name: regressor_stage.getOrDefault(regressor_stage.getParam(name))
        for name in tuned_param_names
    }


def evaluate_real_scale(predictions: DataFrame) -> tuple[float, float, float]:
    """Reverse the log1p target transform and compute RMSE/MAE/R² in real salary units
    (§13). Shared by both validation-time comparison (this module) and the one-time final
    test evaluation (`train_final_model.py`)."""
    predictions = reverse_log1p_predictions(predictions)
    rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse").evaluate(
        predictions
    )
    mae = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae").evaluate(
        predictions
    )
    r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2").evaluate(
        predictions
    )
    return rmse, mae, r2


def cross_validate(
    stages_factory,
    param_grid: list,
    evaluator: RegressionEvaluator,
    data: DataFrame,
    num_folds: int = NUM_FOLDS,
    seed: int = 42,
):
    """Hand-rolled equivalent of `pyspark.ml.tuning.CrossValidator` — see module
    docstring for why. `stages_factory` is a zero-arg callable returning a fresh list of
    Pipeline stages (feature stages + regressor) for each fit call.

    Returns (best_model, best_params_map): `best_model` is fit on the *entire* `data`
    using the best-found params (matching `CrossValidatorModel.bestModel`'s behavior).
    """
    fold_dfs = data.randomSplit([1.0] * num_folds, seed=seed)

    avg_metrics = []
    for params in param_grid:
        fold_metrics = []
        for held_out_index in range(num_folds):
            train_fold = None
            for fold_index, fold_df in enumerate(fold_dfs):
                if fold_index == held_out_index:
                    continue
                train_fold = fold_df if train_fold is None else train_fold.unionByName(fold_df)

            model = Pipeline(stages=stages_factory()).fit(train_fold, params)
            metric = evaluator.evaluate(model.transform(fold_dfs[held_out_index]))
            fold_metrics.append(metric)

        avg_metrics.append(sum(fold_metrics) / len(fold_metrics))

    if evaluator.isLargerBetter():
        best_index = max(range(len(avg_metrics)), key=lambda i: avg_metrics[i])
    else:
        best_index = min(range(len(avg_metrics)), key=lambda i: avg_metrics[i])
    best_params = param_grid[best_index]

    best_model = Pipeline(stages=stages_factory()).fit(data, best_params)
    return best_model, best_params


def tune_candidate(candidate: ModelCandidate, cv_train_df: DataFrame, validation_df: DataFrame) -> TuningResult:
    evaluator = RegressionEvaluator(labelCol="log_label", predictionCol="log_prediction", metricName="rmse")

    def stages_factory():
        return build_feature_stages() + [candidate.regressor]

    logger.info("Tuning %s over %d parameter combinations", candidate.name, len(candidate.param_grid))
    started_at = time.time()
    best_model, _best_params_map = cross_validate(
        stages_factory, candidate.param_grid, evaluator, cv_train_df, seed=settings.RANDOM_SEED
    )
    training_time_seconds = time.time() - started_at

    best_params = extract_best_params(best_model, candidate.param_grid)
    predictions = best_model.transform(validation_df)
    rmse, mae, r2 = evaluate_real_scale(predictions)

    logger.info(
        "%s: validation RMSE=%.2f MAE=%.2f R2=%.4f (%.1fs, best params=%s)",
        candidate.name, rmse, mae, r2, training_time_seconds, best_params,
    )

    return TuningResult(
        model_name=candidate.name,
        best_params=best_params,
        validation_rmse=rmse,
        validation_mae=mae,
        validation_r2=r2,
        training_time_seconds=training_time_seconds,
        pipeline_model=best_model,
    )


def evaluate_mean_baseline(cv_train_df: DataFrame, validation_df: DataFrame) -> dict:
    """Naive mean-prediction baseline (§13), for comparison against every tuned model."""
    mean_label = cv_train_df.agg(F.avg("label")).first()[0]
    baseline_predictions = validation_df.withColumn("prediction", F.lit(mean_label))

    rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse").evaluate(
        baseline_predictions
    )
    mae = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae").evaluate(
        baseline_predictions
    )
    r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2").evaluate(
        baseline_predictions
    )
    return {"mean_prediction": mean_label, "rmse": rmse, "mae": mae, "r2": r2}


def tune_all_candidates(cv_train_df: DataFrame, validation_df: DataFrame) -> list[TuningResult]:
    return [tune_candidate(candidate, cv_train_df, validation_df) for candidate in build_candidates()]

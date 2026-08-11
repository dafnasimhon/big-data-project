"""Hyperparameter tuning + candidate comparison (PLAN.md §12), fixing §23 Known Issues #1/#2.

For each of the 4 required candidates:
  1. Wrap `feature_pipeline` stages + the regressor in one `Pipeline` (indexers/encoders/
     imputer/vocab are fit fresh per candidate, inside the CV folds, so nothing leaks
     across models or out of `cv_train_df` — PLAN.md rule 7).
  2. Run `CrossValidator` (§12: "CrossValidator or TrainValidationSplit on the training
     portion") on `cv_train_df` only, tuning against log-space RMSE.
  3. Evaluate the resulting best-per-model on `validation_df` — data CrossValidator never
     saw — reporting real-scale (post-`expm1`) RMSE/MAE/R², since §13 requires metrics in
     the original salary scale even though training targets `log_label`.

`test_df` is never referenced here — it's reserved for `train_final_model.py`'s one-time
final evaluation of the already-selected winner (§23 Known Issue #1 fix).
"""

import time
from dataclasses import dataclass, field

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from config import settings
from src.common.logging_config import get_logger
from src.common.spark_utils import reverse_log1p_predictions
from src.training.feature_pipeline import build_feature_stages
from src.training.model_candidates import ModelCandidate, build_candidates

logger = get_logger(__name__)


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


def tune_candidate(candidate: ModelCandidate, cv_train_df: DataFrame, validation_df: DataFrame) -> TuningResult:
    pipeline = Pipeline(stages=build_feature_stages() + [candidate.regressor])
    cross_validator = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=candidate.param_grid,
        evaluator=RegressionEvaluator(labelCol="log_label", predictionCol="log_prediction", metricName="rmse"),
        numFolds=3,
        seed=settings.RANDOM_SEED,
    )

    logger.info("Tuning %s over %d parameter combinations", candidate.name, len(candidate.param_grid))
    started_at = time.time()
    cv_model = cross_validator.fit(cv_train_df)
    training_time_seconds = time.time() - started_at

    best_params = extract_best_params(cv_model.bestModel, candidate.param_grid)
    predictions = cv_model.bestModel.transform(validation_df)
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
        pipeline_model=cv_model.bestModel,
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

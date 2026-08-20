
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.util import inheritable_thread_target

from config import settings
from src.common.logging_config import get_logger
from src.common.spark_utils import reverse_log1p_predictions
from src.training.feature_pipeline import build_feature_stages
from src.training.model_candidates import ModelCandidate, build_candidates

logger = get_logger(__name__)

NUM_FOLDS = 3


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
    regressor_stage = pipeline_model.stages[-1]
    tuned_param_names = {param.name for combo in param_grid for param in combo}
    return {
        name: regressor_stage.getOrDefault(regressor_stage.getParam(name))
        for name in tuned_param_names
    }


def evaluate_real_scale(predictions: DataFrame) -> tuple[float, float, float]:
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
    parallelism: int | None = None,
):
    parallelism = settings.TUNING_PARALLELISM if parallelism is None else parallelism

    data = data.repartition(settings.SPARK_SHUFFLE_PARTITIONS)
    fold_dfs = data.randomSplit([1.0] * num_folds, seed=seed)

    fold_splits = []
    for held_out_index in range(num_folds):
        train_fold = None
        for fold_index, fold_df in enumerate(fold_dfs):
            if fold_index == held_out_index:
                continue
            train_fold = fold_df if train_fold is None else train_fold.unionByName(fold_df)
        held_out_fold = fold_dfs[held_out_index].repartition(settings.SPARK_SHUFFLE_PARTITIONS)
        train_fold = train_fold.repartition(settings.SPARK_SHUFFLE_PARTITIONS)
        fold_splits.append((train_fold, held_out_fold))

    def fit_and_evaluate(param_index: int, fold_index: int) -> tuple[int, float]:
        train_fold, held_out_fold = fold_splits[fold_index]
        model = Pipeline(stages=stages_factory()).fit(train_fold, param_grid[param_index])
        metric = evaluator.evaluate(model.transform(held_out_fold))
        return param_index, metric

    tasks = [
        (param_index, fold_index)
        for param_index in range(len(param_grid))
        for fold_index in range(num_folds)
    ]

    metrics_by_param: list[list[float]] = [[] for _ in param_grid]

    if parallelism <= 1:
        for param_index, fold_index in tasks:
            resolved_index, metric = fit_and_evaluate(param_index, fold_index)
            metrics_by_param[resolved_index].append(metric)
    else:
        wrapped_fit = inheritable_thread_target(fit_and_evaluate)
        with ThreadPoolExecutor(max_workers=parallelism) as executor:
            futures = [executor.submit(wrapped_fit, param_index, fold_index) for param_index, fold_index in tasks]
            for future in as_completed(futures):
                resolved_index, metric = future.result()
                metrics_by_param[resolved_index].append(metric)

    avg_metrics = [sum(metrics) / len(metrics) for metrics in metrics_by_param]

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

    logger.info(
        "Tuning %s over %d parameter combinations x %d folds (parallelism=%d)",
        candidate.name, len(candidate.param_grid), NUM_FOLDS, settings.TUNING_PARALLELISM,
    )
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

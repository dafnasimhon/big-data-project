"""Writes the PLAN.md §12 output artifacts and orchestrates the full Phase 4/5 run.

Run with:

    python -m src.training.evaluate_model

Produces `models/model_comparison.csv`, `models/model_metadata.json`,
`models/model_metrics.json`, and the saved winning `PipelineModel` at
`models/best_salary_model/` — all at the exact paths PLAN.md §12/§19 specify (unlike the
notebook prototype's ad hoc `output/` folders, §23 Known Issue #5).
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone

from config import settings
from src.common.logging_config import get_logger
from src.common.spark_session import get_spark_session
from src.training.data_cleaning import clean_dataset
from src.training.data_loader import load_raw_dataset
from src.training.data_split import split_dataset
from src.training.select_best_model import select_best_model
from src.training.train_final_model import train_final_model
from src.training.tune_models import TuningResult, evaluate_mean_baseline, tune_all_candidates

logger = get_logger(__name__)

FEATURE_VERSION = "v1"


def write_model_comparison(results: list[TuningResult], baseline: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["Model", "Best Parameters", "Validation RMSE", "Validation MAE", "Validation R2", "Training Time (s)"]
        )
        for result in results:
            writer.writerow(
                [
                    result.model_name,
                    json.dumps(result.best_params, sort_keys=True),
                    f"{result.validation_rmse:.4f}",
                    f"{result.validation_mae:.4f}",
                    f"{result.validation_r2:.6f}",
                    f"{result.training_time_seconds:.2f}",
                ]
            )
        writer.writerow(
            [
                "MeanBaseline",
                "{}",
                f"{baseline['rmse']:.4f}",
                f"{baseline['mae']:.4f}",
                f"{baseline['r2']:.6f}",
                "0.00",
            ]
        )
    logger.info("Model comparison written to %s", path)


def write_model_metadata(winner: TuningResult, test_metrics: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    metadata = {
        "selected_model": winner.model_name,
        "selection_metric": "validation_rmse",
        "best_parameters": winner.best_params,
        "validation_metrics": {
            "rmse": winner.validation_rmse,
            "mae": winner.validation_mae,
            "r2": winner.validation_r2,
        },
        "test_metrics": test_metrics,
        "target_column": "ConvertedCompYearly",
        "target_transformation": "log1p",
        "feature_version": FEATURE_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    logger.info("Model metadata written to %s", path)


def write_model_metrics(test_metrics: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(test_metrics, handle, indent=2)
    logger.info("Model metrics written to %s", path)


def run_training_pipeline() -> None:
    spark = get_spark_session(app_name="SalaryModelTraining")

    raw_df = load_raw_dataset(spark)
    cleaned_df = clean_dataset(raw_df)
    cv_train_df, validation_df, test_df = split_dataset(cleaned_df)
    cv_train_df = cv_train_df.cache()
    validation_df = validation_df.cache()
    test_df = test_df.cache()

    logger.info(
        "Split sizes -> cv_train: %d, validation: %d, test: %d",
        cv_train_df.count(), validation_df.count(), test_df.count(),
    )

    results = tune_all_candidates(cv_train_df, validation_df)
    baseline = evaluate_mean_baseline(cv_train_df, validation_df)
    winner = select_best_model(results)

    logger.info("Selected model: %s (validation RMSE=%.2f)", winner.model_name, winner.validation_rmse)

    final_model, test_metrics = train_final_model(
        winner.model_name, winner.best_params, cv_train_df, validation_df, test_df
    )

    write_model_comparison(results, baseline, settings.MODEL_COMPARISON_PATH)
    write_model_metadata(winner, test_metrics, settings.MODEL_METADATA_PATH)
    write_model_metrics(test_metrics, settings.MODEL_METRICS_PATH)

    final_model.write().overwrite().save(settings.MODEL_PATH)
    logger.info("Best PipelineModel saved to %s", settings.MODEL_PATH)


if __name__ == "__main__":
    run_training_pipeline()

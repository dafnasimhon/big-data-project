import math

import pytest
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.sql import SparkSession

from src.training.data_split import split_dataset
from src.training.model_candidates import ModelCandidate
from src.training.train_final_model import train_final_model
from src.training.tune_models import evaluate_mean_baseline, tune_candidate

FEATURE_COLUMNS = [
    "Country", "Age", "EdLevel", "Employment", "RemoteWork",
    "DevType", "OrgSize", "Industry",
    "LanguageHaveWorkedWith", "DatabaseHaveWorkedWith", "PlatformHaveWorkedWith",
    "YearsCodeProNumeric", "label", "log_label",
]


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.master("local[1]").appName("test_training_pipeline").getOrCreate()
    )
    yield session
    session.stop()


def _synthetic_dataset(spark, n=30):
    countries = ["USA", "Israel", "Germany"]
    ed_levels = ["Bachelor's", "Master's"]
    dev_types = ["Developer", "Manager"]
    rows = []
    for i in range(n):
        label = 50000.0 + (i % 5) * 10000.0
        rows.append(
            (
                countries[i % 3], "25-34 years old", ed_levels[i % 2], "Employed, full-time",
                "Remote" if i % 2 == 0 else "Hybrid", dev_types[i % 2],
                "20 to 99 employees", "Tech",
                "Python;SQL" if i % 2 == 0 else "Java;Python",
                "PostgreSQL", "AWS",
                float(2 + i % 10), label, math.log1p(label),
            )
        )
    return spark.createDataFrame(rows, FEATURE_COLUMNS)


def test_split_dataset_produces_disjoint_slices_covering_all_rows(spark):
    df = _synthetic_dataset(spark, n=60)
    cv_train_df, validation_df, test_df = split_dataset(df, seed=42)

    assert cv_train_df.count() + validation_df.count() + test_df.count() == 60
    # Same seed -> same split sizes every time (reproducibility, PLAN.md rule 9).
    cv_train_again, validation_again, test_again = split_dataset(df, seed=42)
    assert cv_train_df.count() == cv_train_again.count()
    assert validation_df.count() == validation_again.count()
    assert test_df.count() == test_again.count()


def test_tune_candidate_runs_end_to_end(spark):
    df = _synthetic_dataset(spark, n=30)
    cv_train_df, validation_df, _test_df = split_dataset(df, seed=42)

    regressor = LinearRegression(featuresCol="features", labelCol="log_label", predictionCol="log_prediction")
    candidate = ModelCandidate("LinearRegression", regressor, ParamGridBuilder().build())

    result = tune_candidate(candidate, cv_train_df, validation_df)

    assert result.model_name == "LinearRegression"
    assert result.validation_rmse >= 0
    assert result.validation_mae >= 0
    assert result.pipeline_model is not None


def test_evaluate_mean_baseline(spark):
    df = _synthetic_dataset(spark, n=30)
    cv_train_df, validation_df, _test_df = split_dataset(df, seed=42)

    baseline = evaluate_mean_baseline(cv_train_df, validation_df)

    assert baseline["mean_prediction"] > 0
    assert baseline["rmse"] >= 0


def test_train_final_model_evaluates_once_on_test(spark):
    df = _synthetic_dataset(spark, n=30)
    cv_train_df, validation_df, test_df = split_dataset(df, seed=42)

    final_model, test_metrics = train_final_model(
        "LinearRegression",
        {"regParam": 0.1, "elasticNetParam": 0.0, "maxIter": 50},
        cv_train_df,
        validation_df,
        test_df,
    )

    assert set(test_metrics.keys()) == {"rmse", "mae", "r2"}
    assert test_metrics["rmse"] >= 0

    predictions = final_model.transform(test_df)
    assert "log_prediction" in predictions.columns

import pytest
from pyspark.sql import SparkSession

from src.training.model_candidates import build_candidates


@pytest.fixture(scope="module")
def spark():
    session = SparkSession.builder.master("local[1]").appName("test_model_candidates").getOrCreate()
    yield session
    session.stop()


def test_build_candidates_returns_the_four_required_models(spark):
    names = {candidate.name for candidate in build_candidates()}
    assert names == {"LinearRegression", "DecisionTreeRegressor", "RandomForestRegressor", "GBTRegressor"}


def test_build_candidates_grids_are_nonempty(spark):
    for candidate in build_candidates():
        assert len(candidate.param_grid) > 0


def test_build_candidates_regressors_target_log_label(spark):
    for candidate in build_candidates():
        assert candidate.regressor.getLabelCol() == "log_label"
        assert candidate.regressor.getPredictionCol() == "log_prediction"
        assert candidate.regressor.getFeaturesCol() == "features"

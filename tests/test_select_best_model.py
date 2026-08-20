import pytest

from src.training.select_best_model import select_best_model
from src.training.tune_models import TuningResult


def _result(name, rmse, mae, r2):
    return TuningResult(
        model_name=name,
        best_params={},
        validation_rmse=rmse,
        validation_mae=mae,
        validation_r2=r2,
        training_time_seconds=0.0,
        pipeline_model=None,
    )


def test_select_best_model_picks_lowest_rmse():
    results = [_result("A", 100.0, 50.0, 0.5), _result("B", 90.0, 60.0, 0.4), _result("C", 95.0, 40.0, 0.6)]
    assert select_best_model(results).model_name == "B"


def test_select_best_model_tie_breaks_on_mae():
    results = [_result("A", 100.0, 50.0, 0.5), _result("B", 100.00000005, 40.0, 0.4)]
    assert select_best_model(results).model_name == "B"


def test_select_best_model_tie_breaks_on_r2():
    results = [_result("A", 100.0, 50.0, 0.5), _result("B", 100.00000005, 50.00000005, 0.7)]
    assert select_best_model(results).model_name == "B"


def test_select_best_model_single_result():
    assert select_best_model([_result("Only", 10.0, 5.0, 0.9)]).model_name == "Only"


def test_select_best_model_empty_raises():
    with pytest.raises(ValueError):
        select_best_model([])

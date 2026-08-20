
from __future__ import annotations

from src.training.tune_models import TuningResult

RELATIVE_TIE_TOLERANCE = 1e-6


def _is_close(a: float, b: float, tolerance: float = RELATIVE_TIE_TOLERANCE) -> bool:
    scale = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / scale <= tolerance


def select_best_model(results: list[TuningResult]) -> TuningResult:
    if not results:
        raise ValueError("select_best_model requires at least one tuning result")

    best = results[0]
    for candidate in results[1:]:
        if not _is_close(candidate.validation_rmse, best.validation_rmse):
            if candidate.validation_rmse < best.validation_rmse:
                best = candidate
            continue

        if not _is_close(candidate.validation_mae, best.validation_mae):
            if candidate.validation_mae < best.validation_mae:
                best = candidate
            continue

        if candidate.validation_r2 > best.validation_r2:
            best = candidate

    return best

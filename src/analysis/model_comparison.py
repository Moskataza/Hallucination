from __future__ import annotations

from typing import Any


def compare_models(model_a: dict[str, Any], model_b: dict[str, Any], metric_names: tuple[str, ...]) -> dict[str, float]:
    return {f"delta_{metric}": float(model_a.get(metric, 0.0)) - float(model_b.get(metric, 0.0)) for metric in metric_names}

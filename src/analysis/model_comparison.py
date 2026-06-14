"""计算两个模型在同一组指标上的差值。"""

from __future__ import annotations

from typing import Any


def compare_models(model_a: dict[str, Any], model_b: dict[str, Any], metric_names: tuple[str, ...]) -> dict[str, float]:
    """返回模型 A 指标减模型 B 指标的差值。"""
    return {f"delta_{metric}": float(model_a.get(metric, 0.0)) - float(model_b.get(metric, 0.0)) for metric in metric_names}

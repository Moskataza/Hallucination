"""计算 CoT 提示相对 direct 提示的指标变化。"""

from __future__ import annotations

from typing import Any


def compute_cot_delta(direct_metrics: dict[str, Any], cot_metrics: dict[str, Any], metric_names: tuple[str, ...]) -> dict[str, float]:
    """计算 CoT 指标减 direct 指标的差值。"""
    deltas: dict[str, float] = {}
    for metric in metric_names:
        deltas[f"delta_{metric}"] = float(cot_metrics.get(metric, 0.0)) - float(direct_metrics.get(metric, 0.0))
    return deltas

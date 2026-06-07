from __future__ import annotations

from math import sqrt


def cohens_kappa(tp: int, fp: int, tn: int, fn: int) -> float:
    total = tp + fp + tn + fn
    if total == 0:
        return 0.0
    observed = (tp + tn) / total
    pred_pos = (tp + fp) / total
    pred_neg = (tn + fn) / total
    human_pos = (tp + fn) / total
    human_neg = (tn + fp) / total
    expected = pred_pos * human_pos + pred_neg * human_neg
    if expected == 1:
        return 0.0
    return (observed - expected) / (1 - expected)


def matthews_corrcoef(tp: int, fp: int, tn: int, fn: int) -> float:
    denominator = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        return 0.0
    return (tp * tn - fp * fn) / denominator

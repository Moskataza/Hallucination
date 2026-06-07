from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def compute_binary_alignment(rows: Iterable[dict[str, Any]], predicted_key: str = "predicted_is_hallucination", human_key: str = "human_is_hallucination") -> dict[str, float | int]:
    tp = fp = tn = fn = skipped = 0
    for row in rows:
        predicted = _to_bool_or_none(row.get(predicted_key))
        human = _to_bool_or_none(row.get(human_key))
        if predicted is None or human is None:
            skipped += 1
            continue
        if predicted and human:
            tp += 1
        elif predicted and not human:
            fp += 1
        elif not predicted and human:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / (tp + fp + tn + fn) if tp + fp + tn + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "skipped": skipped,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def _to_bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"yes", "true", "1", "hallucination", "hallucinated"}:
        return True
    if normalized in {"no", "false", "0", "none", "non-hallucination", "not hallucinated"}:
        return False
    return None

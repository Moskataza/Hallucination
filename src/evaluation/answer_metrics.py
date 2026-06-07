from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from src.models.response_parser import normalize_yes_no


def compute_yes_no_accuracy(rows: Iterable[dict[str, Any]]) -> dict[str, float | int]:
    total = 0
    correct = 0
    unclear = 0
    for row in rows:
        reference = normalize_yes_no(str(row.get("reference_answer", "")))
        prediction = normalize_yes_no(str(row.get("final_answer", row.get("prediction", ""))))
        if reference == "unclear" or prediction == "unclear":
            unclear += 1
            continue
        total += 1
        correct += int(reference == prediction)
    return {
        "scored_count": total,
        "correct_count": correct,
        "unclear_count": unclear,
        "accuracy": correct / total if total else 0.0,
    }


def exact_match(reference: str, prediction: str) -> bool:
    return _normalize_text(reference) == _normalize_text(prediction)


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())

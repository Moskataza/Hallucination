from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from src.detectors.taxonomy import FACTUAL_TYPES, LOGICAL_TYPES, normalize_fine_label


def compute_hallucination_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    total = 0
    hallucinated = 0
    factual = 0
    logical = 0
    unsupported_visual = 0
    answer_correct_and_grounded = 0
    fine_counts: Counter[str] = Counter()

    for row in rows:
        total += 1
        is_hallucination = row.get("is_hallucination")
        labels = _row_fine_labels(row)
        answer_correct = row.get("answer_correct")
        unsupported = row.get("unsupported_visual_claim")

        if is_hallucination is True:
            hallucinated += 1
        if labels & FACTUAL_TYPES:
            factual += 1
        if labels & LOGICAL_TYPES:
            logical += 1
        fine_counts.update(labels)
        if unsupported is True:
            unsupported_visual += 1
        if (
            answer_correct is True
            and is_hallucination is False
            and unsupported is not True
        ):
            answer_correct_and_grounded += 1

    return {
        "count": total,
        "hallucination_count": hallucinated,
        "hallucination_rate": hallucinated / total if total else 0.0,
        "factual_count": factual,
        "factual_rate": factual / total if total else 0.0,
        "logical_count": logical,
        "logical_rate": logical / total if total else 0.0,
        "unsupported_visual_claim_count": unsupported_visual,
        "unsupported_visual_claim_rate": unsupported_visual / total if total else 0.0,
        "grounded_accuracy_count": answer_correct_and_grounded,
        "grounded_accuracy": answer_correct_and_grounded / total if total else 0.0,
        "fine_type_counts": dict(fine_counts),
    }


def _row_fine_labels(row: dict[str, Any]) -> set[str]:
    details = row.get("details") or {}
    raw_labels = details.get("hallucination_labels")
    if isinstance(raw_labels, list):
        labels = {
            normalized
            for raw_label in raw_labels
            if (normalized := normalize_fine_label(str(raw_label)))
            not in {"None", "Unclear"}
        }
        if labels:
            return labels

    taxonomy = row.get("taxonomy") or {}
    fine = normalize_fine_label(str(taxonomy.get("fine", "None")))
    if fine in {"None", "Unclear"}:
        return set()
    return {fine}

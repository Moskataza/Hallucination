"""计算回答正确性相关的基础指标。"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from src.models.response_parser import normalize_yes_no


def compute_yes_no_accuracy(rows: Iterable[dict[str, Any]]) -> dict[str, float | int]:
    """仅在参考答案和预测都可归一为 yes/no 时计算准确率。"""
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
    """使用大小写和空白规范化后的精确匹配。"""
    return _normalize_text(reference) == _normalize_text(prediction)


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())

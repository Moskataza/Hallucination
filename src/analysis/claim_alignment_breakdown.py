"""按数据集、prompt、细粒度标签等维度拆解 Human-as-Judge 对齐结果。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.datasets.jsonl import read_json_records
from src.evaluation.agreement import cohens_kappa, matthews_corrcoef
from src.evaluation.evaluate_human_alignment import join_annotations_with_key
from src.evaluation.human_alignment import compute_binary_alignment

_FINE_LABELS = ("OBJ", "ATT", "SPA", "IR", "CI", "INC", "SO")
_ALIGNMENT_COLUMNS = [
    "breakdown",
    "value",
    "count",
    "evaluated_count",
    "skipped",
    "tp",
    "fp",
    "tn",
    "fn",
    "precision",
    "recall",
    "f1",
    "accuracy",
    "cohens_kappa",
    "matthews_corrcoef",
    "predicted_hallucination_rate",
    "human_hallucination_rate",
]


def build_breakdown_tables(
    annotations: list[dict[str, Any]], key_rows: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """构建按不同维度拆分的人工标注对齐表。"""
    joined = join_annotations_with_key(annotations, key_rows)
    return {
        "dataset_breakdown": _single_dimension_rows(joined, "dataset"),
        "prompt_breakdown": _single_dimension_rows(joined, "prompt_type"),
        "dataset_prompt_breakdown": _multi_dimension_rows(
            joined, ("dataset", "prompt_type")
        ),
        "fine_label_breakdown": _fine_label_rows(joined),
        "confidence_breakdown": _single_dimension_rows(joined, "detector_confidence"),
    }


def write_breakdown_tables(
    annotations: list[dict[str, Any]],
    key_rows: list[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, Path]:
    """写出所有拆分表，供后续报告或论文分析使用。"""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tables = build_breakdown_tables(annotations, key_rows)
    paths = {}
    for name, rows in tables.items():
        path = output / f"{name}.csv"
        _write_csv(path, rows)
        paths[name] = path
    return paths


def _single_dimension_rows(
    rows: list[dict[str, Any]], key: str
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, ""))].append(row)
    return [
        _alignment_row(group_rows, key, value)
        for value, group_rows in sorted(grouped.items())
    ]


def _multi_dimension_rows(
    rows: list[dict[str, Any]], keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(key, "")) for key in keys)].append(row)
    breakdown = "+".join(keys)
    return [
        _alignment_row(group_rows, breakdown, " / ".join(values))
        for values, group_rows in sorted(grouped.items())
    ]


def _fine_label_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table = []
    for label in _FINE_LABELS:
        label_rows = [
            {
                "predicted_is_hallucination": _detector_has_label(row, label),
                "human_is_hallucination": _human_has_fine_label(row, label),
            }
            for row in rows
        ]
        table.append(_alignment_row(label_rows, "fine_label", label))
    return table


def _detector_has_label(row: dict[str, Any], label: str) -> bool:
    labels = _detector_fine_labels(row)
    return label in labels


def _detector_fine_labels(row: dict[str, Any]) -> set[str]:
    """从规范化字段、原始 judge JSON 或 taxonomy 中恢复 detector 细粒度标签。"""
    normalized_labels = row.get("detector_hallucination_labels")
    if isinstance(normalized_labels, list):
        return {str(label) for label in normalized_labels if str(label) in _FINE_LABELS}

    raw_labels = _raw_hallucination_labels(row.get("detector_raw_judge_response"))
    if raw_labels:
        return raw_labels

    detector_taxonomy = row.get("detector_taxonomy")
    if isinstance(detector_taxonomy, dict):
        fine = detector_taxonomy.get("fine")
        if isinstance(fine, list):
            return {str(item) for item in fine if str(item) in _FINE_LABELS}
        if str(fine) in _FINE_LABELS:
            return {str(fine)}
    return set()


def _raw_hallucination_labels(raw_response: Any) -> set[str]:
    if not raw_response:
        return set()
    try:
        payload = json.loads(str(raw_response))
    except json.JSONDecodeError:
        return set()
    labels = payload.get("hallucination_labels") if isinstance(payload, dict) else None
    if not isinstance(labels, list):
        return set()
    return {str(label) for label in labels if str(label) in _FINE_LABELS}


def _alignment_row(
    rows: list[dict[str, Any]], breakdown: str, value: str
) -> dict[str, Any]:
    """把一个子集转换为混淆矩阵、Kappa 和 MCC 等对齐指标。"""
    metrics = compute_binary_alignment(rows)
    tp = int(metrics["tp"])
    fp = int(metrics["fp"])
    tn = int(metrics["tn"])
    fn = int(metrics["fn"])
    evaluated = tp + fp + tn + fn
    predicted_positive = tp + fp
    human_positive = tp + fn
    return {
        "breakdown": breakdown,
        "value": value,
        "count": len(rows),
        "evaluated_count": evaluated,
        "skipped": metrics["skipped"],
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "accuracy": metrics["accuracy"],
        "cohens_kappa": cohens_kappa(tp, fp, tn, fn),
        "matthews_corrcoef": matthews_corrcoef(tp, fp, tn, fn),
        "predicted_hallucination_rate": (
            predicted_positive / evaluated if evaluated else 0.0
        ),
        "human_hallucination_rate": human_positive / evaluated if evaluated else 0.0,
    }


def _human_has_fine_label(row: dict[str, Any], label: str) -> bool | None:
    human_label = _to_bool_or_none(row.get("human_is_hallucination"))
    if human_label is None:
        return None
    return _truthy_label(row.get(label))


def _truthy_label(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _to_bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"yes", "true", "1", "hallucination", "hallucinated"}:
        return True
    if normalized in {
        "no",
        "false",
        "0",
        "none",
        "non-hallucination",
        "not hallucinated",
    }:
        return False
    return None


def _read_csv(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            with input_path.open("r", encoding=encoding, newline="") as file:
                return list(csv.DictReader(file))
        except UnicodeDecodeError:
            continue
    with input_path.open("r", encoding="utf-8", errors="replace", newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=_ALIGNMENT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Human-as-Judge alignment breakdown tables."
    )
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument(
        "--output-dir", default="outputs/metrics/human_alignment/one_tenth_zero_shot_v2"
    )
    args = parser.parse_args()

    annotations = _read_csv(args.annotations)
    key_rows = list(read_json_records(args.key))
    paths = write_breakdown_tables(annotations, key_rows, args.output_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

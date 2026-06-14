"""汇总 detector JSONL 结果，生成总体指标、CoT 效应和模型对比表。"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.datasets.jsonl import read_json_records
from src.evaluation.hallucination_metrics import compute_hallucination_metrics

_GROUP_KEYS = ("detector", "dataset", "model", "prompt_type")
_METRIC_NAMES = (
    "hallucination_rate",
    "factual_rate",
    "logical_rate",
    "unsupported_visual_claim_rate",
    "grounded_accuracy",
)
_FINE_LABELS = ("OBJ", "ATT", "SPA", "IR", "CI", "INC", "SO")
_BASE_COLUMNS = ["detector", "dataset", "model", "prompt_type"]
_OVERALL_COLUMNS = [
    *_BASE_COLUMNS,
    "count",
    "hallucination_count",
    "hallucination_rate",
    "factual_count",
    "factual_rate",
    "logical_count",
    "logical_rate",
    "unsupported_visual_claim_count",
    "unsupported_visual_claim_rate",
    "grounded_accuracy_count",
    "grounded_accuracy",
    *[item for label in _FINE_LABELS for item in (f"{label}_count", f"{label}_rate")],
]
_COT_COLUMNS = [
    "detector",
    "dataset",
    "model",
    *[f"delta_{metric}" for metric in _METRIC_NAMES],
]
_MODEL_COMPARISON_COLUMNS = [
    "detector",
    "dataset",
    "prompt_type",
    "model_a",
    "model_b",
    *[f"delta_{metric}" for metric in _METRIC_NAMES],
]
_TAXONOMY_COLUMNS = [
    *_BASE_COLUMNS,
    "count",
    "factual_count",
    "logical_count",
    "factual_rate",
    "logical_rate",
    *[item for label in _FINE_LABELS for item in (f"{label}_count", f"{label}_rate")],
]
_TABLE_COLUMNS = {
    "overall_results": _OVERALL_COLUMNS,
    "cot_effect": _COT_COLUMNS,
    "model_comparison": _MODEL_COMPARISON_COLUMNS,
    "taxonomy_distribution": _TAXONOMY_COLUMNS,
}


def group_detector_results(
    rows: list[dict[str, Any]],
    keys: tuple[str, ...] = _GROUP_KEYS,
) -> dict[tuple[str, ...], dict[str, Any]]:
    """按 detector、数据集、模型和 prompt 分组后计算幻觉指标。"""
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        # 分组键缺失会导致跨实验结果混合，因此直接失败而不是静默归入空组。
        missing_keys = [
            key for key in keys if key not in row or row.get(key) in (None, "")
        ]
        if missing_keys:
            raise ValueError(
                f"Detector result missing grouping keys {missing_keys}: {row.get('sample_id', '<unknown>')}"
            )
        group_key = tuple(str(row[key]) for key in keys)
        groups[group_key].append(row)
    return {
        group_key: compute_hallucination_metrics(group_rows)
        for group_key, group_rows in groups.items()
    }


def build_experiment_tables(
    rows: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """生成实验所需的总体、CoT、模型对比和 taxonomy 分布表。"""
    overall = build_overall_table(rows)
    return {
        "overall_results": overall,
        "cot_effect": build_cot_effect_table(overall),
        "model_comparison": build_model_comparison_table(overall),
        "taxonomy_distribution": build_taxonomy_distribution_table(overall),
    }


def build_overall_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把每个实验组的指标展开成一行 CSV 友好的记录。"""
    grouped = group_detector_results(rows)
    table = []
    for (detector, dataset, model, prompt_type), metrics in sorted(grouped.items()):
        row: dict[str, Any] = {
            "detector": detector,
            "dataset": dataset,
            "model": model,
            "prompt_type": prompt_type,
            "count": metrics["count"],
            "hallucination_count": metrics["hallucination_count"],
            "hallucination_rate": metrics["hallucination_rate"],
            "factual_count": metrics["factual_count"],
            "factual_rate": metrics["factual_rate"],
            "logical_count": metrics["logical_count"],
            "logical_rate": metrics["logical_rate"],
            "unsupported_visual_claim_count": metrics["unsupported_visual_claim_count"],
            "unsupported_visual_claim_rate": metrics["unsupported_visual_claim_rate"],
            "grounded_accuracy_count": metrics["grounded_accuracy_count"],
            "grounded_accuracy": metrics["grounded_accuracy"],
        }
        fine_counts = metrics.get("fine_type_counts", {})
        for label in _FINE_LABELS:
            row[f"{label}_count"] = int(fine_counts.get(label, 0))
            row[f"{label}_rate"] = (
                row[f"{label}_count"] / metrics["count"] if metrics["count"] else 0.0
            )
        table.append(row)
    return table


def build_cot_effect_table(overall_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """对齐同组 direct 与 CoT 结果，计算 CoT 带来的指标增量。"""
    by_group = {
        (
            str(row["detector"]),
            str(row["dataset"]),
            str(row["model"]),
            str(row["prompt_type"]),
        ): row
        for row in overall_rows
    }
    table = []
    for detector, dataset, model, prompt_type in sorted(by_group):
        if prompt_type != "evidence_grounded_cot":
            continue
        direct = by_group.get((detector, dataset, model, "direct"))
        if direct is None:
            continue
        cot = by_group[(detector, dataset, model, prompt_type)]
        row: dict[str, Any] = {"detector": detector, "dataset": dataset, "model": model}
        for metric in _METRIC_NAMES:
            row[f"delta_{metric}"] = float(cot.get(metric, 0.0)) - float(
                direct.get(metric, 0.0)
            )
        table.append(row)
    return table


def build_model_comparison_table(
    overall_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """在相同数据集和 prompt 下两两比较模型指标差异。"""
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in overall_rows:
        groups[
            (str(row["detector"]), str(row["dataset"]), str(row["prompt_type"]))
        ].append(row)

    table = []
    for (detector, dataset, prompt_type), rows in sorted(groups.items()):
        sorted_rows = sorted(rows, key=lambda row: str(row["model"]))
        for index, left in enumerate(sorted_rows):
            for right in sorted_rows[index + 1 :]:
                comparison_row: dict[str, Any] = {
                    "detector": detector,
                    "dataset": dataset,
                    "prompt_type": prompt_type,
                    "model_a": left["model"],
                    "model_b": right["model"],
                }
                for metric in _METRIC_NAMES:
                    comparison_row[f"delta_{metric}"] = float(
                        left.get(metric, 0.0)
                    ) - float(right.get(metric, 0.0))
                table.append(comparison_row)
    return table


def build_taxonomy_distribution_table(
    overall_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """抽取 hallucination taxonomy 的粗细粒度分布。"""
    table = []
    for row in overall_rows:
        table_row: dict[str, Any] = {
            "detector": row["detector"],
            "dataset": row["dataset"],
            "model": row["model"],
            "prompt_type": row["prompt_type"],
            "count": row["count"],
            "factual_count": row["factual_count"],
            "logical_count": row["logical_count"],
            "factual_rate": row["factual_rate"],
            "logical_rate": row["logical_rate"],
        }
        for label in _FINE_LABELS:
            table_row[f"{label}_count"] = row[f"{label}_count"]
            table_row[f"{label}_rate"] = row[f"{label}_rate"]
        table.append(table_row)
    return table


def write_experiment_tables(
    rows: list[dict[str, Any]], output_dir: str | Path
) -> dict[str, Path]:
    """将所有汇总表写入输出目录。"""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, table in build_experiment_tables(rows).items():
        path = output / f"{name}.csv"
        _write_csv(path, table, _TABLE_COLUMNS[name])
        paths[name] = path
    return paths


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate detector JSONL outputs into experiment summary CSV tables."
    )
    parser.add_argument(
        "--detectors",
        nargs="+",
        required=True,
        help="Detector result JSONL paths. Multiple files are concatenated.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/metrics",
        help="Directory for summary CSV tables.",
    )
    args = parser.parse_args()

    rows = [
        row
        for detector_path in args.detectors
        for row in read_json_records(detector_path)
    ]
    write_experiment_tables(rows, args.output_dir)


if __name__ == "__main__":
    main()

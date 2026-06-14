"""分析 zero-shot judge 的 claim 级输出，评估标签、证据来源和格式质量。"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from src.datasets.jsonl import read_json_records

_GROUP_KEYS = ("detector", "dataset", "model", "prompt_type")
_CLAIM_TYPES = (
    "object_claim",
    "attribute_claim",
    "spatial_claim",
    "reasoning_claim",
    "causal_claim",
    "inconsistency_claim",
    "semantic_claim",
    "answer_claim",
    "non_claim",
)
_EVIDENCE_SOURCES = (
    "image",
    "question",
    "choices",
    "reference_answer",
    "math_rule",
    "diagram_rule",
    "internal_consistency",
    "none",
)
_SUPPORT_STATUSES = ("supported", "contradicted", "unverifiable", "not_applicable")
_TRIGGER_LABELS = ("OBJ", "ATT", "SPA", "IR", "CI", "INC", "SO")
_FINE_LABELS = (*_TRIGGER_LABELS, "None", "Unclear")
_UNSUPPORTED_STATUSES = {"contradicted", "unverifiable"}

_SUMMARY_COLUMNS = [
    *_GROUP_KEYS,
    "response_count",
    "hallucination_count",
    "response_level_hallucination_rate",
    "total_claims",
    "checkable_claims",
    "unsupported_claims",
    "label_triggering_unsupported_claims",
    "claims_per_response",
    "median_claims_per_response",
    "p90_claims_per_response",
    "checkable_claims_per_response",
    "unsupported_claims_per_response",
    "unsupported_claim_ratio",
    "label_triggering_unsupported_claims_per_response",
    "label_triggering_unsupported_claim_ratio",
    "supported_claims",
    "contradicted_claims",
    "unverifiable_claims",
    "not_applicable_claims",
]
_DISTRIBUTION_COLUMNS = [*_GROUP_KEYS, "value", "count", "share"]
_FORMAT_COLUMNS = [
    *_GROUP_KEYS,
    "response_count",
    "rows_missing_claim_checks",
    "rows_invalid_claim_checks",
    "rows_with_raw_normalized_mismatch",
    "rows_summary_inconsistent",
    "invalid_claim_type_count",
    "invalid_support_status_count",
    "invalid_evidence_source_count",
    "invalid_fine_label_count",
    "missing_evidence_source_count",
    "none_evidence_source_count",
    "weak_or_unassigned_evidence_ratio",
]
_COT_EXPOSURE_COLUMNS = [
    "detector",
    "dataset",
    "model",
    "direct_claims_per_response",
    "cot_claims_per_response",
    "delta_claims_per_response",
    "direct_unsupported_claims_per_response",
    "cot_unsupported_claims_per_response",
    "delta_unsupported_claims_per_response",
    "direct_unsupported_claim_ratio",
    "cot_unsupported_claim_ratio",
    "delta_unsupported_claim_ratio",
    "direct_label_triggering_unsupported_claim_ratio",
    "cot_label_triggering_unsupported_claim_ratio",
    "delta_label_triggering_unsupported_claim_ratio",
    "direct_response_level_hallucination_rate",
    "cot_response_level_hallucination_rate",
    "delta_response_level_hallucination_rate",
]
_TABLE_COLUMNS = {
    "claim_summary": _SUMMARY_COLUMNS,
    "claim_type_distribution": _DISTRIBUTION_COLUMNS,
    "evidence_source_distribution": _DISTRIBUTION_COLUMNS,
    "format_quality": _FORMAT_COLUMNS,
    "cot_claim_exposure": _COT_EXPOSURE_COLUMNS,
}


def build_claim_level_tables(
    rows: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """按实验组汇总 claim 级统计、分布和格式质量表。"""
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_group_key(row)].append(row)

    summaries = []
    claim_type_rows = []
    evidence_rows = []
    format_rows = []
    for key, group_rows in sorted(grouped.items()):
        group_stats = _compute_group_stats(key, group_rows)
        summaries.append(group_stats["summary"])
        format_rows.append(group_stats["format_quality"])
        claim_type_rows.extend(
            _distribution_rows(
                key, group_stats["claim_types"], group_stats["total_claims"]
            )
        )
        evidence_rows.extend(
            _distribution_rows(
                key, group_stats["evidence_sources"], group_stats["total_claims"]
            )
        )

    return {
        "claim_summary": summaries,
        "claim_type_distribution": claim_type_rows,
        "evidence_source_distribution": evidence_rows,
        "format_quality": format_rows,
        "cot_claim_exposure": _cot_exposure_rows(summaries),
    }


def write_claim_level_tables(
    rows: list[dict[str, Any]], output_dir: str | Path
) -> dict[str, Path]:
    """写出 claim 级分析的多张 CSV 表。"""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tables = build_claim_level_tables(rows)
    paths = {}
    for name, table_rows in tables.items():
        path = output / f"{name}.csv"
        _write_csv(path, table_rows, _TABLE_COLUMNS[name])
        paths[name] = path
    return paths


def _compute_group_stats(
    key: tuple[str, ...], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """在单个实验组内统计 claim 数量、支持状态和格式异常。"""
    claim_counts = []
    total_claims = 0
    checkable_claims = 0
    unsupported_claims = 0
    label_triggering_unsupported_claims = 0
    hallucination_count = 0
    status_counts: Counter[str] = Counter()
    claim_type_counts: Counter[str] = Counter()
    evidence_counts: Counter[str] = Counter()
    rows_missing_claim_checks = 0
    rows_invalid_claim_checks = 0
    rows_with_raw_normalized_mismatch = 0
    rows_summary_inconsistent = 0
    invalid_claim_type_count = 0
    invalid_support_status_count = 0
    invalid_evidence_source_count = 0
    invalid_fine_label_count = 0
    missing_evidence_source_count = 0
    none_evidence_source_count = 0

    for row in rows:
        if row.get("is_hallucination") is True:
            hallucination_count += 1

        raw_details = row.get("details")
        details: dict[str, Any] = raw_details if isinstance(raw_details, dict) else {}
        if details.get("raw_normalized_mismatch") is True:
            rows_with_raw_normalized_mismatch += 1
        if details.get("summary_consistent_with_claims") is False:
            rows_summary_inconsistent += 1

        claim_checks = details.get("claim_checks")
        if claim_checks is None:
            rows_missing_claim_checks += 1
            claim_counts.append(0)
            continue
        if not isinstance(claim_checks, list):
            rows_invalid_claim_checks += 1
            claim_counts.append(0)
            continue

        claim_counts.append(len(claim_checks))
        total_claims += len(claim_checks)
        for claim in claim_checks:
            if not isinstance(claim, dict):
                rows_invalid_claim_checks += 1
                continue

            claim_type = str(claim.get("claim_type", ""))
            support_status = _normalize_support_status(claim.get("support_status"))
            evidence_source = str(claim.get("evidence_source", ""))
            fine_label = str(claim.get("fine_label", "None"))

            claim_type_counts[claim_type or "<missing>"] += 1
            evidence_counts[evidence_source or "<missing>"] += 1
            status_counts[support_status or "<missing>"] += 1

            if claim_type not in _CLAIM_TYPES:
                invalid_claim_type_count += 1
            if support_status not in _SUPPORT_STATUSES:
                invalid_support_status_count += 1
            if not evidence_source:
                missing_evidence_source_count += 1
            elif evidence_source not in _EVIDENCE_SOURCES:
                invalid_evidence_source_count += 1
            if fine_label not in _FINE_LABELS:
                invalid_fine_label_count += 1
            if evidence_source == "none":
                none_evidence_source_count += 1

            # 只有可验证 claim 进入分母，避免 non-claim 稀释 unsupported 比例。
            if claim_type != "non_claim" and support_status != "not_applicable":
                checkable_claims += 1
            if support_status in _UNSUPPORTED_STATUSES:
                unsupported_claims += 1
                if fine_label in _TRIGGER_LABELS:
                    label_triggering_unsupported_claims += 1

    response_count = len(rows)
    weak_or_unassigned = (
        missing_evidence_source_count
        + none_evidence_source_count
        + invalid_evidence_source_count
    )
    summary = {
        **_dimensions(key),
        "response_count": response_count,
        "hallucination_count": hallucination_count,
        "response_level_hallucination_rate": _ratio(
            hallucination_count, response_count
        ),
        "total_claims": total_claims,
        "checkable_claims": checkable_claims,
        "unsupported_claims": unsupported_claims,
        "label_triggering_unsupported_claims": label_triggering_unsupported_claims,
        "claims_per_response": _ratio(total_claims, response_count),
        "median_claims_per_response": median(claim_counts) if claim_counts else 0,
        "p90_claims_per_response": _percentile_nearest_rank(claim_counts, 0.9),
        "checkable_claims_per_response": _ratio(checkable_claims, response_count),
        "unsupported_claims_per_response": _ratio(unsupported_claims, response_count),
        "unsupported_claim_ratio": _ratio(unsupported_claims, checkable_claims),
        "label_triggering_unsupported_claims_per_response": _ratio(
            label_triggering_unsupported_claims, response_count
        ),
        "label_triggering_unsupported_claim_ratio": _ratio(
            label_triggering_unsupported_claims, checkable_claims
        ),
        "supported_claims": status_counts["supported"],
        "contradicted_claims": status_counts["contradicted"],
        "unverifiable_claims": status_counts["unverifiable"],
        "not_applicable_claims": status_counts["not_applicable"],
    }
    format_quality = {
        **_dimensions(key),
        "response_count": response_count,
        "rows_missing_claim_checks": rows_missing_claim_checks,
        "rows_invalid_claim_checks": rows_invalid_claim_checks,
        "rows_with_raw_normalized_mismatch": rows_with_raw_normalized_mismatch,
        "rows_summary_inconsistent": rows_summary_inconsistent,
        "invalid_claim_type_count": invalid_claim_type_count,
        "invalid_support_status_count": invalid_support_status_count,
        "invalid_evidence_source_count": invalid_evidence_source_count,
        "invalid_fine_label_count": invalid_fine_label_count,
        "missing_evidence_source_count": missing_evidence_source_count,
        "none_evidence_source_count": none_evidence_source_count,
        "weak_or_unassigned_evidence_ratio": _ratio(weak_or_unassigned, total_claims),
    }
    return {
        "summary": summary,
        "format_quality": format_quality,
        "claim_types": claim_type_counts,
        "evidence_sources": evidence_counts,
        "total_claims": total_claims,
    }


def _distribution_rows(
    key: tuple[str, ...], counts: Counter[str], total: int
) -> list[dict[str, Any]]:
    return [
        {
            **_dimensions(key),
            "value": value,
            "count": count,
            "share": _ratio(count, total),
        }
        for value, count in sorted(counts.items())
    ]


def _cot_exposure_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """比较 direct 与 CoT 在 claim 暴露数量和未支持比例上的差异。"""
    by_group = {
        (
            str(row["detector"]),
            str(row["dataset"]),
            str(row["model"]),
            str(row["prompt_type"]),
        ): row
        for row in summaries
    }
    rows = []
    for detector, dataset, model, prompt_type in sorted(by_group):
        if prompt_type != "evidence_grounded_cot":
            continue
        direct = by_group.get((detector, dataset, model, "direct"))
        if direct is None:
            continue
        cot = by_group[(detector, dataset, model, prompt_type)]
        rows.append(
            {
                "detector": detector,
                "dataset": dataset,
                "model": model,
                "direct_claims_per_response": direct["claims_per_response"],
                "cot_claims_per_response": cot["claims_per_response"],
                "delta_claims_per_response": cot["claims_per_response"]
                - direct["claims_per_response"],
                "direct_unsupported_claims_per_response": direct[
                    "unsupported_claims_per_response"
                ],
                "cot_unsupported_claims_per_response": cot[
                    "unsupported_claims_per_response"
                ],
                "delta_unsupported_claims_per_response": cot[
                    "unsupported_claims_per_response"
                ]
                - direct["unsupported_claims_per_response"],
                "direct_unsupported_claim_ratio": direct["unsupported_claim_ratio"],
                "cot_unsupported_claim_ratio": cot["unsupported_claim_ratio"],
                "delta_unsupported_claim_ratio": cot["unsupported_claim_ratio"]
                - direct["unsupported_claim_ratio"],
                "direct_label_triggering_unsupported_claim_ratio": direct[
                    "label_triggering_unsupported_claim_ratio"
                ],
                "cot_label_triggering_unsupported_claim_ratio": cot[
                    "label_triggering_unsupported_claim_ratio"
                ],
                "delta_label_triggering_unsupported_claim_ratio": cot[
                    "label_triggering_unsupported_claim_ratio"
                ]
                - direct["label_triggering_unsupported_claim_ratio"],
                "direct_response_level_hallucination_rate": direct[
                    "response_level_hallucination_rate"
                ],
                "cot_response_level_hallucination_rate": cot[
                    "response_level_hallucination_rate"
                ],
                "delta_response_level_hallucination_rate": cot[
                    "response_level_hallucination_rate"
                ]
                - direct["response_level_hallucination_rate"],
            }
        )
    return rows


def _group_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(row.get(key, "")) for key in _GROUP_KEYS)


def _dimensions(key: tuple[str, ...]) -> dict[str, str]:
    return dict(zip(_GROUP_KEYS, key, strict=True))


def _normalize_support_status(value: Any) -> str:
    normalized = str(value or "").strip()
    if normalized == "unsupported":
        return "unverifiable"
    return normalized


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def _percentile_nearest_rank(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(percentile * len(ordered) + 0.999999) - 1))
    return ordered[index]


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze claim-level Zero-shot Judge outputs."
    )
    parser.add_argument("--detectors", nargs="+", required=True)
    parser.add_argument("--output-dir", default="outputs/analysis/claim_level_judge")
    args = parser.parse_args()

    rows = [
        row
        for detector_path in args.detectors
        for row in read_json_records(detector_path)
    ]
    paths = write_claim_level_tables(rows, args.output_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

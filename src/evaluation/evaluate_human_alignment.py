from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.datasets.jsonl import read_jsonl, write_jsonl
from src.evaluation.agreement import cohens_kappa, matthews_corrcoef
from src.evaluation.human_alignment import compute_binary_alignment

_FINE_LABEL_COLUMNS = ("OBJ", "ATT", "SPA", "IR", "CI", "INC", "SO")
_HUMAN_LABEL_COLUMNS = {
    "human_is_hallucination",
    "human_coarse_taxonomy",
    "human_rationale",
    "human_answer_correct",
    *_FINE_LABEL_COLUMNS,
}
_ALIGNMENT_COLUMNS = [
    "metric_scope",
    "detector",
    "dataset",
    "model",
    "prompt_type",
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
_QUALITY_COLUMNS = [
    "total_rows",
    "labeled_rows",
    "skipped_rows",
    "unclear_rows",
    "missing_required_fields",
]
_PATTERN_COLUMNS = [
    "confusion_type",
    "dataset",
    "model",
    "prompt_type",
    "human_fine_taxonomy",
    "detector_taxonomy_fine",
    "count",
    "share_of_disagreements",
    "share_of_evaluated",
]
_OUTPUT_COLUMNS = {
    "overall_alignment": _ALIGNMENT_COLUMNS,
    "group_alignment": _ALIGNMENT_COLUMNS,
    "error_patterns": _PATTERN_COLUMNS,
    "annotation_quality": _QUALITY_COLUMNS,
}


def build_alignment_outputs(
    annotations: list[dict[str, Any]],
    key_rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    joined = join_annotations_with_key(annotations, key_rows)
    return {
        "overall_alignment": [_alignment_row(joined, {})],
        "group_alignment": _group_alignment_rows(joined),
        "disagreements": _disagreement_rows(joined),
        "error_patterns": _error_pattern_rows(joined),
        "annotation_quality": [_annotation_quality_row(joined)],
    }


def build_versioned_key_rows(
    base_key_rows: list[dict[str, Any]],
    detector_rows: list[dict[str, Any]],
    *,
    source_file: str,
) -> list[dict[str, Any]]:
    detector_by_response_id = _unique_index(detector_rows, "model_response_id")
    versioned_rows = []
    missing = []
    for base_row in base_key_rows:
        response_id = str(base_row.get("model_response_id", ""))
        detector_row = detector_by_response_id.get(response_id)
        if detector_row is None:
            missing.append(response_id)
            continue
        raw_details = detector_row.get("details")
        details: dict[str, Any] = raw_details if isinstance(raw_details, dict) else {}
        versioned_row = dict(base_row)
        versioned_row.update(
            {
                "source_file": source_file,
                "detector": detector_row.get("detector", ""),
                "detector_is_hallucination": detector_row.get("is_hallucination"),
                "detector_answer_correct": detector_row.get("answer_correct"),
                "detector_unsupported_visual_claim": detector_row.get(
                    "unsupported_visual_claim"
                ),
                "detector_confidence": detector_row.get("confidence", ""),
                "detector_taxonomy": detector_row.get("taxonomy", {}),
                "detector_hallucination_labels": (
                    details.get("hallucination_labels")
                    if "hallucination_labels" in details
                    else None
                ),
                "detector_explanation": detector_row.get("explanation", ""),
                "detector_raw_judge_response": detector_row.get(
                    "raw_judge_response", ""
                ),
            }
        )
        versioned_rows.append(versioned_row)
    if missing:
        examples = ", ".join(missing[:5])
        raise ValueError(
            f"Missing {len(missing)} detector rows for fixed annotation keys: {examples}"
        )
    return versioned_rows


def join_annotations_with_key(
    annotations: list[dict[str, Any]],
    key_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    key_by_annotation_id = _unique_index(key_rows, "annotation_id")
    key_by_response_id = _unique_index(key_rows, "model_response_id")
    annotation_by_id = _unique_index(annotations, "annotation_id")
    joined = []
    unmatched_annotations = []

    for annotation in annotations:
        annotation_id = str(annotation.get("annotation_id", ""))
        key = key_by_annotation_id.get(annotation_id)
        if key is None:
            key = key_by_response_id.get(str(annotation.get("model_response_id", "")))
        if key is None:
            unmatched_annotations.append(
                annotation_id or str(annotation.get("model_response_id", ""))
            )
            continue
        joined_row = dict(key)
        for column in _HUMAN_LABEL_COLUMNS:
            joined_row[column] = annotation.get(column, "")
        joined_row["human_fine_taxonomy"] = _fine_taxonomy_from_columns(joined_row)
        joined_row["predicted_is_hallucination"] = key.get("detector_is_hallucination")
        joined.append(joined_row)

    missing_annotations = [
        annotation_id
        for annotation_id in key_by_annotation_id
        if annotation_id not in annotation_by_id
    ]
    if unmatched_annotations:
        examples = ", ".join(unmatched_annotations[:5])
        raise ValueError(
            f"Could not match {len(unmatched_annotations)} annotation rows to key rows: {examples}"
        )
    if missing_annotations:
        examples = ", ".join(missing_annotations[:5])
        raise ValueError(
            f"Missing {len(missing_annotations)} annotation rows from filled CSV: {examples}"
        )
    return joined


def write_alignment_outputs(
    outputs: dict[str, list[dict[str, Any]]], output_dir: str | Path
) -> dict[str, Path]:
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, rows in outputs.items():
        if name == "disagreements":
            path = base / "disagreements.jsonl"
            write_jsonl(path, rows)
        else:
            path = base / f"{name}.csv"
            _write_csv(path, rows, _OUTPUT_COLUMNS[name])
        paths[name] = path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate detector-human hallucination alignment."
    )
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    annotations = _read_csv(args.annotations)
    key_rows = list(read_jsonl(args.key))
    outputs = build_alignment_outputs(annotations, key_rows)
    paths = write_alignment_outputs(outputs, args.output_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


def _alignment_row(
    rows: list[dict[str, Any]], dimensions: dict[str, str]
) -> dict[str, Any]:
    metrics = compute_binary_alignment(rows)
    tp = int(metrics["tp"])
    fp = int(metrics["fp"])
    tn = int(metrics["tn"])
    fn = int(metrics["fn"])
    evaluated = tp + fp + tn + fn
    predicted_positive = tp + fp
    human_positive = tp + fn
    return {
        "metric_scope": "validation_sample_only",
        "detector": dimensions.get("detector", _single_or_all(rows, "detector")),
        "dataset": dimensions.get("dataset", _single_or_all(rows, "dataset")),
        "model": dimensions.get("model", _single_or_all(rows, "model")),
        "prompt_type": dimensions.get(
            "prompt_type", _single_or_all(rows, "prompt_type")
        ),
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


def _group_alignment_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("detector", "")),
            str(row.get("dataset", "")),
            str(row.get("model", "")),
            str(row.get("prompt_type", "")),
        )
        grouped[key].append(row)
    return [
        _alignment_row(
            group_rows,
            {
                "detector": key[0],
                "dataset": key[1],
                "model": key[2],
                "prompt_type": key[3],
            },
        )
        for key, group_rows in sorted(grouped.items())
    ]


def _disagreement_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    disagreements = []
    for row in rows:
        predicted = _to_bool_or_none(row.get("predicted_is_hallucination"))
        human = _to_bool_or_none(row.get("human_is_hallucination"))
        if predicted is None or human is None or predicted == human:
            continue
        taxonomy_value = row.get("detector_taxonomy")
        taxonomy = taxonomy_value if isinstance(taxonomy_value, dict) else {}
        disagreements.append(
            {
                "annotation_id": row.get("annotation_id", ""),
                "sample_id": row.get("sample_id", ""),
                "model_response_id": row.get("model_response_id", ""),
                "dataset": row.get("dataset", ""),
                "model": row.get("model", ""),
                "prompt_type": row.get("prompt_type", ""),
                "question": row.get("question", ""),
                "reference_answer": row.get("reference_answer", ""),
                "model_response": row.get("model_response", ""),
                "predicted_is_hallucination": predicted,
                "human_is_hallucination": human,
                "confusion_type": "false_positive" if predicted else "false_negative",
                "detector_confidence": row.get("detector_confidence", ""),
                "detector_taxonomy": taxonomy,
                "detector_explanation": row.get("detector_explanation", ""),
                "human_coarse_taxonomy": row.get("human_coarse_taxonomy", ""),
                "human_fine_taxonomy": row.get("human_fine_taxonomy", ""),
                "human_rationale": row.get("human_rationale", ""),
                "image_path": row.get("image_path", ""),
            }
        )
    return disagreements


def _error_pattern_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    disagreements = _disagreement_rows(rows)
    evaluated = sum(
        1
        for row in rows
        if _to_bool_or_none(row.get("predicted_is_hallucination")) is not None
        and _to_bool_or_none(row.get("human_is_hallucination")) is not None
    )
    counts: Counter[tuple[str, str, str, str, str, str]] = Counter()
    for row in disagreements:
        taxonomy_value = row.get("detector_taxonomy")
        taxonomy = taxonomy_value if isinstance(taxonomy_value, dict) else {}
        key = (
            str(row.get("confusion_type", "")),
            str(row.get("dataset", "")),
            str(row.get("model", "")),
            str(row.get("prompt_type", "")),
            str(row.get("human_fine_taxonomy", "")),
            str(taxonomy.get("fine", "")),
        )
        counts[key] += 1
    return [
        {
            "confusion_type": key[0],
            "dataset": key[1],
            "model": key[2],
            "prompt_type": key[3],
            "human_fine_taxonomy": key[4],
            "detector_taxonomy_fine": key[5],
            "count": count,
            "share_of_disagreements": (
                count / len(disagreements) if disagreements else 0.0
            ),
            "share_of_evaluated": count / evaluated if evaluated else 0.0,
        }
        for key, count in sorted(counts.items())
    ]


def _annotation_quality_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    skipped = 0
    unclear = 0
    missing_required = 0
    for row in rows:
        human = row.get("human_is_hallucination")
        parsed = _to_bool_or_none(human)
        if parsed is None:
            skipped += 1
        if str(human).strip().lower() == "unclear":
            unclear += 1
        if str(row.get("annotation_id", "")).strip() == "" or str(human).strip() == "":
            missing_required += 1
    return {
        "total_rows": len(rows),
        "labeled_rows": len(rows) - skipped,
        "skipped_rows": skipped,
        "unclear_rows": unclear,
        "missing_required_fields": missing_required,
    }


def _fine_taxonomy_from_columns(row: dict[str, Any]) -> str:
    labels = [
        label
        for label in _FINE_LABEL_COLUMNS
        if _to_bool_or_none(row.get(label)) is True
    ]
    if labels:
        return ";".join(labels)
    human_label = _to_bool_or_none(row.get("human_is_hallucination"))
    if human_label is False:
        return "None"
    if human_label is None:
        return "Unclear"
    return "Unclear"


def _unique_index(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    duplicates = []
    for row in rows:
        value = str(row.get(key, ""))
        if not value:
            continue
        if value in indexed:
            duplicates.append(value)
            continue
        indexed[value] = row
    if duplicates:
        examples = ", ".join(duplicates[:5])
        raise ValueError(f"Duplicate key rows for {key}: {examples}")
    return indexed


def _single_or_all(rows: list[dict[str, Any]], key: str) -> str:
    values = {str(row.get(key, "")) for row in rows if str(row.get(key, ""))}
    if len(values) == 1:
        return next(iter(values))
    return "all"


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


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


if __name__ == "__main__":
    main()

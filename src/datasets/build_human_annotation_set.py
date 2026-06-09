from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.datasets.jsonl import read_jsonl, write_jsonl
from src.pipelines.experiment_groups import select_detector_groups

_GROUP_KEYS = ("dataset", "model", "prompt_type")
_BLIND_COLUMNS = [
    "annotation_id",
    "sample_id",
    "model_response_id",
    "dataset",
    "model",
    "prompt_type",
    "task_type",
    "image_path",
    "question",
    "choices",
    "reference_answer",
    "model_raw_response",
    "model_parsed_final_answer",
    "model_parsed_reasoning",
    "model_parsed_visual_evidence",
    "human_is_hallucination",
    "human_coarse_taxonomy",
    "OBJ",
    "ATT",
    "SPA",
    "IR",
    "CI",
    "INC",
    "SO",
    "human_rationale",
    "human_answer_correct",
]
_HUMAN_COLUMNS = {
    "human_is_hallucination",
    "human_coarse_taxonomy",
    "OBJ",
    "ATT",
    "SPA",
    "IR",
    "CI",
    "INC",
    "SO",
    "human_rationale",
    "human_answer_correct",
}
_KEY_COLUMNS = [
    "annotation_id",
    "sample_id",
    "model_response_id",
    "dataset",
    "model",
    "prompt_type",
    "source_file",
    "sampling_reason",
    "detector",
    "detector_is_hallucination",
    "detector_answer_correct",
    "detector_unsupported_visual_claim",
    "detector_confidence",
    "detector_taxonomy",
    "detector_explanation",
    "detector_raw_judge_response",
    "question",
    "reference_answer",
    "model_response",
    "image_path",
]
_GUIDELINE = """You are asked to judge whether the model response contains hallucination.

A hallucination means that the response contains at least one claim that is unsupported by, or contradicts, the image, question, choices, or reference context.

Please inspect the image, question, reference answer, and model response.

Do not mark hallucination solely because the final answer differs from the reference answer. Instead, check whether the response contains unsupported visual, factual, or reasoning claims.

For CoT responses, visible reasoning steps are part of the model response and should be judged.

Use unclear only when the image, question, or response is insufficient to decide."""


def build_annotation_rows(
    samples: list[dict[str, Any]],
    responses: list[dict[str, Any]],
    detector_rows: list[dict[str, Any]],
    *,
    source_file: str,
    strict: bool = True,
) -> list[dict[str, Any]]:
    samples_by_id = {str(row["sample_id"]): row for row in samples}
    responses_by_id = {_model_response_id(row): row for row in responses}
    joined: list[dict[str, Any]] = []
    unmatched: list[str] = []

    for detector in detector_rows:
        model_response_id = str(detector.get("model_response_id", ""))
        response = responses_by_id.get(model_response_id)
        if response is None:
            unmatched.append(f"missing response for {model_response_id}")
            continue
        sample_id = str(detector.get("sample_id", response.get("sample_id", "")))
        sample = samples_by_id.get(sample_id)
        if sample is None:
            unmatched.append(f"missing sample for {sample_id}")
            continue
        parsed_value = response.get("parsed")
        parsed = parsed_value if isinstance(parsed_value, dict) else {}
        joined.append(
            {
                "sample_id": str(sample["sample_id"]),
                "model_response_id": model_response_id,
                "dataset": str(
                    detector.get("dataset")
                    or response.get("dataset")
                    or sample.get("dataset", "")
                ),
                "model": str(detector.get("model") or response.get("model", "")),
                "prompt_type": str(
                    detector.get("prompt_type") or response.get("prompt_type", "")
                ),
                "task_type": str(sample.get("task_type", "")),
                "image_path": str(sample.get("image_path", "")),
                "question": str(sample.get("question", "")),
                "choices": _format_cell(sample.get("choices")),
                "reference_answer": str(sample.get("reference_answer", "")),
                "model_raw_response": str(response.get("raw_response", "")),
                "model_parsed_final_answer": str(parsed.get("final_answer", "")),
                "model_parsed_reasoning": str(parsed.get("reasoning", "")),
                "model_parsed_visual_evidence": str(parsed.get("visual_evidence", "")),
                "source_file": source_file,
                "detector": str(detector.get("detector", "")),
                "detector_is_hallucination": detector.get("is_hallucination"),
                "detector_answer_correct": detector.get("answer_correct"),
                "detector_unsupported_visual_claim": detector.get(
                    "unsupported_visual_claim"
                ),
                "detector_confidence": detector.get("confidence"),
                "detector_taxonomy": detector.get("taxonomy")
                or {"coarse": "None", "fine": "None"},
                "detector_explanation": str(detector.get("explanation", "")),
                "detector_raw_judge_response": str(
                    detector.get("raw_judge_response", "")
                ),
            }
        )

    if strict and unmatched:
        examples = "; ".join(unmatched[:5])
        raise ValueError(f"Could not join {len(unmatched)} detector rows: {examples}")
    return joined


def sample_detector_validation_rows(
    rows: list[dict[str, Any]],
    *,
    per_group: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not _is_usable_annotation_response(row):
            continue
        group_key = tuple(str(row.get(key, "")) for key in _GROUP_KEYS)
        if len(group_key) != 3:
            raise ValueError(f"Invalid group key: {group_key}")
        groups[group_key].append(row)

    selected: list[dict[str, Any]] = []
    for group_key in sorted(groups):
        group_rows = list(groups[group_key])
        rng.shuffle(group_rows)
        selected.extend(_sample_group(group_rows, per_group, rng))
    return selected


def write_annotation_outputs(
    selected_rows: list[dict[str, Any]],
    annotation_output: str | Path,
    key_output: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    annotation_path = Path(annotation_output)
    key_path = Path(key_output)
    _ensure_can_write(annotation_path, overwrite)
    _ensure_can_write(key_path, overwrite)

    blind_rows = []
    key_rows = []
    for index, row in enumerate(selected_rows, start=1):
        annotation_id = str(row.get("annotation_id") or f"ann_{index:04d}")
        blind = {column: "" for column in _BLIND_COLUMNS}
        for column in _BLIND_COLUMNS:
            if column in _HUMAN_COLUMNS:
                continue
            blind[column] = _format_cell(row.get(column, ""))
        blind["annotation_id"] = annotation_id
        blind_rows.append(blind)

        key = {column: row.get(column, "") for column in _KEY_COLUMNS}
        key["annotation_id"] = annotation_id
        key["sampling_reason"] = str(row.get("sampling_reason", "unsampled"))
        key["model_response"] = str(
            row.get("model_raw_response", row.get("model_response", ""))
        )
        key_rows.append(key)

    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    with annotation_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=_BLIND_COLUMNS)
        writer.writeheader()
        writer.writerows(blind_rows)
    write_jsonl(key_path, key_rows)


def load_registered_annotation_rows(
    *,
    experiment: str,
    detector: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in select_detector_groups(experiments={experiment}, detectors={detector}):
        samples = list(read_jsonl(group.samples_path))
        responses = list(read_jsonl(group.responses_path))
        detector_rows = list(read_jsonl(group.output_path))
        rows.extend(
            build_annotation_rows(
                samples,
                responses,
                detector_rows,
                source_file=group.output_path,
            )
        )
    return rows


def summarize_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group_key = tuple(str(row.get(key, "")) for key in _GROUP_KEYS)
        if len(group_key) != 3:
            raise ValueError(f"Invalid group key: {group_key}")
        groups[group_key].append(row)

    summary = []
    for group_key, group_rows in sorted(groups.items()):
        positives = sum(
            row.get("detector_is_hallucination") is True for row in group_rows
        )
        negatives = sum(
            row.get("detector_is_hallucination") is False for row in group_rows
        )
        summary.append(
            {
                "dataset": group_key[0],
                "model": group_key[1],
                "prompt_type": group_key[2],
                "count": len(group_rows),
                "detector_positive": positives,
                "detector_negative": negatives,
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a blind Human-as-Judge annotation set from detector outputs."
    )
    parser.add_argument("--experiment", default="one_tenth")
    parser.add_argument("--detector", default="zero_shot")
    parser.add_argument("--per-group", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--annotation-output",
        default="outputs/human_annotation/one_tenth_annotation_blind.csv",
    )
    parser.add_argument(
        "--key-output",
        default="outputs/human_annotation/one_tenth_annotation_key.jsonl",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = load_registered_annotation_rows(
        experiment=args.experiment, detector=args.detector
    )
    selected = sample_detector_validation_rows(
        rows, per_group=args.per_group, seed=args.seed
    )
    for summary in summarize_groups(selected if selected else rows):
        print(summary)
    print(_GUIDELINE)
    if args.dry_run:
        return
    write_annotation_outputs(
        selected, args.annotation_output, args.key_output, overwrite=args.overwrite
    )


def _sample_group(
    rows: list[dict[str, Any]],
    per_group: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used: set[str] = set()

    def take(
        candidates: list[dict[str, Any]],
        count: int,
        reason: str | None,
        *,
        shuffle: bool = True,
    ) -> None:
        pool = list(candidates)
        if shuffle:
            rng.shuffle(pool)
        taken = 0
        for row in pool:
            row_id = _row_id(row)
            if row_id in used:
                continue
            selected.append(
                {**row, "sampling_reason": reason or _diagnostic_reason(row)}
            )
            used.add(row_id)
            taken += 1
            if taken >= count:
                break

    positives = [row for row in rows if row.get("detector_is_hallucination") is True]
    negatives = [row for row in rows if row.get("detector_is_hallucination") is False]
    target_binary = min(2, per_group)
    take(positives, target_binary, "detector_positive")
    take(negatives, target_binary, "detector_negative")

    if len(selected) < per_group:
        take(_diagnostic_candidates(rows, used), 1, None, shuffle=False)

    remaining = [row for row in rows if _row_id(row) not in used]
    while len(selected) < min(per_group, len(rows)) and remaining:
        row = remaining.pop(0)
        selected.append({**row, "sampling_reason": "fallback_available"})
        used.add(_row_id(row))
    return selected[:per_group]


def _is_usable_annotation_response(row: dict[str, Any]) -> bool:
    raw_response = row.get("model_raw_response")
    if raw_response is None:
        return False
    response = str(raw_response).strip()
    if not response:
        return False
    if response.endswith((":", "：")):
        return False
    prompt_type = str(row.get("prompt_type", ""))
    final_answer = str(row.get("model_parsed_final_answer", "")).strip()
    normalized_response = response.lower()
    has_final_answer_marker = "final answer" in normalized_response or "答案" in response
    if prompt_type == "direct" and len(final_answer) > 1200 and not has_final_answer_marker:
        return False
    return True


def _diagnostic_candidates(
    rows: list[dict[str, Any]], used: set[str]
) -> list[dict[str, Any]]:
    candidates = [row for row in rows if _row_id(row) not in used]
    return sorted(candidates, key=_diagnostic_rank)


def _diagnostic_rank(row: dict[str, Any]) -> tuple[int, int, int]:
    confidence_rank = 0 if row.get("detector_confidence") == "low" else 1
    taxonomy_value = row.get("detector_taxonomy")
    taxonomy = taxonomy_value if isinstance(taxonomy_value, dict) else {}
    taxonomy_rank = (
        0 if taxonomy.get("fine") not in {None, "", "None", "Unclear"} else 1
    )
    response_length_rank = -len(str(row.get("model_raw_response", "")))
    return (confidence_rank, taxonomy_rank, response_length_rank)


def _diagnostic_reason(row: dict[str, Any]) -> str:
    if row.get("detector_confidence") == "low":
        return "diagnostic_low_confidence"
    taxonomy_value = row.get("detector_taxonomy")
    taxonomy = taxonomy_value if isinstance(taxonomy_value, dict) else {}
    if taxonomy.get("fine") not in {None, "", "None", "Unclear"}:
        return "diagnostic_taxonomy_rich"
    if len(str(row.get("model_raw_response", ""))) > 500:
        return "diagnostic_long_response"
    return "diagnostic"


def _model_response_id(response: dict[str, Any]) -> str:
    return f"{response['run_id']}:{response['sample_id']}"


def _row_id(row: dict[str, Any]) -> str:
    return str(row.get("model_response_id") or row.get("sample_id"))


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _ensure_can_write(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")


if __name__ == "__main__":
    main()

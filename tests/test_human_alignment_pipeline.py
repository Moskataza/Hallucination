from pathlib import Path

import pytest

from src.datasets.build_human_annotation_set import (
    build_annotation_rows,
    sample_detector_validation_rows,
    write_annotation_outputs,
)
from src.datasets.jsonl import read_jsonl
from src.evaluation.evaluate_human_alignment import (
    build_alignment_outputs,
    join_annotations_with_key,
)


def test_sampling_balances_detector_predictions_and_preserves_reasons():
    rows = [
        _joined_row("pope", "gemini", "direct", "p1", True, confidence="high"),
        _joined_row("pope", "gemini", "direct", "p2", True, confidence="medium"),
        _joined_row("pope", "gemini", "direct", "p3", False, confidence="high"),
        _joined_row("pope", "gemini", "direct", "p4", False, confidence="medium"),
        _joined_row(
            "pope",
            "gemini",
            "direct",
            "p5",
            True,
            confidence="low",
            raw_response="long " * 80,
        ),
        _joined_row("pope", "gemini", "direct", "p6", False, confidence="high"),
    ]

    selected = sample_detector_validation_rows(rows, per_group=5, seed=7)

    assert len(selected) == 5
    assert sum(row["detector_is_hallucination"] is True for row in selected) >= 2
    assert sum(row["detector_is_hallucination"] is False for row in selected) >= 2
    assert {row["sampling_reason"] for row in selected} >= {
        "detector_positive",
        "detector_negative",
        "diagnostic_low_confidence",
    }


def test_sampling_falls_back_when_prediction_bucket_is_small():
    rows = [
        _joined_row("pope", "gemini", "direct", "p1", True),
        _joined_row("pope", "gemini", "direct", "p2", False),
        _joined_row("pope", "gemini", "direct", "p3", False),
        _joined_row("pope", "gemini", "direct", "p4", False),
        _joined_row("pope", "gemini", "direct", "p5", False),
    ]

    selected = sample_detector_validation_rows(rows, per_group=5, seed=7)

    assert len(selected) == 5
    assert any(row["sampling_reason"].startswith("fallback") for row in selected)


def test_diagnostic_sampling_selects_actual_low_confidence_row():
    rows = [
        _joined_row("pope", "gemini", "direct", "p1", True, confidence="high"),
        _joined_row("pope", "gemini", "direct", "p2", True, confidence="high"),
        _joined_row("pope", "gemini", "direct", "p3", False, confidence="high"),
        _joined_row("pope", "gemini", "direct", "p4", False, confidence="high"),
        _joined_row("pope", "gemini", "direct", "p5", None, confidence="low"),
        _joined_row("pope", "gemini", "direct", "p6", None, confidence="medium"),
    ]

    selected = sample_detector_validation_rows(rows, per_group=5, seed=7)
    diagnostic = next(
        row for row in selected if row["sampling_reason"] == "diagnostic_low_confidence"
    )

    assert diagnostic["sample_id"] == "p5"
    assert diagnostic["detector_confidence"] == "low"


def test_sampling_excludes_unusable_incomplete_responses():
    rows = [
        _joined_row("mathvista", "gemini", "direct", "p1", True, raw_response=""),
        _joined_row("mathvista", "gemini", "direct", "p2", True, raw_response="unfinished:"),
        _joined_row(
            "mathvista",
            "gemini",
            "direct",
            "p3",
            True,
            raw_response="long reasoning " * 120,
            final_answer="long reasoning " * 120,
        ),
        _joined_row("mathvista", "gemini", "direct", "p4", True, raw_response="Final Answer: A"),
        _joined_row("mathvista", "gemini", "direct", "p5", False, raw_response="No"),
        _joined_row("mathvista", "gemini", "direct", "p6", False, raw_response=None),
        _joined_row(
            "mathvista",
            "gemini",
            "direct",
            "p7",
            True,
            raw_response="long reasoning " * 120 + "final answer: A",
            final_answer="long reasoning " * 120 + "final answer: A",
        ),
    ]

    selected = sample_detector_validation_rows(rows, per_group=5, seed=7)

    assert {row["sample_id"] for row in selected} == {"p4", "p5", "p7"}


def test_build_annotation_rows_joins_sample_response_and_detector_context():
    sample = _sample_row("p1")
    response = _response_row("p1", "gemini", "direct")
    detector = _detector_row(response, True, confidence="low")

    rows = build_annotation_rows(
        [sample], [response], [detector], source_file="detector.jsonl"
    )

    assert rows == [
        {
            "sample_id": "p1",
            "model_response_id": "run:p1",
            "dataset": "pope",
            "model": "gemini",
            "prompt_type": "direct",
            "task_type": "vqa_yes_no",
            "image_path": "images/p1.jpg",
            "question": "Is there a dog?",
            "choices": "",
            "reference_answer": "no",
            "model_raw_response": "Yes, there is a dog.",
            "model_parsed_final_answer": "Yes",
            "model_parsed_reasoning": "",
            "model_parsed_visual_evidence": "",
            "source_file": "detector.jsonl",
            "detector": "response_claim_zero_shot_judge",
            "detector_is_hallucination": True,
            "detector_answer_correct": False,
            "detector_unsupported_visual_claim": True,
            "detector_confidence": "low",
            "detector_taxonomy": {"coarse": "Factual", "fine": "OBJ"},
            "detector_explanation": "Object is absent.",
            "detector_raw_judge_response": "{}",
        }
    ]


def test_build_annotation_rows_rejects_unmatched_detector_rows():
    sample = _sample_row("p1")
    response = _response_row("p1", "gemini", "direct")
    detector = _detector_row(response, True)
    detector["model_response_id"] = "missing:p1"

    with pytest.raises(ValueError, match="Could not join"):
        build_annotation_rows(
            [sample], [response], [detector], source_file="detector.jsonl"
        )


def test_write_annotation_outputs_separates_blind_csv_and_key_jsonl(tmp_path: Path):
    selected = [_joined_row("pope", "gemini", "direct", "p1", True)]
    annotation_path = tmp_path / "blind.csv"
    key_path = tmp_path / "key.jsonl"

    write_annotation_outputs(selected, annotation_path, key_path)

    blind_text = annotation_path.read_text(encoding="utf-8")
    assert "detector_is_hallucination" not in blind_text
    assert "detector_confidence" not in blind_text
    assert "OBJ" in blind_text
    assert "ATT" in blind_text
    assert "SPA" in blind_text
    assert "human_fine_taxonomy" not in blind_text
    assert "human_rationale" in blind_text
    assert "human_answer_correct" in blind_text
    assert blind_text.splitlines()[0].endswith("human_answer_correct")
    assert "human_unsupported_visual_claim" not in blind_text
    assert "human_error_pattern" not in blind_text
    assert "annotator_id" not in blind_text
    assert "adjudication_status" not in blind_text

    key_rows = list(read_jsonl(key_path))
    assert key_rows[0]["annotation_id"] == "ann_0001"
    assert key_rows[0]["detector_is_hallucination"] is True
    assert key_rows[0]["sampling_reason"] == "unsampled"


def test_alignment_outputs_include_metrics_skips_disagreements_and_patterns():
    annotations = [
        _annotation_row("ann_0001", True, fine_labels=("OBJ", "ATT", "INC")),
        _annotation_row("ann_0002", False, pattern="judge_overstrict"),
        _annotation_row("ann_0003", False),
        _annotation_row("ann_0004", True, pattern="reasoning_chain_error_missed"),
        _annotation_row("ann_0005", "unclear"),
    ]
    key_rows = [
        _key_row("ann_0001", True),
        _key_row("ann_0002", True),
        _key_row("ann_0003", False),
        _key_row("ann_0004", False),
        _key_row("ann_0005", True),
    ]

    outputs = build_alignment_outputs(annotations, key_rows)

    overall = outputs["overall_alignment"][0]
    assert overall["tp"] == 1
    assert overall["fp"] == 1
    assert overall["tn"] == 1
    assert overall["fn"] == 1
    assert overall["skipped"] == 1
    assert overall["precision"] == 0.5
    assert overall["recall"] == 0.5
    assert overall["f1"] == 0.5
    assert overall["accuracy"] == 0.5
    assert overall["metric_scope"] == "validation_sample_only"
    assert "cohens_kappa" in overall
    assert "matthews_corrcoef" in overall

    disagreements = outputs["disagreements"]
    assert {row["confusion_type"] for row in disagreements} == {
        "false_positive",
        "false_negative",
    }
    assert disagreements[0]["human_fine_taxonomy"] in {"OBJ;ATT;INC", "None"}

    patterns = outputs["error_patterns"]
    assert any(row["human_fine_taxonomy"] == "None" for row in patterns)
    assert outputs["annotation_quality"][0]["skipped_rows"] == 1
    assert outputs["group_alignment"]


def test_alignment_join_keeps_key_metadata_authoritative():
    annotation = _annotation_row("ann_0001", True)
    annotation.update(
        {
            "dataset": "edited_dataset",
            "model": "edited_model",
            "prompt_type": "edited_prompt",
            "question": "edited question",
        }
    )

    joined = join_annotations_with_key([annotation], [_key_row("ann_0001", True)])

    assert joined[0]["dataset"] == "pope"
    assert joined[0]["model"] == "gemini"
    assert joined[0]["prompt_type"] == "direct"
    assert joined[0]["question"] == "Is there a dog?"
    assert joined[0]["human_is_hallucination"] is True
    assert joined[0]["human_answer_correct"] is False


def test_alignment_join_rejects_unmatched_and_duplicate_rows():
    with pytest.raises(ValueError, match="Could not match"):
        join_annotations_with_key(
            [_annotation_row("ann_missing", True)], [_key_row("ann_0001", True)]
        )

    with pytest.raises(ValueError, match="Duplicate key"):
        join_annotations_with_key(
            [_annotation_row("ann_0001", True)],
            [_key_row("ann_0001", True), _key_row("ann_0001", False)],
        )

    with pytest.raises(ValueError, match="Missing 1 annotation rows"):
        join_annotations_with_key(
            [_annotation_row("ann_0001", True)],
            [_key_row("ann_0001", True), _key_row("ann_0002", False)],
        )


def _sample_row(sample_id: str) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "dataset": "pope",
        "task_type": "vqa_yes_no",
        "image_path": f"images/{sample_id}.jpg",
        "question": "Is there a dog?",
        "reference_answer": "no",
        "choices": None,
    }


def _response_row(sample_id: str, model: str, prompt_type: str) -> dict[str, object]:
    return {
        "run_id": "run",
        "sample_id": sample_id,
        "dataset": "pope",
        "model": model,
        "model_type": "closed" if model == "gemini" else "open",
        "prompt_type": prompt_type,
        "prompt_version": "v1",
        "raw_response": "Yes, there is a dog.",
        "parsed": {
            "visual_evidence": "",
            "reasoning": "",
            "final_answer": "Yes",
            "parse_status": "ok",
        },
    }


def _detector_row(
    response: dict[str, object], is_hallucination: bool, confidence: str = "high"
) -> dict[str, object]:
    return {
        "run_id": "judge_run",
        "sample_id": response["sample_id"],
        "model_response_id": f"{response['run_id']}:{response['sample_id']}",
        "detector": "response_claim_zero_shot_judge",
        "answer_correct": False,
        "is_hallucination": is_hallucination,
        "taxonomy": {
            "coarse": "Factual",
            "fine": "OBJ" if is_hallucination else "None",
        },
        "unsupported_visual_claim": is_hallucination,
        "confidence": confidence,
        "explanation": "Object is absent.",
        "raw_judge_response": "{}",
        "dataset": response["dataset"],
        "model": response["model"],
        "prompt_type": response["prompt_type"],
    }


def _joined_row(
    dataset: str,
    model: str,
    prompt_type: str,
    sample_id: str,
    detector_is_hallucination: bool,
    confidence: str = "high",
    raw_response: str = "Yes",
    final_answer: str = "Yes",
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "model_response_id": f"run:{sample_id}",
        "dataset": dataset,
        "model": model,
        "prompt_type": prompt_type,
        "task_type": "vqa_yes_no",
        "image_path": f"images/{sample_id}.jpg",
        "question": "Is there a dog?",
        "choices": "",
        "reference_answer": "no",
        "model_raw_response": raw_response,
        "model_parsed_final_answer": final_answer,
        "model_parsed_reasoning": "",
        "model_parsed_visual_evidence": "",
        "source_file": "detector.jsonl",
        "detector": "response_claim_zero_shot_judge",
        "detector_is_hallucination": detector_is_hallucination,
        "detector_answer_correct": False,
        "detector_unsupported_visual_claim": detector_is_hallucination,
        "detector_confidence": confidence,
        "detector_taxonomy": {
            "coarse": "Factual",
            "fine": "OBJ" if detector_is_hallucination else "None",
        },
        "detector_explanation": "Object is absent.",
        "detector_raw_judge_response": "{}",
    }


def _annotation_row(
    annotation_id: str,
    human_label: object,
    fine_labels: tuple[str, ...] = (),
    pattern: str = "",
) -> dict[str, object]:
    row: dict[str, object] = {
        "annotation_id": annotation_id,
        "human_is_hallucination": human_label,
        "human_coarse_taxonomy": "Unclear",
        "human_rationale": pattern or "rationale",
        "human_answer_correct": False,
    }
    for label in ("OBJ", "ATT", "SPA", "IR", "CI", "INC", "SO"):
        row[label] = 1 if label in fine_labels else 0
    return row


def _key_row(annotation_id: str, detector_label: bool) -> dict[str, object]:
    sample_id = annotation_id.replace("ann_", "p")
    return {
        "annotation_id": annotation_id,
        "sample_id": sample_id,
        "model_response_id": f"run:{sample_id}",
        "dataset": "pope",
        "model": "gemini",
        "prompt_type": "direct",
        "source_file": "detector.jsonl",
        "sampling_reason": "test",
        "detector": "response_claim_zero_shot_judge",
        "detector_is_hallucination": detector_label,
        "detector_answer_correct": not detector_label,
        "detector_unsupported_visual_claim": detector_label,
        "detector_confidence": "high",
        "detector_taxonomy": {
            "coarse": "Factual",
            "fine": "OBJ" if detector_label else "None",
        },
        "detector_explanation": "explanation",
        "detector_raw_judge_response": "{}",
        "question": "Is there a dog?",
        "reference_answer": "no",
        "model_response": "Yes",
        "image_path": "images/p.jpg",
    }

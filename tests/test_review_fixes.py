from pathlib import Path

import pytest

from src.analysis.aggregate_results import (
    build_experiment_tables,
    group_detector_results,
    write_experiment_tables,
)
from src.analysis.tables import rows_to_markdown
from src.datasets.convert_mathvista import convert_mathvista_record
from src.datasets.schema import EvalSample, ModelResponse, ParsedResponse, TaxonomyLabel
from src.detectors.checklist_judge import render_checklist_judge_prompt
from src.detectors.pope_rule_based import detect_pope_hallucination
from src.detectors.taxonomy import normalize_fine_label
from src.detectors.zero_shot_judge import render_zero_shot_judge_prompt


def _sample() -> EvalSample:
    return EvalSample(
        sample_id="pope_1",
        dataset="pope",
        task_type="vqa_yes_no",
        image_path="x.jpg",
        question="Is there a dog?",
        reference_answer="no",
    )


def _response() -> ModelResponse:
    return ModelResponse(
        run_id="resp_1",
        sample_id="pope_1",
        dataset="pope",
        model="gemini",
        model_type="closed",
        prompt_type="direct",
        prompt_version="v1",
        raw_response="Yes",
        parsed=ParsedResponse(final_answer="Yes", parse_status="ok"),
    )


def test_judge_prompt_rendering_preserves_json_examples():
    taxonomy = Path("prompts/judge/taxonomy_definition.txt").read_text(encoding="utf-8")
    zero_shot = render_zero_shot_judge_prompt(_sample(), _response(), taxonomy)
    checklist = render_checklist_judge_prompt(_sample(), _response(), taxonomy)
    assert '"answer_correct"' in zero_shot
    assert '"claim_checks"' in checklist
    assert "{question}" not in zero_shot
    assert "Is there a dog?" in checklist


def test_zero_shot_prompt_contains_v2_claim_grounding_rules():
    taxonomy = Path("prompts/judge/taxonomy_definition.txt").read_text(encoding="utf-8")
    zero_shot = render_zero_shot_judge_prompt(_sample(), _response(), taxonomy)

    assert "atomic checkable claims" in zero_shot
    assert "evidence_source" in zero_shot
    assert "not_applicable" in zero_shot
    assert "non_claim" in zero_shot
    assert "summary_consistent_with_claims" in zero_shot
    assert "Concise yes/no" in zero_shot
    assert "reference_answer=UNAVAILABLE" in zero_shot


def test_taxonomy_none_normalization():
    assert normalize_fine_label("None") == "None"
    assert normalize_fine_label("none") == "None"
    assert normalize_fine_label("Unclear") == "Unclear"
    assert normalize_fine_label("obj") == "OBJ"
    assert TaxonomyLabel(coarse="None", fine="None").to_dict() == {
        "coarse": "None",
        "fine": "None",
    }


def test_schema_rejects_invalid_values():
    with pytest.raises(ValueError):
        EvalSample(
            sample_id="bad",
            dataset="unknown",
            task_type="x",
            image_path="x.jpg",
            question="q",
            reference_answer="a",
        )
    with pytest.raises(ValueError):
        ParsedResponse(parse_status="bad")


def test_pope_detector_outputs_grouping_metadata():
    result = detect_pope_hallucination(_sample(), _response()).to_dict()
    assert result["dataset"] == "pope"
    assert result["model"] == "gemini"
    assert result["prompt_type"] == "direct"
    grouped = group_detector_results([result])
    assert ("pope_rule_based", "pope", "gemini", "direct") in grouped


def test_group_detector_results_rejects_missing_keys():
    with pytest.raises(ValueError):
        group_detector_results(
            [
                {
                    "sample_id": "x",
                    "is_hallucination": False,
                    "taxonomy": {"coarse": "None", "fine": "None"},
                }
            ]
        )


def test_markdown_table_escapes_cells():
    table = rows_to_markdown([{"explanation": "a | b\nnext"}], ["explanation"])
    assert "a \\| b next" in table


def test_mathvista_missing_answer_is_explicitly_marked():
    sample = convert_mathvista_record(
        {
            "pid": "1001",
            "image": "images/1001.jpg",
            "question": "What is shown?",
            "answer": None,
        },
        image_root="data/raw/mathvista",
    )
    assert sample.reference_answer == "UNAVAILABLE"
    assert sample.metadata["answer_available"] is False
    assert sample.image_path == "data/raw/mathvista/images/1001.jpg"


def test_experiment_tables_summarize_detector_results(tmp_path: Path):
    rows = [
        _detector_row(
            dataset="pope",
            model="gemini",
            prompt_type="direct",
            sample_id="p1",
            answer_correct=True,
            is_hallucination=False,
            labels=[],
        ),
        _detector_row(
            dataset="pope",
            model="gemini",
            prompt_type="evidence_grounded_cot",
            sample_id="p1",
            answer_correct=True,
            is_hallucination=True,
            labels=["OBJ", "CI"],
            unsupported_visual_claim=True,
        ),
        _detector_row(
            dataset="pope",
            model="qwen",
            prompt_type="direct",
            sample_id="p1",
            answer_correct=False,
            is_hallucination=True,
            labels=["ATT"],
            unsupported_visual_claim=True,
        ),
    ]

    tables = build_experiment_tables(rows)
    overall = tables["overall_results"]
    cot_effect = tables["cot_effect"]
    model_comparison = tables["model_comparison"]
    taxonomy = tables["taxonomy_distribution"]

    gemini_cot = next(
        row
        for row in overall
        if row["model"] == "gemini" and row["prompt_type"] == "evidence_grounded_cot"
    )
    assert gemini_cot["hallucination_rate"] == 1.0
    assert gemini_cot["factual_rate"] == 1.0
    assert gemini_cot["logical_rate"] == 1.0
    assert gemini_cot["OBJ_count"] == 1
    assert gemini_cot["CI_count"] == 1
    assert cot_effect == [
        {
            "detector": "response_claim_zero_shot_judge",
            "dataset": "pope",
            "model": "gemini",
            "delta_hallucination_rate": 1.0,
            "delta_factual_rate": 1.0,
            "delta_logical_rate": 1.0,
            "delta_unsupported_visual_claim_rate": 1.0,
            "delta_grounded_accuracy": -1.0,
        }
    ]
    assert model_comparison[0]["model_a"] == "gemini"
    assert model_comparison[0]["model_b"] == "qwen"
    assert taxonomy[0]["dataset"] == "pope"

    paths = write_experiment_tables(rows, tmp_path)
    assert set(paths) == {
        "overall_results",
        "cot_effect",
        "model_comparison",
        "taxonomy_distribution",
    }
    assert (
        (tmp_path / "overall_results.csv")
        .read_text(encoding="utf-8")
        .startswith("detector,dataset,model,prompt_type")
    )


def test_experiment_tables_normalize_labels_and_fallback_to_taxonomy():
    rows = [
        _detector_row(
            dataset="pope",
            model="gemini",
            prompt_type="direct",
            sample_id="p1",
            answer_correct=False,
            is_hallucination=True,
            labels=[" obj ", "bad_label"],
        ),
        _detector_row(
            dataset="pope",
            model="gemini",
            prompt_type="direct",
            sample_id="p2",
            answer_correct=False,
            is_hallucination=True,
            labels=[],
            taxonomy_fine="SPA",
        ),
    ]

    overall = build_experiment_tables(rows)["overall_results"]

    assert overall[0]["OBJ_count"] == 1
    assert overall[0]["SPA_count"] == 1
    assert overall[0]["factual_rate"] == 1.0


def test_experiment_tables_keep_detectors_separate():
    rows = [
        _detector_row(
            detector="detector_a",
            dataset="pope",
            model="gemini",
            prompt_type="direct",
            sample_id="p1",
            answer_correct=True,
            is_hallucination=False,
            labels=[],
        ),
        _detector_row(
            detector="detector_b",
            dataset="pope",
            model="gemini",
            prompt_type="direct",
            sample_id="p1",
            answer_correct=False,
            is_hallucination=True,
            labels=["OBJ"],
        ),
    ]

    overall = build_experiment_tables(rows)["overall_results"]

    assert len(overall) == 2
    assert {row["detector"] for row in overall} == {"detector_a", "detector_b"}


def test_experiment_table_csvs_keep_headers_when_empty(tmp_path: Path):
    paths = write_experiment_tables([], tmp_path)

    assert (
        paths["overall_results"]
        .read_text(encoding="utf-8")
        .startswith("detector,dataset,model,prompt_type")
    )
    assert (
        paths["cot_effect"]
        .read_text(encoding="utf-8")
        .startswith("detector,dataset,model,delta_hallucination_rate")
    )
    assert (
        paths["model_comparison"]
        .read_text(encoding="utf-8")
        .startswith("detector,dataset,prompt_type,model_a,model_b")
    )


def _detector_row(
    *,
    dataset: str,
    model: str,
    prompt_type: str,
    sample_id: str,
    answer_correct: bool,
    is_hallucination: bool,
    labels: list[str],
    unsupported_visual_claim: bool = False,
    detector: str = "response_claim_zero_shot_judge",
    taxonomy_fine: str | None = None,
) -> dict[str, object]:
    primary = taxonomy_fine or (labels[0] if labels else "None")
    coarse = "Factual" if primary in {"OBJ", "ATT", "SPA"} else "Logical"
    if primary == "None":
        coarse = "None"
    return {
        "run_id": "judge_run",
        "sample_id": sample_id,
        "model_response_id": f"response_run:{sample_id}:{model}:{prompt_type}",
        "detector": detector,
        "answer_correct": answer_correct,
        "is_hallucination": is_hallucination,
        "taxonomy": {"coarse": coarse, "fine": primary},
        "unsupported_visual_claim": unsupported_visual_claim,
        "confidence": "high",
        "dataset": dataset,
        "model": model,
        "prompt_type": prompt_type,
        "details": {"hallucination_labels": labels},
    }

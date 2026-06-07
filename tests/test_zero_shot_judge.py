from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.datasets.jsonl import read_jsonl, write_jsonl
from src.datasets.schema import (
    DetectorResult,
    EvalSample,
    ModelResponse,
    ParsedResponse,
)
from src.detectors import zero_shot_judge
from src.detectors.zero_shot_judge import (
    details_to_detector_result,
    judge_response,
    parse_judge_output,
    render_zero_shot_judge_prompt,
)


class FakeJudgeClient:
    def __init__(self, raw_response: str) -> None:
        self.raw_response = raw_response
        self.calls: list[dict[str, Any]] = []

    def chat_completion(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"choices": [{"message": {"content": self.raw_response}}]}


def _sample(image_path: str = "data/raw/pope/sample.jpg") -> EvalSample:
    return EvalSample(
        sample_id="pope_1",
        dataset="pope",
        task_type="vqa_yes_no",
        image_path=image_path,
        question="Is there a dog in the image?",
        reference_answer="no",
        metadata={"split": "random"},
    )


def _response() -> ModelResponse:
    return ModelResponse(
        run_id="model_run_1",
        sample_id="pope_1",
        dataset="pope",
        model="qwen/qwen3-vl-8b-instruct",
        model_type="open",
        prompt_type="evidence_grounded_cot",
        prompt_version="v1",
        raw_response="Visual Evidence: I see a dog.\nFinal Answer: Yes",
        parsed=ParsedResponse(
            visual_evidence="I see a dog.",
            final_answer="Yes",
            parse_status="ok",
        ),
    )


def _judge_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "answer_correct": False,
        "is_hallucination": True,
        "label_order": ["OBJ", "ATT", "SPA", "IR", "CI", "INC", "SO"],
        "hallucination_vector": [1, 0, 0, 0, 0, 0, 0],
        "hallucination_labels": ["OBJ"],
        "primary_label": "OBJ",
        "coarse_labels": ["Factual Hallucination"],
        "unsupported_visual_claim": True,
        "confidence": "high",
        "claim_checks": [
            {
                "claim_id": "c1",
                "claim": "There is a dog in the image.",
                "source": "final_answer",
                "claim_type": "object_claim",
                "relevance_to_question": "directly_relevant",
                "support_status": "contradicted",
                "fine_label": "OBJ",
                "reason": "The reference answer does not support a dog.",
            }
        ],
        "explanation": "The response claims an absent object is visible.",
    }
    payload.update(overrides)
    return payload


def test_zero_shot_prompt_includes_claim_level_inputs():
    prompt = render_zero_shot_judge_prompt(
        _sample(),
        _response(),
        taxonomy_definition="OBJ means object hallucination.",
    )

    assert "Response Claim Extraction" in prompt
    assert "Task Type: vqa_yes_no" in prompt
    assert "Metadata:" in prompt
    assert "causal_claim" in prompt
    assert "inconsistency_claim" in prompt
    assert "semantic_claim" in prompt
    assert "Is there a dog" in prompt
    assert "OBJ means object hallucination" in prompt
    assert "{question}" not in prompt


def test_zero_shot_prompt_uses_reference_answer_only_as_auxiliary_context():
    prompt = render_zero_shot_judge_prompt(
        _sample(),
        _response(),
        taxonomy_definition="OBJ means object hallucination.",
    )

    assert (
        "Use the ground-truth/reference answer only to identify possible contradictions"
        in prompt
    )
    assert (
        "Do not label a response as hallucinated only because the final answer differs"
        in prompt
    )
    assert "Hallucination labels must be based on unsupported" in prompt


def test_parse_judge_output_recovers_fenced_json_and_normalizes_details():
    raw = "```json\n" + json.dumps(_judge_payload(), ensure_ascii=False) + "\n```"

    details = parse_judge_output(raw)

    assert details["is_hallucination"] is True
    assert details["hallucination_vector"] == [1, 0, 0, 0, 0, 0, 0]
    assert details["hallucination_labels"] == ["OBJ"]
    assert details["primary_label"] == "OBJ"
    assert details["claim_checks"][0]["fine_label"] == "OBJ"


def test_parse_judge_output_recomputes_vector_from_claim_checks():
    details = parse_judge_output(
        json.dumps(
            _judge_payload(
                hallucination_vector=[0, 0, 0, 0, 0, 0, 0],
                hallucination_labels=[],
                primary_label="None",
                claim_checks=[
                    {
                        "claim": "The cup is blue.",
                        "claim_type": "attribute_claim",
                        "support_status": "contradicted",
                        "fine_label": "ATT",
                    },
                    {
                        "claim": "The model infers the area is industrial because buildings are visible.",
                        "claim_type": "causal_claim",
                        "support_status": "unverifiable",
                        "fine_label": "CI",
                    },
                ],
            )
        )
    )

    assert details["hallucination_labels"] == ["ATT", "CI"]
    assert details["hallucination_vector"] == [0, 1, 0, 0, 1, 0, 0]
    assert details["primary_label"] == "ATT"
    assert details["coarse_labels"] == [
        "Factual Hallucination",
        "Logical Hallucination",
    ]


def test_parse_judge_output_uses_claim_checks_over_inconsistent_top_level_labels():
    details = parse_judge_output(
        json.dumps(
            _judge_payload(
                hallucination_vector=[1, 0, 0, 0, 0, 0, 0],
                hallucination_labels=["OBJ"],
                primary_label="OBJ",
                claim_checks=[
                    {
                        "claim": "There is a cat in the image.",
                        "claim_type": "object_claim",
                        "support_status": "supported",
                        "fine_label": "OBJ",
                    }
                ],
            )
        )
    )

    assert details["is_hallucination"] is False
    assert details["hallucination_labels"] == []
    assert details["hallucination_vector"] == [0, 0, 0, 0, 0, 0, 0]
    assert details["primary_label"] == "None"
    assert details["claim_checks"][0]["fine_label"] == "None"


def test_parse_judge_output_derives_fine_label_from_claim_type():
    details = parse_judge_output(
        json.dumps(
            _judge_payload(
                hallucination_vector=[0, 0, 0, 0, 0, 0, 0],
                hallucination_labels=[],
                primary_label="None",
                claim_checks=[
                    {
                        "claim": "There is a dog in the image.",
                        "claim_type": "object_claim",
                        "support_status": "contradicted",
                        "fine_label": "None",
                    }
                ],
            )
        )
    )

    assert details["is_hallucination"] is True
    assert details["hallucination_labels"] == ["OBJ"]
    assert details["hallucination_vector"] == [1, 0, 0, 0, 0, 0, 0]
    assert details["claim_checks"][0]["fine_label"] == "OBJ"


def test_parse_judge_output_empty_claim_checks_override_top_level_labels():
    details = parse_judge_output(
        json.dumps(
            _judge_payload(
                hallucination_vector=[1, 0, 0, 0, 0, 0, 0],
                hallucination_labels=["OBJ"],
                primary_label="OBJ",
                claim_checks=[],
            )
        )
    )

    assert details["is_hallucination"] is False
    assert details["hallucination_labels"] == []
    assert details["hallucination_vector"] == [0, 0, 0, 0, 0, 0, 0]
    assert details["primary_label"] == "None"


def test_parse_judge_output_malformed_claim_checks_falls_back_to_top_level_labels():
    details = parse_judge_output(
        json.dumps(
            _judge_payload(
                hallucination_vector=[0, 0, 0, 0, 1, 0, 0],
                hallucination_labels=["CI"],
                primary_label="CI",
                claim_checks="not a list",
            )
        )
    )

    assert details["is_hallucination"] is True
    assert details["hallucination_labels"] == ["CI"]
    assert details["hallucination_vector"] == [0, 0, 0, 0, 1, 0, 0]
    assert details["primary_label"] == "CI"
    assert details["coarse_labels"] == ["Logical Hallucination"]


def test_parse_judge_output_does_not_derive_hallucination_from_answer_claim():
    details = parse_judge_output(
        json.dumps(
            _judge_payload(
                hallucination_vector=[0, 0, 0, 0, 0, 0, 0],
                hallucination_labels=[],
                primary_label="None",
                claim_checks=[
                    {
                        "claim": "The final answer differs from the reference answer.",
                        "claim_type": "answer_claim",
                        "support_status": "contradicted",
                        "fine_label": "None",
                    }
                ],
            )
        )
    )

    assert details["hallucination_labels"] == []
    assert details["hallucination_vector"] == [0, 0, 0, 0, 0, 0, 0]
    assert details["claim_checks"][0]["fine_label"] == "None"


def test_parse_judge_output_derives_internal_inconsistency_from_claim_type():
    details = parse_judge_output(
        json.dumps(
            _judge_payload(
                hallucination_vector=[0, 0, 0, 0, 0, 0, 0],
                hallucination_labels=[],
                primary_label="None",
                claim_checks=[
                    {
                        "claim": "The response says there are three cars, then concludes there are five cars.",
                        "claim_type": "inconsistency_claim",
                        "support_status": "contradicted",
                        "fine_label": "None",
                    }
                ],
            )
        )
    )

    assert details["hallucination_labels"] == ["INC"]
    assert details["hallucination_vector"] == [0, 0, 0, 0, 0, 1, 0]
    assert details["claim_checks"][0]["fine_label"] == "INC"


def test_parse_judge_output_derives_semantic_over_attribution_from_claim_type():
    details = parse_judge_output(
        json.dumps(
            _judge_payload(
                hallucination_vector=[0, 0, 0, 0, 0, 0, 0],
                hallucination_labels=[],
                primary_label="None",
                claim_checks=[
                    {
                        "claim": "The building is a hospital.",
                        "claim_type": "semantic_claim",
                        "support_status": "unverifiable",
                        "fine_label": "None",
                    }
                ],
            )
        )
    )

    assert details["hallucination_labels"] == ["SO"]
    assert details["hallucination_vector"] == [0, 0, 0, 0, 0, 0, 1]
    assert details["claim_checks"][0]["fine_label"] == "SO"


def test_parse_judge_output_returns_unclear_fallback_for_malformed_text():
    details = parse_judge_output("not json")

    assert details["answer_correct"] is None
    assert details["is_hallucination"] is None
    assert details["primary_label"] == "Unclear"
    assert details["confidence"] == "low"


def test_details_to_detector_result_preserves_claim_details():
    raw = json.dumps(_judge_payload())
    details = parse_judge_output(raw)

    result = details_to_detector_result(
        sample=_sample(),
        response=_response(),
        details=details,
        raw_judge_response=raw,
        run_id="judge_run_1",
    )
    row = result.to_dict()
    restored = DetectorResult.from_dict(row)

    assert row["taxonomy"] == {"coarse": "Factual", "fine": "OBJ"}
    assert row["model_response_id"] == "model_run_1:pope_1"
    assert row["details"]["hallucination_vector"] == [1, 0, 0, 0, 0, 0, 0]
    assert restored.details["claim_checks"][0]["fine_label"] == "OBJ"


def test_judge_response_attaches_image_and_returns_detector_result(tmp_path: Path):
    raw = json.dumps(_judge_payload())
    client = FakeJudgeClient(raw)
    sample = _sample("data/raw/pope/sample.jpg")

    result = judge_response(
        sample,
        _response(),
        client=client,
        run_id="judge_run_1",
        path_base=tmp_path,
        image_root="data/raw",
    )

    assert result.detector == "response_claim_zero_shot_judge"
    assert result.is_hallucination is True
    assert result.taxonomy.fine == "OBJ"
    assert result.dataset == "pope"
    assert result.model == "qwen/qwen3-vl-8b-instruct"
    assert (
        client.calls[0]["image_path"]
        == (tmp_path / "data/raw/pope/sample.jpg").resolve()
    )
    assert client.calls[0]["allowed_image_root"] == (tmp_path / "data/raw").resolve()
    assert client.calls[0]["model"] == "gpt-5.4-mini"


def test_detect_file_requires_resume_or_overwrite(tmp_path: Path):
    output = tmp_path / "judge.jsonl"
    output.write_text('{"model_response_id":"model_run_1:pope_1"}\n', encoding="utf-8")

    with pytest.raises(FileExistsError):
        zero_shot_judge._prepare_output_rows(output, overwrite=False, resume=False)


def test_detect_file_resume_skips_existing_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    samples = tmp_path / "samples.jsonl"
    responses = tmp_path / "responses.jsonl"
    output = tmp_path / "judge.jsonl"
    write_jsonl(samples, [_sample("data/raw/pope/sample.jpg").to_dict()])
    write_jsonl(responses, [_response().to_dict()])
    write_jsonl(
        output,
        [
            {
                "run_id": "judge_run_1",
                "sample_id": "pope_1",
                "model_response_id": "model_run_1:pope_1",
                "detector": "response_claim_zero_shot_judge",
                "answer_correct": False,
                "is_hallucination": True,
                "taxonomy": {"coarse": "Factual", "fine": "OBJ"},
                "unsupported_visual_claim": True,
                "confidence": "high",
                "explanation": "already judged",
                "raw_judge_response": "{}",
                "dataset": "pope",
                "model": "qwen/qwen3-vl-8b-instruct",
                "prompt_type": "evidence_grounded_cot",
                "details": _judge_payload(),
            }
        ],
    )

    class UnexpectedClient:
        def __init__(self, config: Any) -> None:
            raise AssertionError("client should not be created for skipped responses")

    monkeypatch.setattr(zero_shot_judge, "OpenAICompatibleClient", UnexpectedClient)

    zero_shot_judge.detect_file(
        samples,
        responses,
        output,
        provider="openrouter_qwen3_vl_instruct",
        run_id="judge_run_1",
        resume=True,
    )

    rows = list(read_jsonl(output))
    assert len(rows) == 1
    assert rows[0]["explanation"] == "already judged"

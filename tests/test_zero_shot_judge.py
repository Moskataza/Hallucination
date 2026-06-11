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


def _response(sample_id: str = "pope_1") -> ModelResponse:
    return ModelResponse(
        run_id="model_run_1",
        sample_id=sample_id,
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


def test_zero_shot_prompt_defines_updated_taxonomy():
    prompt = render_zero_shot_judge_prompt(
        _sample(),
        _response(),
        taxonomy_definition=Path("prompts/judge/taxonomy_definition.txt").read_text(
            encoding="utf-8"
        ),
    )

    assert "image facts, the question requirements" in prompt
    assert "truncated response with no explicit final answer" in prompt
    assert "CI: Context Inconsistency" in prompt
    assert "INC: Internal Inconsistency" in prompt
    assert "SO: Semantic Over-attribution / Subjective Opinion" in prompt
    assert "answer_claim alone is not a hallucination label" in prompt


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

    assert details["hallucination_labels"] == ["ATT", "SO"]
    assert details["hallucination_vector"] == [0, 1, 0, 0, 0, 0, 1]
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
    assert details["raw_normalized_mismatch"] is True


def test_parse_judge_output_handles_not_applicable_and_non_claim():
    details = parse_judge_output(
        json.dumps(
            _judge_payload(
                is_hallucination=False,
                hallucination_vector=[0, 0, 0, 0, 0, 0, 0],
                hallucination_labels=[],
                primary_label="None",
                unsupported_visual_claim=False,
                claim_checks=[
                    {
                        "claim": "The response says it cannot determine the answer.",
                        "claim_type": "non_claim",
                        "support_status": "supported",
                        "fine_label": "IR",
                        "evidence_source": "none",
                    },
                    {
                        "claim": "Yes.",
                        "claim_type": "answer_claim",
                        "support_status": "not_applicable",
                        "fine_label": "OBJ",
                        "evidence_source": "none",
                    },
                ],
            )
        )
    )

    assert details["is_hallucination"] is False
    assert details["hallucination_labels"] == []
    assert details["claim_checks"][0]["claim_type"] == "non_claim"
    assert details["claim_checks"][0]["support_status"] == "not_applicable"
    assert details["claim_checks"][0]["fine_label"] == "None"
    assert details["claim_checks"][1]["fine_label"] == "None"


def test_parse_judge_output_normalizes_unsupported_alias_and_preserves_audit_fields():
    details = parse_judge_output(
        json.dumps(
            _judge_payload(
                summary_consistent_with_claims=False,
                aggregation_rule="claim_checks_source_of_truth",
                claim_checks=[
                    {
                        "claim": "There is a dog in the image.",
                        "claim_type": "object_claim",
                        "support_status": "unsupported",
                        "fine_label": "OBJ",
                        "evidence_source": "image",
                    }
                ],
            )
        )
    )

    assert details["is_hallucination"] is True
    assert details["hallucination_labels"] == ["OBJ"]
    assert details["claim_checks"][0]["support_status"] == "unverifiable"
    assert details["claim_checks"][0]["evidence_source"] == "image"
    assert details["summary_consistent_with_claims"] is False
    assert details["aggregation_rule"] == "claim_checks_source_of_truth"


def test_parse_judge_output_normalizes_claim_type_aliases():
    details = parse_judge_output(
        json.dumps(
            _judge_payload(
                is_hallucination=False,
                hallucination_vector=[0, 0, 0, 0, 0, 0, 0],
                hallucination_labels=[],
                primary_label="None",
                unsupported_visual_claim=False,
                claim_checks=[
                    {
                        "claim": "Yes.",
                        "claim_type": "answer-claim",
                        "support_status": "contradicted",
                        "fine_label": "None",
                    },
                    {
                        "claim": "The response gives formatting text.",
                        "claim_type": "non-claim",
                        "support_status": "supported",
                        "fine_label": "IR",
                    },
                    {
                        "claim": "There is a dog in the image.",
                        "claim_type": "object claim",
                        "support_status": "contradicted",
                        "fine_label": "None",
                    },
                ],
            )
        )
    )

    assert details["is_hallucination"] is True
    assert details["hallucination_labels"] == ["OBJ"]
    assert details["claim_checks"][0]["claim_type"] == "answer_claim"
    assert details["claim_checks"][0]["fine_label"] == "None"
    assert details["claim_checks"][1]["claim_type"] == "non_claim"
    assert details["claim_checks"][1]["support_status"] == "not_applicable"
    assert details["claim_checks"][1]["fine_label"] == "None"
    assert details["claim_checks"][2]["claim_type"] == "object_claim"
    assert details["claim_checks"][2]["fine_label"] == "OBJ"


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


def test_parse_judge_output_preserves_answer_claim_consistency_exceptions():
    details = parse_judge_output(
        json.dumps(
            _judge_payload(
                hallucination_vector=[0, 0, 0, 0, 0, 0, 0],
                hallucination_labels=[],
                primary_label="None",
                claim_checks=[
                    {
                        "claim": "The reasoning supports A, but the final answer is B.",
                        "claim_type": "answer_claim",
                        "support_status": "contradicted",
                        "fine_label": "INC",
                    },
                    {
                        "claim": "The answer violates the required option format.",
                        "claim_type": "answer_claim",
                        "support_status": "contradicted",
                        "fine_label": "CI",
                    },
                ],
            )
        )
    )

    assert details["hallucination_labels"] == ["CI", "INC"]
    assert details["hallucination_vector"] == [0, 0, 0, 0, 1, 1, 0]


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
    assert client.calls[0]["response_format"]["type"] == "json_schema"
    schema = client.calls[0]["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["is_hallucination"] == {"type": "boolean"}
    assert "hallucination_labels" not in schema["properties"]
    assert "hallucination_labels" not in schema["required"]
    assert result.details["hallucination_labels"] == ["OBJ"]
    assert schema["additionalProperties"] is False


def test_judge_response_falls_back_when_response_format_is_unsupported(tmp_path: Path):
    raw = json.dumps(_judge_payload())

    class FallbackClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def chat_completion(self, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(kwargs)
            if kwargs.get("response_format") is not None:
                raise RuntimeError("Unsupported parameter: response_format")
            return {"choices": [{"message": {"content": raw}}]}

    client = FallbackClient()

    result = judge_response(
        _sample("data/raw/pope/sample.jpg"),
        _response(),
        client=client,
        run_id="judge_run_1",
        path_base=tmp_path,
        image_root="data/raw",
    )

    assert result.is_hallucination is True
    assert len(client.calls) == 2
    assert client.calls[0]["response_format"]["type"] == "json_schema"
    assert "response_format" not in client.calls[1]


def test_judge_response_does_not_fallback_for_schema_validation_errors(tmp_path: Path):
    class SchemaErrorClient:
        def chat_completion(self, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("Invalid json_schema: additionalProperties is required")

    with pytest.raises(RuntimeError, match="Invalid json_schema"):
        judge_response(
            _sample("data/raw/pope/sample.jpg"),
            _response(),
            client=SchemaErrorClient(),
            run_id="judge_run_1",
            path_base=tmp_path,
            image_root="data/raw",
        )


def test_detect_file_requires_resume_or_overwrite(tmp_path: Path):
    output = tmp_path / "judge.jsonl"
    output.write_text('{"model_response_id":"model_run_1:pope_1"}\n', encoding="utf-8")

    with pytest.raises(FileExistsError):
        zero_shot_judge._prepare_output_rows(
            output, samples={}, overwrite=False, resume=False
        )


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
                "details": {
                    **_judge_payload(),
                    "judge_provider": "openrouter_qwen3_vl_instruct",
                },
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


def test_detect_file_resume_retries_judge_failure_fallback(
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
                "answer_correct": None,
                "is_hallucination": None,
                "taxonomy": {"coarse": "Unclear", "fine": "Unclear"},
                "unsupported_visual_claim": None,
                "confidence": "low",
                "explanation": "Judge inference failed: Model API request failed with status 402",
                "raw_judge_response": "",
                "dataset": "pope",
                "model": "qwen/qwen3-vl-8b-instruct",
                "prompt_type": "evidence_grounded_cot",
                "details": {
                    "fallback_source": "judge_inference_failed",
                    "explanation": "Judge inference failed: Model API request failed with status 402",
                },
            }
        ],
    )

    class RecoveringClient:
        def __init__(self, config: Any) -> None:
            self.config = config

        def chat_completion(self, **kwargs: Any) -> dict[str, Any]:
            return {"choices": [{"message": {"content": json.dumps(_judge_payload())}}]}

    monkeypatch.setattr(zero_shot_judge, "OpenAICompatibleClient", RecoveringClient)

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
    assert rows[0]["answer_correct"] is False
    assert rows[0]["is_hallucination"] is True
    assert rows[0]["taxonomy"] == {"coarse": "Factual", "fine": "OBJ"}
    assert rows[0]["details"]["primary_label"] == "OBJ"


def test_detect_file_does_not_judge_token_limit_truncated_model_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    samples = tmp_path / "samples.jsonl"
    responses = tmp_path / "responses.jsonl"
    output = tmp_path / "judge.jsonl"
    write_jsonl(samples, [_sample("data/raw/pope/sample.jpg").to_dict()])
    truncated_response = _response().to_dict()
    truncated_response["raw_response"] = (
        "Visual Evidence: I see a dog.\nFinal Answer: Yes"
    )
    truncated_response["parsed"] = {
        "visual_evidence": "I see a dog.",
        "reasoning": "",
        "final_answer": "Yes",
        "parse_status": "ok",
    }
    truncated_response["inference_metadata"] = {"finish_reason": "length"}
    write_jsonl(responses, [truncated_response])

    class UnexpectedClient:
        def __init__(self, config: Any) -> None:
            raise AssertionError(
                "judge client should not be created for truncated responses"
            )

    monkeypatch.setattr(zero_shot_judge, "OpenAICompatibleClient", UnexpectedClient)

    with pytest.raises(RuntimeError, match="truncated by the token limit"):
        zero_shot_judge.detect_file(
            samples,
            responses,
            output,
            provider="openrouter_qwen3_vl_instruct",
            run_id="judge_run_1",
            overwrite=True,
        )

    assert list(read_jsonl(output)) == []


def test_detect_file_resume_rejects_stale_detector_row_for_invalid_response(
    tmp_path: Path,
):
    samples = tmp_path / "samples.jsonl"
    responses = tmp_path / "responses.jsonl"
    output = tmp_path / "judge.jsonl"
    write_jsonl(samples, [_sample("data/raw/pope/sample.jpg").to_dict()])
    truncated_response = _response().to_dict()
    truncated_response["inference_metadata"] = {"finish_reason": "length"}
    write_jsonl(responses, [truncated_response])
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
                "explanation": "stale detector row",
                "raw_judge_response": "{}",
                "dataset": "pope",
                "model": "qwen/qwen3-vl-8b-instruct",
                "prompt_type": "evidence_grounded_cot",
                "details": {
                    **_judge_payload(),
                    "judge_provider": "openrouter_qwen3_vl_instruct",
                },
            }
        ],
    )

    with pytest.raises(RuntimeError, match="invalid model response"):
        zero_shot_judge.detect_file(
            samples,
            responses,
            output,
            provider="openrouter_qwen3_vl_instruct",
            run_id="judge_run_1",
            resume=True,
        )

    assert list(read_jsonl(output)) == []


def test_detect_file_does_not_judge_incomplete_model_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    samples = tmp_path / "samples.jsonl"
    responses = tmp_path / "responses.jsonl"
    output = tmp_path / "judge.jsonl"
    write_jsonl(samples, [_sample("data/raw/pope/sample.jpg").to_dict()])
    incomplete_response = _response().to_dict()
    incomplete_response["raw_response"] = (
        "Visual Evidence: The shape appears to be an L.\n"
        "Reasoning: Rechecking the L rotation"
    )
    incomplete_response["parsed"] = {
        "visual_evidence": "The shape appears to be an L.",
        "reasoning": "Rechecking the L rotation",
        "final_answer": "Reasoning: Rechecking the L rotation",
        "parse_status": "fallback",
    }
    write_jsonl(responses, [incomplete_response])

    class UnexpectedClient:
        def __init__(self, config: Any) -> None:
            raise AssertionError(
                "judge client should not be created for incomplete responses"
            )

    monkeypatch.setattr(zero_shot_judge, "OpenAICompatibleClient", UnexpectedClient)

    with pytest.raises(RuntimeError, match="missing an explicit Final Answer"):
        zero_shot_judge.detect_file(
            samples,
            responses,
            output,
            provider="openrouter_qwen3_vl_instruct",
            run_id="judge_run_1",
            overwrite=True,
        )

    assert list(read_jsonl(output)) == []


def test_detect_file_does_not_write_null_row_for_failed_model_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    samples = tmp_path / "samples.jsonl"
    responses = tmp_path / "responses.jsonl"
    output = tmp_path / "judge.jsonl"
    write_jsonl(samples, [_sample("data/raw/pope/sample.jpg").to_dict()])
    failed_response = _response().to_dict()
    failed_response["raw_response"] = ""
    failed_response["parsed"] = {"final_answer": "", "parse_status": "failed"}
    failed_response["inference_metadata"] = {
        "status": "failed",
        "error": "Read timed out",
    }
    write_jsonl(responses, [failed_response])

    class UnexpectedClient:
        def __init__(self, config: Any) -> None:
            raise AssertionError(
                "judge client should not be created for failed responses"
            )

    monkeypatch.setattr(zero_shot_judge, "OpenAICompatibleClient", UnexpectedClient)

    with pytest.raises(RuntimeError, match="invalid model response"):
        zero_shot_judge.detect_file(
            samples,
            responses,
            output,
            provider="openrouter_qwen3_vl_instruct",
            run_id="judge_run_1",
            overwrite=True,
            concurrency=2,
        )

    assert list(read_jsonl(output)) == []


def test_detect_file_does_not_write_null_row_for_judge_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    samples = tmp_path / "samples.jsonl"
    responses = tmp_path / "responses.jsonl"
    output = tmp_path / "judge.jsonl"
    write_jsonl(samples, [_sample("data/raw/pope/sample.jpg").to_dict()])
    response = _response().to_dict()
    response["raw_response"] = (
        "Visual Evidence: trigger judge failure.\nFinal Answer: Yes"
    )
    write_jsonl(responses, [response])

    class FailingClient:
        def __init__(self, config: Any) -> None:
            self.config = config

        def chat_completion(self, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("judge failed after retries")

    monkeypatch.setattr(zero_shot_judge, "OpenAICompatibleClient", FailingClient)

    with pytest.raises(RuntimeError, match="Judge failed after 3 attempts"):
        zero_shot_judge.detect_file(
            samples,
            responses,
            output,
            provider="openrouter_qwen3_vl_instruct",
            run_id="judge_run_1",
            overwrite=True,
        )

    assert list(read_jsonl(output)) == []


def test_detect_file_retries_api_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    samples = tmp_path / "samples.jsonl"
    responses = tmp_path / "responses.jsonl"
    output = tmp_path / "judge.jsonl"
    write_jsonl(samples, [_sample("data/raw/pope/sample.jpg").to_dict()])
    write_jsonl(responses, [_response().to_dict()])

    class RecoveringClient:
        def __init__(self, config: Any) -> None:
            self.calls = 0

        def chat_completion(self, **kwargs: Any) -> dict[str, Any]:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("Model API request failed with status 402")
            return {"choices": [{"message": {"content": json.dumps(_judge_payload())}}]}

    monkeypatch.setattr(zero_shot_judge, "OpenAICompatibleClient", RecoveringClient)

    zero_shot_judge.detect_file(
        samples,
        responses,
        output,
        provider="openrouter_qwen3_vl_instruct",
        run_id="judge_run_1",
        overwrite=True,
    )

    rows = list(read_jsonl(output))
    assert len(rows) == 1
    assert rows[0]["answer_correct"] is False
    assert rows[0]["taxonomy"] == {"coarse": "Factual", "fine": "OBJ"}


def test_detect_file_retries_null_required_judge_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    samples = tmp_path / "samples.jsonl"
    responses = tmp_path / "responses.jsonl"
    output = tmp_path / "judge.jsonl"
    write_jsonl(samples, [_sample("data/raw/pope/sample.jpg").to_dict()])
    write_jsonl(responses, [_response().to_dict()])

    class RecoveringClient:
        def __init__(self, config: Any) -> None:
            self.calls = 0

        def chat_completion(self, **kwargs: Any) -> dict[str, Any]:
            self.calls += 1
            if self.calls == 1:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    _judge_payload(
                                        answer_correct=None,
                                        is_hallucination=None,
                                    )
                                )
                            }
                        }
                    ]
                }
            return {"choices": [{"message": {"content": json.dumps(_judge_payload())}}]}

    monkeypatch.setattr(zero_shot_judge, "OpenAICompatibleClient", RecoveringClient)

    zero_shot_judge.detect_file(
        samples,
        responses,
        output,
        provider="openrouter_qwen3_vl_instruct",
        run_id="judge_run_1",
        overwrite=True,
    )

    rows = list(read_jsonl(output))
    assert len(rows) == 1
    assert rows[0]["answer_correct"] is False
    assert rows[0]["is_hallucination"] is True
    assert rows[0]["taxonomy"] == {"coarse": "Factual", "fine": "OBJ"}


def test_detect_file_retries_null_hallucination_even_with_answer_correct(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    samples = tmp_path / "samples.jsonl"
    responses = tmp_path / "responses.jsonl"
    output = tmp_path / "judge.jsonl"
    write_jsonl(samples, [_sample("data/raw/pope/sample.jpg").to_dict()])
    write_jsonl(responses, [_response().to_dict()])

    class RecoveringClient:
        def __init__(self, config: Any) -> None:
            self.calls = 0

        def chat_completion(self, **kwargs: Any) -> dict[str, Any]:
            self.calls += 1
            if self.calls == 1:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    _judge_payload(
                                        answer_correct=False,
                                        is_hallucination=None,
                                    )
                                )
                            }
                        }
                    ]
                }
            return {"choices": [{"message": {"content": json.dumps(_judge_payload())}}]}

    monkeypatch.setattr(zero_shot_judge, "OpenAICompatibleClient", RecoveringClient)

    zero_shot_judge.detect_file(
        samples,
        responses,
        output,
        provider="openrouter_qwen3_vl_instruct",
        run_id="judge_run_1",
        overwrite=True,
    )

    rows = list(read_jsonl(output))
    assert len(rows) == 1
    assert rows[0]["answer_correct"] is False
    assert rows[0]["is_hallucination"] is True
    assert rows[0]["details"]["raw_is_hallucination"] is True


def test_detect_file_allows_null_answer_correct_for_unavailable_reference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    samples = tmp_path / "samples.jsonl"
    responses = tmp_path / "responses.jsonl"
    output = tmp_path / "judge.jsonl"
    mathvista_sample = _sample("data/raw/mathvista/sample.jpg").to_dict()
    mathvista_sample.update(
        {
            "sample_id": "mathvista_1",
            "dataset": "mathvista",
            "task_type": "visual_math_reasoning",
            "question": "What is the answer?",
            "reference_answer": "UNAVAILABLE",
        }
    )
    mathvista_response = _response("mathvista_1").to_dict()
    mathvista_response.update(
        {
            "dataset": "mathvista",
            "raw_response": "Final Answer: I cannot determine from the image.",
        }
    )
    write_jsonl(samples, [mathvista_sample])
    write_jsonl(responses, [mathvista_response])

    class UnavailableAnswerClient:
        def __init__(self, config: Any) -> None:
            self.calls = 0

        def chat_completion(self, **kwargs: Any) -> dict[str, Any]:
            self.calls += 1
            payload = _judge_payload(
                answer_correct=None,
                is_hallucination=self.calls > 1,
                hallucination_vector=[0, 0, 0, 0, 0, 0, 0],
                hallucination_labels=[],
                primary_label="None",
                coarse_labels=["None"],
                unsupported_visual_claim=False,
                claim_checks=[],
                explanation="No unsupported visual claim is present.",
            )
            if self.calls == 1:
                payload["is_hallucination"] = None
            return {"choices": [{"message": {"content": json.dumps(payload)}}]}

    monkeypatch.setattr(
        zero_shot_judge, "OpenAICompatibleClient", UnavailableAnswerClient
    )

    zero_shot_judge.detect_file(
        samples,
        responses,
        output,
        provider="openrouter_qwen3_vl_instruct",
        run_id="judge_run_1",
        overwrite=True,
    )

    rows = list(read_jsonl(output))
    assert len(rows) == 1
    assert rows[0]["answer_correct"] is None
    assert rows[0]["is_hallucination"] is False
    assert rows[0]["taxonomy"] == {"coarse": "None", "fine": "None"}


def test_detect_file_retries_malformed_judge_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    samples = tmp_path / "samples.jsonl"
    responses = tmp_path / "responses.jsonl"
    output = tmp_path / "judge.jsonl"
    write_jsonl(samples, [_sample("data/raw/pope/sample.jpg").to_dict()])
    write_jsonl(responses, [_response().to_dict()])

    class RecoveringClient:
        def __init__(self, config: Any) -> None:
            self.calls = 0

        def chat_completion(self, **kwargs: Any) -> dict[str, Any]:
            self.calls += 1
            if self.calls == 1:
                return {"choices": [{"message": {"content": "not json"}}]}
            return {"choices": [{"message": {"content": json.dumps(_judge_payload())}}]}

    monkeypatch.setattr(zero_shot_judge, "OpenAICompatibleClient", RecoveringClient)

    zero_shot_judge.detect_file(
        samples,
        responses,
        output,
        provider="openrouter_qwen3_vl_instruct",
        run_id="judge_run_1",
        overwrite=True,
    )

    rows = list(read_jsonl(output))
    assert len(rows) == 1
    assert rows[0]["answer_correct"] is False
    assert rows[0]["taxonomy"] == {"coarse": "Factual", "fine": "OBJ"}


def test_detect_file_concurrent_writes_results_in_response_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    samples = tmp_path / "samples.jsonl"
    responses = tmp_path / "responses.jsonl"
    output = tmp_path / "judge.jsonl"
    write_jsonl(
        samples,
        [
            _sample("data/raw/pope/sample.jpg").to_dict(),
            {**_sample("data/raw/pope/sample.jpg").to_dict(), "sample_id": "pope_2"},
            {**_sample("data/raw/pope/sample.jpg").to_dict(), "sample_id": "pope_3"},
        ],
    )
    write_jsonl(
        responses,
        [
            _response("pope_1").to_dict(),
            _response("pope_2").to_dict(),
            _response("pope_3").to_dict(),
        ],
    )

    class FakeClient:
        def __init__(self, config: Any) -> None:
            self.config = config

        def chat_completion(self, **kwargs: Any) -> dict[str, Any]:
            return {"choices": [{"message": {"content": json.dumps(_judge_payload())}}]}

    monkeypatch.setattr(zero_shot_judge, "OpenAICompatibleClient", FakeClient)

    zero_shot_judge.detect_file(
        samples,
        responses,
        output,
        provider="openrouter_qwen3_vl_instruct",
        run_id="judge_run_1",
        overwrite=True,
        concurrency=2,
    )

    rows = list(read_jsonl(output))
    assert [row["model_response_id"] for row in rows] == [
        "model_run_1:pope_1",
        "model_run_1:pope_2",
        "model_run_1:pope_3",
    ]


def test_detect_file_concurrent_failure_persists_only_successful_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    samples = tmp_path / "samples.jsonl"
    responses = tmp_path / "responses.jsonl"
    output = tmp_path / "judge.jsonl"
    write_jsonl(
        samples,
        [
            _sample("data/raw/pope/sample.jpg").to_dict(),
            {**_sample("data/raw/pope/sample.jpg").to_dict(), "sample_id": "pope_2"},
            {**_sample("data/raw/pope/sample.jpg").to_dict(), "sample_id": "pope_3"},
        ],
    )
    second_response = _response("pope_2").to_dict()
    second_response["raw_response"] = (
        "Visual Evidence: trigger judge failure.\nFinal Answer: Yes"
    )
    write_jsonl(
        responses,
        [_response("pope_1").to_dict(), second_response, _response("pope_3").to_dict()],
    )

    class FakeClient:
        def __init__(self, config: Any) -> None:
            self.config = config

        def chat_completion(self, **kwargs: Any) -> dict[str, Any]:
            if "trigger judge failure" in kwargs["prompt"]:
                raise RuntimeError("judge failed after retries")
            return {"choices": [{"message": {"content": json.dumps(_judge_payload())}}]}

    monkeypatch.setattr(zero_shot_judge, "OpenAICompatibleClient", FakeClient)

    with pytest.raises(RuntimeError, match="model_run_1:pope_2"):
        zero_shot_judge.detect_file(
            samples,
            responses,
            output,
            provider="openrouter_qwen3_vl_instruct",
            run_id="judge_run_1",
            overwrite=True,
            concurrency=3,
        )

    rows = list(read_jsonl(output))
    assert [row["model_response_id"] for row in rows] == [
        "model_run_1:pope_1",
        "model_run_1:pope_3",
    ]
    assert all(row["answer_correct"] is not None for row in rows)

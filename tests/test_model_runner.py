from pathlib import Path
import threading
from typing import Any

import pytest
import requests

from src.datasets.jsonl import read_json_records, write_jsonl

from src.models.openai_compatible import (
    OpenAICompatibleClient,
    build_chat_payload,
    extract_message_text,
    extract_native_reasoning,
    get_provider_config,
    image_to_data_url,
)
from src.models.run_inference import (
    _prepare_output_rows,
    model_response_invalid_reason,
    render_prompt,
    resolve_prompt_type,
    resolve_sample_image_path,
    run_inference,
)
from src.models.run_one_tenth_inference import (
    InferenceGroup,
    count_completed_target_prefix,
    count_completed_target_samples,
    resume_groups,
)


def test_image_to_data_url_encodes_local_file(tmp_path: Path):
    image = tmp_path / "sample.jpg"
    image.write_bytes(b"abc")

    assert image_to_data_url(image) == "data:image/jpeg;base64,YWJj"


def test_image_to_data_url_rejects_paths_outside_allowed_root(tmp_path: Path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    secret = tmp_path / "secret.jpg"
    secret.write_bytes(b"abc")

    with pytest.raises(ValueError, match="outside allowed root"):
        image_to_data_url(secret, allowed_root=allowed)


def test_image_to_data_url_rejects_unsupported_extension(tmp_path: Path):
    text_file = tmp_path / "sample.txt"
    text_file.write_text("not an image", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported image extension"):
        image_to_data_url(text_file, allowed_root=tmp_path)


def test_image_to_data_url_rejects_oversized_file(tmp_path: Path):
    image = tmp_path / "sample.jpg"
    image.write_bytes(b"abc")

    with pytest.raises(ValueError, match="too large"):
        image_to_data_url(image, allowed_root=tmp_path, max_bytes=2)


def test_build_chat_payload_uses_openai_multimodal_format(tmp_path: Path):
    image = tmp_path / "sample.png"
    image.write_bytes(b"abc")

    payload = build_chat_payload(
        model="qwen-vl-plus",
        prompt="Question?",
        image_path=image,
        temperature=0,
        max_tokens=128,
        allowed_image_root=tmp_path,
    )

    assert payload["model"] == "qwen-vl-plus"
    assert payload["temperature"] == 0
    assert payload["max_tokens"] == 128
    content = payload["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "Question?"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "response_format" not in payload


def test_build_chat_payload_adds_response_format(tmp_path: Path):
    image = tmp_path / "sample.png"
    image.write_bytes(b"abc")
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "judge_result",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
        },
    }

    payload = build_chat_payload(
        model="qwen-vl-plus",
        prompt="Question?",
        image_path=image,
        allowed_image_root=tmp_path,
        response_format=response_format,
    )

    assert payload["response_format"] == response_format


def test_build_chat_payload_adds_openrouter_reasoning_options(tmp_path: Path):
    image = tmp_path / "sample.png"
    image.write_bytes(b"abc")

    payload = build_chat_payload(
        model="qwen/qwen3-vl-8b-thinking",
        prompt="Question?",
        image_path=image,
        native_reasoning=True,
        reasoning_max_tokens=256,
        provider_name="openrouter_qwen3_vl_thinking",
    )

    assert payload["reasoning"] == {
        "enabled": True,
        "exclude": False,
        "max_tokens": 256,
    }
    assert "google" not in payload


def test_build_chat_payload_rejects_native_reasoning_for_unsupported_provider(
    tmp_path: Path,
):
    image = tmp_path / "sample.png"
    image.write_bytes(b"abc")

    with pytest.raises(ValueError, match="Native reasoning is not supported"):
        build_chat_payload(
            model="qwen-vl-plus",
            prompt="Question?",
            image_path=image,
            native_reasoning=True,
            provider_name="qwen",
        )


def test_build_chat_payload_adds_gemini_thinking_options(tmp_path: Path):
    image = tmp_path / "sample.png"
    image.write_bytes(b"abc")

    payload = build_chat_payload(
        model="gemini-2.5-pro",
        prompt="Question?",
        image_path=image,
        native_reasoning=True,
        provider_name="gemini_local",
    )

    assert payload["google"] == {"thinking_config": {"include_thoughts": True}}
    assert "reasoning" not in payload


def test_extract_message_text_handles_string_and_text_parts():
    assert (
        extract_message_text(
            {"choices": [{"message": {"content": "Final Answer: Yes"}}]}
        )
        == "Final Answer: Yes"
    )
    assert (
        extract_message_text(
            {
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "text", "text": "A"},
                                {"type": "text", "text": "thought", "thought": True},
                                {"type": "text", "text": "B"},
                            ]
                        }
                    }
                ]
            }
        )
        == "AB"
    )


def test_extract_native_reasoning_handles_reasoning_fields_and_thought_parts():
    assert extract_native_reasoning(
        {
            "choices": [
                {
                    "message": {
                        "reasoning": "reasoning text",
                        "reasoning_details": [
                            {"type": "reasoning.text", "text": "detail"}
                        ],
                        "content": "Final answer",
                    }
                }
            ]
        }
    ) == {
        "reasoning": "reasoning text",
        "reasoning_details": [{"type": "reasoning.text", "text": "detail"}],
    }
    assert extract_native_reasoning(
        {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "summary", "thought": True},
                            {"type": "text", "text": "final"},
                        ]
                    }
                }
            ]
        }
    ) == {"thought_summary": "summary"}


def test_render_prompt_replaces_question_and_choices():
    prompt = render_prompt(
        "Question: {question}\nChoices:\n{choices}",
        {"question": "Pick one", "choices": ["Yes", "No"]},
    )

    assert "Question: Pick one" in prompt
    assert "(A) Yes" in prompt
    assert "(B) No" in prompt


def test_resolve_sample_image_path_uses_path_base(tmp_path: Path):
    assert (
        resolve_sample_image_path("data/raw/x.jpg", path_base=tmp_path)
        == (tmp_path / "data/raw/x.jpg").resolve()
    )


def test_resolve_prompt_type_infers_from_prompt_path():
    assert resolve_prompt_type("prompts/answer/direct_pope.txt") == "direct"
    assert (
        resolve_prompt_type("prompts/answer/evidence_grounded_cot_pope.txt")
        == "evidence_grounded_cot"
    )


def test_resolve_prompt_type_allows_explicit_type_for_custom_prompt():
    assert resolve_prompt_type("prompts/answer/custom.txt", "direct") == "direct"


def _valid_response_row(
    sample_id: str,
    *,
    run_id: str = "run_1",
    dataset: str = "pope",
    model: str = "gemini-2.5-flash",
    model_type: str = "closed",
    prompt_type: str = "direct",
    provider: str = "gemini_local",
    max_tokens: int = 128,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "sample_id": sample_id,
        "dataset": dataset,
        "model": model,
        "model_type": model_type,
        "prompt_type": prompt_type,
        "prompt_version": "v1",
        "raw_response": "No",
        "parsed": {"parse_status": "ok", "final_answer": "No"},
        "inference_metadata": {
            "provider": provider,
            "max_tokens": max_tokens,
            "temperature": 0,
        },
    }


def test_resolve_prompt_type_rejects_mismatch():
    with pytest.raises(ValueError, match="suggests prompt_type=direct"):
        resolve_prompt_type(
            "prompts/answer/direct_pope.txt",
            "evidence_grounded_cot",
        )


def test_resolve_prompt_type_requires_explicit_type_for_unknown_prompt():
    with pytest.raises(ValueError, match="Cannot infer prompt type"):
        resolve_prompt_type("prompts/answer/custom.txt")


def test_prepare_output_rows_requires_explicit_mode_for_existing_output(tmp_path: Path):
    output = tmp_path / "responses.jsonl"
    write_jsonl(output, [_valid_response_row("x")])

    with pytest.raises(FileExistsError):
        _prepare_output_rows(output, overwrite=False, resume=False)

    rows, sample_ids = _prepare_output_rows(output, overwrite=False, resume=True)
    assert rows == [_valid_response_row("x")]
    assert sample_ids == {"x"}

    rows, sample_ids = _prepare_output_rows(output, overwrite=True, resume=False)
    assert rows == []
    assert sample_ids == set()
    assert output.read_text(encoding="utf-8") == ""


def test_cot_response_without_explicit_final_answer_is_invalid():
    row = {
        "sample_id": "x",
        "prompt_type": "evidence_grounded_cot",
        "raw_response": "Visual Evidence: The shape is an L.\nReasoning: Rechecking the L rotation",
        "parsed": {
            "parse_status": "fallback",
            "final_answer": "Reasoning: Rechecking the L rotation",
        },
        "inference_metadata": {},
    }

    assert (
        model_response_invalid_reason(row)
        == "CoT response is missing an explicit Final Answer section"
    )


def test_cot_response_with_reparsable_markdown_final_answer_is_valid():
    row = {
        "sample_id": "x",
        "prompt_type": "evidence_grounded_cot",
        "raw_response": (
            "1.  **Visual Evidence**: A dog is visible.\n"
            "2.  **Reasoning:** The dog supports answering yes.\n"
            "3.  **Final Answer**: Yes"
        ),
        "parsed": {
            "parse_status": "fallback",
            "final_answer": "3.  **Final Answer**: Yes",
        },
        "inference_metadata": {},
    }

    assert model_response_invalid_reason(row) is None


def test_cot_response_with_heading_final_answer_is_valid():
    row = {
        "sample_id": "x",
        "prompt_type": "evidence_grounded_cot",
        "raw_response": (
            "### 1. Visual Evidence\nA car is visible.\n"
            "### 3. Reasoning\nThe vehicle shape supports the count.\n"
            "### 4. Final Answer\nThere is 1 car visible."
        ),
        "parsed": {
            "parse_status": "fallback",
            "final_answer": "There is 1 car visible.",
        },
        "inference_metadata": {},
    }

    assert model_response_invalid_reason(row) is None


def test_prepare_output_rows_resume_removes_token_limit_truncated_rows(tmp_path: Path):
    output = tmp_path / "responses.jsonl"
    valid = _valid_response_row("valid")
    truncated = {
        **_valid_response_row("truncated"),
        "inference_metadata": {"finish_reason": "length"},
    }
    write_jsonl(output, [valid, truncated])

    rows, sample_ids = _prepare_output_rows(output, overwrite=False, resume=True)

    assert rows == [valid]
    assert sample_ids == {"valid"}


def test_run_inference_saves_native_reasoning_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    image_root = tmp_path / "data/raw"
    image_root.mkdir(parents=True)
    image = image_root / "sample.jpg"
    image.write_bytes(b"abc")
    dataset = tmp_path / "samples.jsonl"
    prompt = tmp_path / "direct_test.txt"
    output = tmp_path / "responses.jsonl"
    write_jsonl(
        dataset,
        [
            {
                "sample_id": "sample_1",
                "dataset": "pope",
                "task_type": "vqa_yes_no",
                "image_path": "data/raw/sample.jpg",
                "question": "Is there a dog?",
                "reference_answer": "no",
            }
        ],
    )
    prompt.write_text("Question: {question}", encoding="utf-8")

    class FakeClient:
        def __init__(self, config: Any, timeout: int = 120) -> None:
            self.config = config
            self.timeout = timeout

        def chat_completion(self, **kwargs: Any) -> dict[str, Any]:
            assert kwargs["native_reasoning"] is True
            assert kwargs["reasoning_max_tokens"] == 128
            return {
                "choices": [
                    {
                        "message": {
                            "content": "Final Answer: No",
                            "reasoning": "native reasoning text",
                        }
                    }
                ],
                "usage": {"total_tokens": 10},
            }

    monkeypatch.setattr("src.models.run_inference.OpenAICompatibleClient", FakeClient)

    run_inference(
        dataset_path=dataset,
        prompt_path=prompt,
        output_path=output,
        provider="openrouter_qwen3_vl_thinking",
        run_id="run_1",
        prompt_type="direct",
        path_base=tmp_path,
        native_reasoning=True,
        reasoning_max_tokens=128,
        overwrite=True,
    )

    rows = list(read_json_records(output))
    assert rows[0]["raw_response"] == "Final Answer: No"
    assert rows[0]["inference_metadata"]["native_reasoning_enabled"] is True
    assert rows[0]["inference_metadata"]["native_reasoning"] == {
        "reasoning": "native reasoning text"
    }


def test_run_inference_records_finish_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    image_root = tmp_path / "data/raw"
    image_root.mkdir(parents=True)
    image = image_root / "sample.jpg"
    image.write_bytes(b"abc")
    dataset = tmp_path / "samples.jsonl"
    prompt = tmp_path / "direct_test.txt"
    output = tmp_path / "responses.jsonl"
    write_jsonl(
        dataset,
        [
            {
                "sample_id": "sample_1",
                "dataset": "pope",
                "task_type": "vqa_yes_no",
                "image_path": "data/raw/sample.jpg",
                "question": "Is there a dog?",
                "reference_answer": "no",
            }
        ],
    )
    prompt.write_text("Question: {question}", encoding="utf-8")

    class FakeClient:
        def __init__(self, config: Any, timeout: int = 120) -> None:
            self.config = config
            self.timeout = timeout

        def chat_completion(self, **kwargs: Any) -> dict[str, Any]:
            return {
                "choices": [
                    {
                        "message": {"content": "Final Answer: No"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 10},
            }

    monkeypatch.setattr("src.models.run_inference.OpenAICompatibleClient", FakeClient)

    run_inference(
        dataset_path=dataset,
        prompt_path=prompt,
        output_path=output,
        provider="gemini_local",
        run_id="run_1",
        prompt_type="direct",
        path_base=tmp_path,
        overwrite=True,
    )

    rows = list(read_json_records(output))
    assert rows[0]["inference_metadata"]["finish_reason"] == "stop"


def test_count_completed_target_samples_uses_existing_target_sample_ids(tmp_path: Path):
    dataset = tmp_path / "samples.jsonl"
    output = tmp_path / "responses.jsonl"
    write_jsonl(
        dataset,
        [{"sample_id": f"sample_{index}"} for index in range(5)],
    )
    write_jsonl(
        output,
        [
            _valid_response_row("sample_0"),
            _valid_response_row("sample_2"),
            {"sample_id": "sample_4", "inference_metadata": {"status": "failed"}},
            _valid_response_row("outside"),
        ],
    )
    group = InferenceGroup(
        run_id="run_1",
        dataset_path=str(dataset),
        prompt_path="prompt.txt",
        output_path=str(output),
        provider="gemini_local",
        limit=3,
        max_tokens=128,
        dataset="pope",
        model="gemini",
        prompt="direct",
    )

    assert count_completed_target_samples(group) == 2
    assert count_completed_target_prefix(group) == 1


def test_count_completed_target_samples_accepts_existing_qwen_provider_rows(
    tmp_path: Path,
):
    dataset = tmp_path / "samples.jsonl"
    output = tmp_path / "responses.jsonl"
    write_jsonl(dataset, [{"sample_id": "sample_0", "dataset": "pope"}])
    write_jsonl(
        output,
        [
            _valid_response_row(
                "sample_0",
                model="qwen/qwen3-vl-8b-instruct",
                model_type="open",
                provider="openrouter_qwen3_vl_instruct",
                max_tokens=256,
            )
        ],
    )
    group = InferenceGroup(
        run_id="run_1",
        dataset_path=str(dataset),
        prompt_path="prompt.txt",
        output_path=str(output),
        provider="qwen",
        limit=1,
        max_tokens=256,
        dataset="pope",
        model="qwen",
        prompt="direct",
    )

    assert count_completed_target_samples(group) == 1
    assert count_completed_target_prefix(group) == 1


def test_run_inference_provider_model_agnostic_resume_accepts_existing_qwen_variants(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    image_root = tmp_path / "data/raw"
    image_root.mkdir(parents=True)
    image = image_root / "sample.jpg"
    image.write_bytes(b"abc")
    dataset = tmp_path / "samples.jsonl"
    prompt = tmp_path / "direct_test.txt"
    output = tmp_path / "responses.jsonl"
    write_jsonl(
        dataset,
        [
            {
                "sample_id": "sample_0",
                "dataset": "pope",
                "task_type": "vqa_yes_no",
                "image_path": "data/raw/sample.jpg",
                "question": "Is there a dog?",
                "reference_answer": "no",
            }
        ],
    )
    prompt.write_text("Question: {question}", encoding="utf-8")
    write_jsonl(
        output,
        [
            _valid_response_row(
                "sample_0",
                model="qwen/qwen3-vl-8b-instruct",
                model_type="open",
                provider="openrouter_qwen3_vl_instruct",
                max_tokens=512,
            )
        ],
    )
    calls = []

    class FakeClient:
        def __init__(self, config: Any, timeout: int = 120) -> None:
            self.config = config
            self.timeout = timeout

        def chat_completion(self, **kwargs: Any) -> dict[str, Any]:
            calls.append(kwargs)
            return {"choices": [{"message": {"content": "No"}}], "usage": {}}

    monkeypatch.setattr("src.models.run_inference.OpenAICompatibleClient", FakeClient)

    run_inference(
        dataset_path=dataset,
        prompt_path=prompt,
        output_path=output,
        provider="qwen",
        run_id="run_1",
        prompt_type="direct",
        path_base=tmp_path,
        limit=1,
        max_tokens=256,
        resume=True,
        resume_provider_model_agnostic=True,
    )

    rows = list(read_json_records(output))
    assert len(rows) == 1
    assert rows[0]["model"] == "qwen/qwen3-vl-8b-instruct"
    assert rows[0]["inference_metadata"]["provider"] == "openrouter_qwen3_vl_instruct"
    assert rows[0]["inference_metadata"]["max_tokens"] == 512
    assert calls == []


def test_run_inference_concurrent_resume_skips_existing_samples(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    image_root = tmp_path / "data/raw"
    image_root.mkdir(parents=True)
    image = image_root / "sample.jpg"
    image.write_bytes(b"abc")
    dataset = tmp_path / "samples.jsonl"
    prompt = tmp_path / "direct_test.txt"
    output = tmp_path / "responses.jsonl"
    write_jsonl(
        dataset,
        [
            {
                "sample_id": f"sample_{index}",
                "dataset": "pope",
                "task_type": "vqa_yes_no",
                "image_path": "data/raw/sample.jpg",
                "question": "Is there a dog?",
                "reference_answer": "no",
            }
            for index in range(4)
        ],
    )
    prompt.write_text("Question: {question}", encoding="utf-8")
    write_jsonl(
        output,
        [
            {
                "run_id": "run_1",
                "sample_id": "sample_0",
                "dataset": "pope",
                "model": "gemini-2.5-flash",
                "model_type": "closed",
                "prompt_type": "direct",
                "prompt_version": "v1",
                "raw_response": "No",
                "parsed": {"final_answer": "No", "parse_status": "ok"},
                "inference_metadata": {
                    "provider": "gemini_local",
                    "max_tokens": 512,
                    "temperature": 0,
                },
            }
        ],
    )
    seen_questions = []

    class FakeClient:
        def __init__(self, config: Any, timeout: int = 120) -> None:
            self.config = config
            self.timeout = timeout

        def chat_completion(self, **kwargs: Any) -> dict[str, Any]:
            seen_questions.append(kwargs["prompt"])
            return {"choices": [{"message": {"content": "No"}}], "usage": {}}

    monkeypatch.setattr("src.models.run_inference.OpenAICompatibleClient", FakeClient)

    run_inference(
        dataset_path=dataset,
        prompt_path=prompt,
        output_path=output,
        provider="gemini_local",
        run_id="run_1",
        prompt_type="direct",
        path_base=tmp_path,
        limit=4,
        resume=True,
        concurrency=2,
    )

    rows = list(read_json_records(output))
    assert [row["sample_id"] for row in rows] == [
        "sample_0",
        "sample_1",
        "sample_2",
        "sample_3",
    ]
    assert len(seen_questions) == 3


def test_run_inference_concurrent_provider_model_agnostic_resume_preserves_qwen_variant(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    image_root = tmp_path / "data/raw"
    image_root.mkdir(parents=True)
    image = image_root / "sample.jpg"
    image.write_bytes(b"abc")
    dataset = tmp_path / "samples.jsonl"
    prompt = tmp_path / "direct_test.txt"
    output = tmp_path / "responses.jsonl"
    write_jsonl(
        dataset,
        [
            {
                "sample_id": f"sample_{index}",
                "dataset": "pope",
                "task_type": "vqa_yes_no",
                "image_path": "data/raw/sample.jpg",
                "question": "Is there a dog?",
                "reference_answer": "no",
            }
            for index in range(2)
        ],
    )
    prompt.write_text("Question: {question}", encoding="utf-8")
    write_jsonl(
        output,
        [
            _valid_response_row(
                "sample_0",
                model="qwen/qwen3-vl-8b-instruct",
                model_type="open",
                provider="openrouter_qwen3_vl_instruct",
                max_tokens=256,
            )
        ],
    )
    calls = []

    class FakeClient:
        def __init__(self, config: Any, timeout: int = 120) -> None:
            self.config = config
            self.timeout = timeout

        def chat_completion(self, **kwargs: Any) -> dict[str, Any]:
            calls.append(kwargs)
            return {"choices": [{"message": {"content": "No"}}], "usage": {}}

    monkeypatch.setattr("src.models.run_inference.OpenAICompatibleClient", FakeClient)

    run_inference(
        dataset_path=dataset,
        prompt_path=prompt,
        output_path=output,
        provider="qwen",
        run_id="run_1",
        prompt_type="direct",
        path_base=tmp_path,
        limit=2,
        max_tokens=256,
        resume=True,
        concurrency=2,
        resume_provider_model_agnostic=True,
    )

    rows = list(read_json_records(output))
    assert [row["sample_id"] for row in rows] == ["sample_0", "sample_1"]
    assert rows[0]["model"] == "qwen/qwen3-vl-8b-instruct"
    assert rows[0]["inference_metadata"]["provider"] == "openrouter_qwen3_vl_instruct"
    assert rows[1]["model"] == "qwen-vl-plus"
    assert len(calls) == 1


def test_run_inference_strict_resume_rejects_gemini_provider_variant(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    image_root = tmp_path / "data/raw"
    image_root.mkdir(parents=True)
    image = image_root / "sample.jpg"
    image.write_bytes(b"abc")
    dataset = tmp_path / "samples.jsonl"
    prompt = tmp_path / "direct_test.txt"
    output = tmp_path / "responses.jsonl"
    write_jsonl(
        dataset,
        [
            {
                "sample_id": "sample_0",
                "dataset": "pope",
                "task_type": "vqa_yes_no",
                "image_path": "data/raw/sample.jpg",
                "question": "Is there a dog?",
                "reference_answer": "no",
            }
        ],
    )
    prompt.write_text("Question: {question}", encoding="utf-8")
    write_jsonl(
        output,
        [
            _valid_response_row(
                "sample_0",
                provider="other_gemini_provider",
                max_tokens=512,
            )
        ],
    )

    calls = []

    class FakeClient:
        def __init__(self, config: Any, timeout: int = 120) -> None:
            self.config = config
            self.timeout = timeout

        def chat_completion(self, **kwargs: Any) -> dict[str, Any]:
            calls.append(kwargs)
            return {"choices": [{"message": {"content": "No"}}], "usage": {}}

    monkeypatch.setattr("src.models.run_inference.OpenAICompatibleClient", FakeClient)

    run_inference(
        dataset_path=dataset,
        prompt_path=prompt,
        output_path=output,
        provider="gemini_local",
        run_id="run_1",
        prompt_type="direct",
        path_base=tmp_path,
        limit=1,
        max_tokens=512,
        resume=True,
    )

    rows = list(read_json_records(output))
    assert len(rows) == 1
    assert rows[0]["inference_metadata"]["provider"] == "gemini_local"
    assert len(calls) == 1


def test_run_inference_resume_retries_and_replaces_failed_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    image_root = tmp_path / "data/raw"
    image_root.mkdir(parents=True)
    image = image_root / "sample.jpg"
    image.write_bytes(b"abc")
    dataset = tmp_path / "samples.jsonl"
    prompt = tmp_path / "direct_test.txt"
    output = tmp_path / "responses.jsonl"
    write_jsonl(
        dataset,
        [
            {
                "sample_id": "sample_0",
                "dataset": "pope",
                "task_type": "vqa_yes_no",
                "image_path": "data/raw/sample.jpg",
                "question": "Question?",
                "reference_answer": "no",
            }
        ],
    )
    prompt.write_text("Question: {question}", encoding="utf-8")
    write_jsonl(
        output,
        [
            {
                "run_id": "run_1",
                "sample_id": "sample_0",
                "dataset": "pope",
                "model": "gemini-2.5-flash",
                "model_type": "closed",
                "prompt_type": "direct",
                "prompt_version": "v1",
                "raw_response": "",
                "parsed": {"final_answer": "unclear", "parse_status": "failed"},
                "inference_metadata": {"status": "failed", "error": "Read timed out"},
            }
        ],
    )
    seen_questions = []

    class FakeClient:
        def __init__(self, config: Any, timeout: int = 120) -> None:
            self.config = config
            self.timeout = timeout

        def chat_completion(self, **kwargs: Any) -> dict[str, Any]:
            seen_questions.append(kwargs["prompt"])
            return {"choices": [{"message": {"content": "No"}}], "usage": {}}

    monkeypatch.setattr("src.models.run_inference.OpenAICompatibleClient", FakeClient)

    run_inference(
        dataset_path=dataset,
        prompt_path=prompt,
        output_path=output,
        provider="gemini_local",
        run_id="run_1",
        prompt_type="direct",
        path_base=tmp_path,
        limit=1,
        resume=True,
        concurrency=1,
    )

    rows = list(read_json_records(output))
    assert len(rows) == 1
    assert rows[0]["sample_id"] == "sample_0"
    assert rows[0]["raw_response"] == "No"
    assert rows[0]["inference_metadata"].get("status") != "failed"
    assert len(seen_questions) == 1


def test_run_inference_concurrent_failure_persists_successful_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    image_root = tmp_path / "data/raw"
    image_root.mkdir(parents=True)
    image = image_root / "sample.jpg"
    image.write_bytes(b"abc")
    dataset = tmp_path / "samples.jsonl"
    prompt = tmp_path / "direct_test.txt"
    output = tmp_path / "responses.jsonl"
    write_jsonl(
        dataset,
        [
            {
                "sample_id": f"sample_{index}",
                "dataset": "pope",
                "task_type": "vqa_yes_no",
                "image_path": "data/raw/sample.jpg",
                "question": f"Question {index}?",
                "reference_answer": "no",
            }
            for index in range(3)
        ],
    )
    prompt.write_text("Question: {question}", encoding="utf-8")

    class FakeClient:
        def __init__(self, config: Any, timeout: int = 120) -> None:
            self.config = config
            self.timeout = timeout

        def chat_completion(self, **kwargs: Any) -> dict[str, Any]:
            if "Question 1?" in kwargs["prompt"]:
                raise RuntimeError("Model API request failed with status 429")
            return {"choices": [{"message": {"content": "No"}}], "usage": {}}

    monkeypatch.setattr("src.models.run_inference.OpenAICompatibleClient", FakeClient)

    with pytest.raises(RuntimeError, match="sample_1"):
        run_inference(
            dataset_path=dataset,
            prompt_path=prompt,
            output_path=output,
            provider="gemini_local",
            run_id="run_1",
            prompt_type="direct",
            path_base=tmp_path,
            limit=3,
            overwrite=True,
            concurrency=3,
        )

    rows = list(read_json_records(output))
    assert [row["sample_id"] for row in rows] == ["sample_0", "sample_2"]


def test_run_inference_record_failures_writes_failed_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    image_root = tmp_path / "data/raw"
    image_root.mkdir(parents=True)
    image = image_root / "sample.jpg"
    image.write_bytes(b"abc")
    dataset = tmp_path / "samples.jsonl"
    prompt = tmp_path / "direct_test.txt"
    output = tmp_path / "responses.jsonl"
    write_jsonl(
        dataset,
        [
            {
                "sample_id": f"sample_{index}",
                "dataset": "pope",
                "task_type": "vqa_yes_no",
                "image_path": "data/raw/sample.jpg",
                "question": f"Question {index}?",
                "reference_answer": "no",
            }
            for index in range(3)
        ],
    )
    prompt.write_text("Question: {question}", encoding="utf-8")

    class FakeClient:
        def __init__(self, config: Any, timeout: int = 120) -> None:
            self.config = config
            self.timeout = timeout

        def chat_completion(self, **kwargs: Any) -> dict[str, Any]:
            if "Question 1?" in kwargs["prompt"]:
                raise RuntimeError("Model API request failed with status 429")
            return {"choices": [{"message": {"content": "No"}}], "usage": {}}

    monkeypatch.setattr("src.models.run_inference.OpenAICompatibleClient", FakeClient)

    with pytest.raises(RuntimeError, match="incomplete or invalid"):
        run_inference(
            dataset_path=dataset,
            prompt_path=prompt,
            output_path=output,
            provider="gemini_local",
            run_id="run_1",
            prompt_type="direct",
            path_base=tmp_path,
            limit=3,
            overwrite=True,
            concurrency=3,
            record_failures=True,
        )

    rows = list(read_json_records(output))
    assert [row["sample_id"] for row in rows] == ["sample_0", "sample_1", "sample_2"]
    failed = rows[1]
    assert failed["parsed"]["parse_status"] == "failed"
    assert failed["inference_metadata"]["status"] == "failed"
    assert "status 429" in failed["inference_metadata"]["error"]


def test_run_inference_passes_request_timeout_to_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    image_root = tmp_path / "data/raw"
    image_root.mkdir(parents=True)
    image = image_root / "sample.jpg"
    image.write_bytes(b"abc")
    dataset = tmp_path / "samples.jsonl"
    prompt = tmp_path / "direct_test.txt"
    output = tmp_path / "responses.jsonl"
    write_jsonl(
        dataset,
        [
            {
                "sample_id": "sample_0",
                "dataset": "pope",
                "task_type": "vqa_yes_no",
                "image_path": "data/raw/sample.jpg",
                "question": "Question?",
                "reference_answer": "no",
            }
        ],
    )
    prompt.write_text("Question: {question}", encoding="utf-8")
    seen_timeouts = []

    class FakeClient:
        def __init__(self, config: Any, timeout: int = 120) -> None:
            self.config = config
            seen_timeouts.append(timeout)

        def chat_completion(self, **kwargs: Any) -> dict[str, Any]:
            return {"choices": [{"message": {"content": "No"}}], "usage": {}}

    monkeypatch.setattr("src.models.run_inference.OpenAICompatibleClient", FakeClient)

    run_inference(
        dataset_path=dataset,
        prompt_path=prompt,
        output_path=output,
        provider="gemini_local",
        run_id="run_1",
        prompt_type="direct",
        path_base=tmp_path,
        limit=1,
        overwrite=True,
        request_timeout_seconds=33,
    )

    assert seen_timeouts == [33]


def test_run_inference_record_failures_writes_read_timeout_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    image_root = tmp_path / "data/raw"
    image_root.mkdir(parents=True)
    image = image_root / "sample.jpg"
    image.write_bytes(b"abc")
    dataset = tmp_path / "samples.jsonl"
    prompt = tmp_path / "direct_test.txt"
    output = tmp_path / "responses.jsonl"
    write_jsonl(
        dataset,
        [
            {
                "sample_id": "sample_0",
                "dataset": "mathvista",
                "task_type": "visual_math_reasoning",
                "image_path": "data/raw/sample.jpg",
                "question": "Question?",
                "reference_answer": "no",
            }
        ],
    )
    prompt.write_text("Question: {question}", encoding="utf-8")

    class FakeClient:
        def __init__(self, config: Any, timeout: int = 120) -> None:
            self.config = config
            self.timeout = timeout

        def chat_completion(self, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("Read timed out. (read timeout=33)")

    monkeypatch.setattr("src.models.run_inference.OpenAICompatibleClient", FakeClient)

    with pytest.raises(RuntimeError, match="incomplete or invalid"):
        run_inference(
            dataset_path=dataset,
            prompt_path=prompt,
            output_path=output,
            provider="gemini_local",
            run_id="run_1",
            prompt_type="direct",
            path_base=tmp_path,
            limit=1,
            overwrite=True,
            record_failures=True,
            request_timeout_seconds=33,
        )

    rows = list(read_json_records(output))
    assert rows[0]["inference_metadata"]["status"] == "failed"
    assert "Read timed out" in rows[0]["inference_metadata"]["error"]


@pytest.mark.parametrize(
    "error_message",
    [
        "Model API request failed with status 500",
        "ConnectionError: connection aborted",
    ],
)
def test_run_inference_record_failures_writes_retryable_transport_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error_message: str
):
    image_root = tmp_path / "data/raw"
    image_root.mkdir(parents=True)
    image = image_root / "sample.jpg"
    image.write_bytes(b"abc")
    dataset = tmp_path / "samples.jsonl"
    prompt = tmp_path / "direct_test.txt"
    output = tmp_path / "responses.jsonl"
    write_jsonl(
        dataset,
        [
            {
                "sample_id": "sample_0",
                "dataset": "mathvista",
                "task_type": "visual_math_reasoning",
                "image_path": "data/raw/sample.jpg",
                "question": "Question?",
                "reference_answer": "no",
            }
        ],
    )
    prompt.write_text("Question: {question}", encoding="utf-8")

    class FakeClient:
        def __init__(self, config: Any, timeout: int = 120) -> None:
            self.config = config
            self.timeout = timeout

        def chat_completion(self, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError(error_message)

    monkeypatch.setattr("src.models.run_inference.OpenAICompatibleClient", FakeClient)

    with pytest.raises(RuntimeError, match="incomplete or invalid"):
        run_inference(
            dataset_path=dataset,
            prompt_path=prompt,
            output_path=output,
            provider="gemini_local",
            run_id="run_1",
            prompt_type="direct",
            path_base=tmp_path,
            limit=1,
            overwrite=True,
            record_failures=True,
        )

    rows = list(read_json_records(output))
    assert rows[0]["inference_metadata"]["status"] == "failed"
    assert error_message in rows[0]["inference_metadata"]["error"]


def test_run_inference_record_failures_uses_wrapped_transport_error_type(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    image_root = tmp_path / "data/raw"
    image_root.mkdir(parents=True)
    image = image_root / "sample.jpg"
    image.write_bytes(b"abc")
    dataset = tmp_path / "samples.jsonl"
    prompt = tmp_path / "direct_test.txt"
    output = tmp_path / "responses.jsonl"
    write_jsonl(
        dataset,
        [
            {
                "sample_id": "sample_0",
                "dataset": "mathvista",
                "task_type": "visual_math_reasoning",
                "image_path": "data/raw/sample.jpg",
                "question": "Question?",
                "reference_answer": "no",
            }
        ],
    )
    prompt.write_text("Question: {question}", encoding="utf-8")

    class FakeClient:
        def __init__(self, config: Any, timeout: int = 120) -> None:
            self.config = config
            self.timeout = timeout

        def chat_completion(self, **kwargs: Any) -> dict[str, Any]:
            raise requests.ConnectionError("temporary network failure")

    monkeypatch.setattr("src.models.run_inference.OpenAICompatibleClient", FakeClient)

    with pytest.raises(RuntimeError, match="incomplete or invalid"):
        run_inference(
            dataset_path=dataset,
            prompt_path=prompt,
            output_path=output,
            provider="gemini_local",
            run_id="run_1",
            prompt_type="direct",
            path_base=tmp_path,
            limit=1,
            overwrite=True,
            record_failures=True,
        )

    rows = list(read_json_records(output))
    assert rows[0]["inference_metadata"]["status"] == "failed"
    assert "temporary network failure" in rows[0]["inference_metadata"]["error"]


def test_run_inference_record_failures_does_not_record_non_rate_limit_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    image_root = tmp_path / "data/raw"
    image_root.mkdir(parents=True)
    image = image_root / "sample.jpg"
    image.write_bytes(b"abc")
    dataset = tmp_path / "samples.jsonl"
    prompt = tmp_path / "direct_test.txt"
    output = tmp_path / "responses.jsonl"
    write_jsonl(
        dataset,
        [
            {
                "sample_id": "sample_0",
                "dataset": "pope",
                "task_type": "vqa_yes_no",
                "image_path": "data/raw/sample.jpg",
                "question": "Question?",
                "reference_answer": "no",
            }
        ],
    )
    prompt.write_text("Question: {question}", encoding="utf-8")

    class FakeClient:
        def __init__(self, config: Any, timeout: int = 120) -> None:
            self.config = config
            self.timeout = timeout

        def chat_completion(self, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("Missing API key environment variable: TEST_KEY")

    monkeypatch.setattr("src.models.run_inference.OpenAICompatibleClient", FakeClient)

    with pytest.raises(RuntimeError, match="Missing API key"):
        run_inference(
            dataset_path=dataset,
            prompt_path=prompt,
            output_path=output,
            provider="gemini_local",
            run_id="run_1",
            prompt_type="direct",
            path_base=tmp_path,
            limit=1,
            overwrite=True,
            concurrency=1,
            record_failures=True,
        )

    assert list(read_json_records(output)) == []


def test_run_inference_concurrent_non_recordable_error_stops_submitting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    image_root = tmp_path / "data/raw"
    image_root.mkdir(parents=True)
    image = image_root / "sample.jpg"
    image.write_bytes(b"abc")
    dataset = tmp_path / "samples.jsonl"
    prompt = tmp_path / "direct_test.txt"
    output = tmp_path / "responses.jsonl"
    write_jsonl(
        dataset,
        [
            {
                "sample_id": f"sample_{index}",
                "dataset": "pope",
                "task_type": "vqa_yes_no",
                "image_path": "data/raw/sample.jpg",
                "question": f"Question {index}?",
                "reference_answer": "no",
            }
            for index in range(5)
        ],
    )
    prompt.write_text("Question: {question}", encoding="utf-8")
    seen_prompts = []
    barrier = threading.Barrier(2)

    class FakeClient:
        def __init__(self, config: Any, timeout: int = 120) -> None:
            self.config = config
            self.timeout = timeout

        def chat_completion(self, **kwargs: Any) -> dict[str, Any]:
            seen_prompts.append(kwargs["prompt"])
            if "Question 0?" in kwargs["prompt"] or "Question 1?" in kwargs["prompt"]:
                barrier.wait(timeout=5)
            if "Question 1?" in kwargs["prompt"]:
                raise RuntimeError("Missing API key environment variable: TEST_KEY")
            return {"choices": [{"message": {"content": "No"}}], "usage": {}}

    monkeypatch.setattr("src.models.run_inference.OpenAICompatibleClient", FakeClient)

    with pytest.raises(RuntimeError, match="Missing API key"):
        run_inference(
            dataset_path=dataset,
            prompt_path=prompt,
            output_path=output,
            provider="gemini_local",
            run_id="run_1",
            prompt_type="direct",
            path_base=tmp_path,
            limit=5,
            overwrite=True,
            concurrency=2,
            record_failures=True,
        )

    assert len(seen_prompts) <= 2
    assert [row["sample_id"] for row in read_json_records(output)] == ["sample_0"]


def test_resume_groups_uses_existing_output_total_and_resume_true(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    dataset = tmp_path / "samples.jsonl"
    prompt = tmp_path / "direct_test.txt"
    output = tmp_path / "responses.jsonl"
    write_jsonl(
        dataset,
        [{"sample_id": f"sample_{index}"} for index in range(5)],
    )
    prompt.write_text("Question: {question}", encoding="utf-8")
    write_jsonl(output, [_valid_response_row("sample_0")])
    group = InferenceGroup(
        run_id="run_1",
        dataset_path=str(dataset),
        prompt_path=str(prompt),
        output_path=str(output),
        provider="gemini_local",
        limit=3,
        max_tokens=128,
        dataset="pope",
        model="gemini",
        prompt="direct",
    )
    calls = []

    def fake_run_inference(**kwargs: Any) -> None:
        calls.append(kwargs)
        rows = list(read_json_records(output))
        target_limit = kwargs["limit"]
        existing = {row["sample_id"] for row in rows}
        for index, sample in enumerate(read_json_records(dataset)):
            if index >= target_limit:
                break
            if sample["sample_id"] not in existing:
                rows.append(_valid_response_row(sample["sample_id"]))
        write_jsonl(output, rows)

    monkeypatch.setattr(
        "src.models.run_one_tenth_inference.run_inference", fake_run_inference
    )

    resume_groups([group], chunk_size=1, concurrency=7, request_timeout_seconds=33)

    assert [call["limit"] for call in calls] == [2, 3]
    assert all(call["resume"] is True for call in calls)
    assert all(call["concurrency"] == 7 for call in calls)
    assert all(call["record_failures"] is True for call in calls)
    assert all(call["request_timeout_seconds"] == 33 for call in calls)
    assert [row["sample_id"] for row in read_json_records(output)] == [
        "sample_0",
        "sample_1",
        "sample_2",
    ]


def test_resume_groups_expands_by_total_when_prefix_has_gap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    dataset = tmp_path / "samples.jsonl"
    prompt = tmp_path / "direct_test.txt"
    output = tmp_path / "responses.jsonl"
    write_jsonl(
        dataset,
        [{"sample_id": f"sample_{index}"} for index in range(5)],
    )
    prompt.write_text("Question: {question}", encoding="utf-8")
    write_jsonl(
        output, [_valid_response_row("sample_0"), _valid_response_row("sample_2")]
    )
    group = InferenceGroup(
        run_id="run_1",
        dataset_path=str(dataset),
        prompt_path=str(prompt),
        output_path=str(output),
        provider="gemini_local",
        limit=4,
        max_tokens=128,
        dataset="pope",
        model="gemini",
        prompt="direct",
    )
    calls = []

    def fake_run_inference(**kwargs: Any) -> None:
        calls.append(kwargs)
        rows = list(read_json_records(output))
        existing = {row["sample_id"] for row in rows}
        target_limit = kwargs["limit"]
        for index, sample in enumerate(read_json_records(dataset)):
            if index >= target_limit:
                break
            if sample["sample_id"] not in existing:
                rows.append(_valid_response_row(sample["sample_id"]))
        write_jsonl(output, rows)

    monkeypatch.setattr(
        "src.models.run_one_tenth_inference.run_inference", fake_run_inference
    )

    resume_groups([group], chunk_size=2, concurrency=7)

    assert [call["limit"] for call in calls] == [4]
    assert [row["sample_id"] for row in read_json_records(output)] == [
        "sample_0",
        "sample_2",
        "sample_1",
        "sample_3",
    ]


def test_resume_groups_continues_after_chunk_error_with_progress(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    dataset = tmp_path / "samples.jsonl"
    prompt = tmp_path / "direct_test.txt"
    output = tmp_path / "responses.jsonl"
    write_jsonl(
        dataset,
        [{"sample_id": f"sample_{index}"} for index in range(4)],
    )
    prompt.write_text("Question: {question}", encoding="utf-8")
    write_jsonl(output, [_valid_response_row("sample_0")])
    group = InferenceGroup(
        run_id="run_1",
        dataset_path=str(dataset),
        prompt_path=str(prompt),
        output_path=str(output),
        provider="gemini_local",
        limit=4,
        max_tokens=128,
        dataset="pope",
        model="gemini",
        prompt="direct",
    )
    calls = []

    def fake_run_inference(**kwargs: Any) -> None:
        calls.append(kwargs)
        rows = list(read_json_records(output))
        existing = {row["sample_id"] for row in rows}
        target_limit = kwargs["limit"]
        for index, sample in enumerate(read_json_records(dataset)):
            if index >= target_limit:
                break
            if sample["sample_id"] not in existing:
                rows.append(_valid_response_row(sample["sample_id"]))
                break
        write_jsonl(output, rows)
        if len(calls) == 1:
            raise RuntimeError("temporary chunk failure")

    monkeypatch.setattr(
        "src.models.run_one_tenth_inference.run_inference", fake_run_inference
    )

    resume_groups([group], chunk_size=2, concurrency=7)

    assert [call["limit"] for call in calls] == [3, 4, 4]
    assert [row["sample_id"] for row in read_json_records(output)] == [
        "sample_0",
        "sample_1",
        "sample_2",
        "sample_3",
    ]


def test_resume_groups_continues_other_groups_when_one_group_has_no_progress(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    dataset_a = tmp_path / "samples_a.jsonl"
    dataset_b = tmp_path / "samples_b.jsonl"
    prompt = tmp_path / "direct_test.txt"
    output_a = tmp_path / "responses_a.jsonl"
    output_b = tmp_path / "responses_b.jsonl"
    write_jsonl(dataset_a, [{"sample_id": "a_0"}])
    write_jsonl(dataset_b, [{"sample_id": "b_0"}])
    prompt.write_text("Question: {question}", encoding="utf-8")
    group_a = InferenceGroup(
        run_id="run_a",
        dataset_path=str(dataset_a),
        prompt_path=str(prompt),
        output_path=str(output_a),
        provider="gemini_local",
        limit=1,
        max_tokens=128,
        dataset="pope",
        model="gemini",
        prompt="direct",
    )
    group_b = InferenceGroup(
        run_id="run_b",
        dataset_path=str(dataset_b),
        prompt_path=str(prompt),
        output_path=str(output_b),
        provider="gemini_local",
        limit=1,
        max_tokens=128,
        dataset="pope",
        model="gemini",
        prompt="direct",
    )
    calls = []

    def fake_run_inference(**kwargs: Any) -> None:
        calls.append(kwargs["run_id"])
        if kwargs["run_id"] == "run_b":
            write_jsonl(output_b, [_valid_response_row("b_0")])

    monkeypatch.setattr(
        "src.models.run_one_tenth_inference.run_inference", fake_run_inference
    )

    with pytest.raises(RuntimeError, match="Resume stalled"):
        resume_groups(
            [group_a, group_b], chunk_size=1, concurrency=7, max_chunk_attempts=1
        )

    assert calls == ["run_a", "run_b"]
    assert count_completed_target_samples(group_b) == 1


def test_resume_groups_retries_when_chunk_only_records_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    dataset = tmp_path / "samples.jsonl"
    prompt = tmp_path / "direct_test.txt"
    output = tmp_path / "responses.jsonl"
    write_jsonl(
        dataset,
        [
            {"sample_id": "sample_0"},
        ],
    )
    prompt.write_text("Question: {question}", encoding="utf-8")
    group = InferenceGroup(
        run_id="run_1",
        dataset_path=str(dataset),
        prompt_path=str(prompt),
        output_path=str(output),
        provider="gemini_local",
        limit=1,
        max_tokens=128,
        dataset="pope",
        model="gemini",
        prompt="direct",
    )
    calls = []

    def fake_run_inference(**kwargs: Any) -> None:
        calls.append(kwargs)
        if len(calls) == 1:
            write_jsonl(
                output,
                [
                    {
                        "sample_id": "sample_0",
                        "inference_metadata": {"status": "failed"},
                    }
                ],
            )
            return
        write_jsonl(output, [_valid_response_row("sample_0")])

    monkeypatch.setattr(
        "src.models.run_one_tenth_inference.run_inference", fake_run_inference
    )

    resume_groups([group], chunk_size=1, concurrency=7, max_chunk_attempts=2)

    assert len(calls) == 2
    assert count_completed_target_samples(group) == 1


def test_run_inference_resume_limit_does_not_process_beyond_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    image_root = tmp_path / "data/raw"
    image_root.mkdir(parents=True)
    image = image_root / "sample.jpg"
    image.write_bytes(b"abc")
    dataset = tmp_path / "samples.jsonl"
    prompt = tmp_path / "direct_test.txt"
    output = tmp_path / "responses.jsonl"
    write_jsonl(
        dataset,
        [
            {
                "sample_id": f"sample_{index}",
                "dataset": "pope",
                "task_type": "vqa_yes_no",
                "image_path": "data/raw/sample.jpg",
                "question": "Is there a dog?",
                "reference_answer": "no",
            }
            for index in range(3)
        ],
    )
    prompt.write_text("Question: {question}", encoding="utf-8")
    write_jsonl(
        output,
        [
            {
                "run_id": "run_1",
                "sample_id": "sample_0",
                "dataset": "pope",
                "model": "gemini-2.5-flash",
                "model_type": "closed",
                "prompt_type": "direct",
                "prompt_version": "v1",
                "raw_response": "No",
                "parsed": {"final_answer": "No", "parse_status": "ok"},
                "inference_metadata": {
                    "provider": "gemini_local",
                    "max_tokens": 512,
                    "temperature": 0,
                },
            }
        ],
    )

    class UnexpectedClient:
        def __init__(self, config: Any, timeout: int = 120) -> None:
            self.config = config
            self.timeout = timeout

        def chat_completion(self, **kwargs: Any) -> dict[str, Any]:
            raise AssertionError("resume should not process samples outside limit")

    monkeypatch.setattr(
        "src.models.run_inference.OpenAICompatibleClient", UnexpectedClient
    )

    run_inference(
        dataset_path=dataset,
        prompt_path=prompt,
        output_path=output,
        provider="gemini_local",
        run_id="run_1",
        prompt_type="direct",
        path_base=tmp_path,
        limit=1,
        resume=True,
    )

    rows = list(read_json_records(output))
    assert [row["sample_id"] for row in rows] == ["sample_0"]


class _FakeResponse:
    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text

    def json(self) -> dict[str, Any]:
        return {"choices": [{"message": {"content": "ok"}}]}


def test_client_retries_transient_connection_errors(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = []

    def fake_post(*args: Any, **kwargs: Any) -> _FakeResponse:
        calls.append((args, kwargs))
        if len(calls) == 1:
            raise requests.ConnectionError("temporary network failure")
        return _FakeResponse(200)

    monkeypatch.setattr("src.models.openai_compatible.requests.post", fake_post)
    monkeypatch.setattr("src.models.openai_compatible.time.sleep", lambda _: None)
    client = OpenAICompatibleClient(
        get_provider_config("gemini_local"), max_retries=1, retry_delay_seconds=0
    )

    response = client._post_with_retries("https://example.test", headers={}, payload={})

    assert response.status_code == 200
    assert len(calls) == 2


def test_client_does_not_retry_bad_request(monkeypatch: pytest.MonkeyPatch):
    calls = []

    def fake_post(*args: Any, **kwargs: Any) -> _FakeResponse:
        calls.append((args, kwargs))
        return _FakeResponse(400)

    monkeypatch.setattr("src.models.openai_compatible.requests.post", fake_post)
    client = OpenAICompatibleClient(get_provider_config("gemini_local"), max_retries=2)

    with pytest.raises(RuntimeError, match="status 400"):
        client._post_with_retries("https://example.test", headers={}, payload={})

    assert len(calls) == 1


def test_client_error_message_includes_response_body(monkeypatch: pytest.MonkeyPatch):
    def fake_post(*args: Any, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(400, text="Unsupported parameter: response_format")

    monkeypatch.setattr("src.models.openai_compatible.requests.post", fake_post)
    client = OpenAICompatibleClient(get_provider_config("gemini_local"), max_retries=0)

    with pytest.raises(RuntimeError, match="Unsupported parameter: response_format"):
        client._post_with_retries("https://example.test", headers={}, payload={})


def test_provider_configs():
    gemini = get_provider_config("gemini_local")
    gpt54 = get_provider_config("gpt54_local")
    instruct = get_provider_config("openrouter_qwen3_vl_instruct")
    thinking = get_provider_config("openrouter_qwen3_vl_thinking")

    assert gemini.default_model == "gemini-2.5-flash"
    assert gemini.base_url_env == "GEMINI_LOCAL_BASE_URL"
    assert gpt54.default_model == "gpt-5.4-mini"
    assert gpt54.api_key_env == "CHATGPT_LOCAL_KEY"
    assert gpt54.base_url_env == "GPT54_LOCAL_BASE_URL"
    assert instruct.base_url == "https://openrouter.ai/api/v1"
    assert instruct.api_key_env == "OPENROUTER_API_KEY"
    assert instruct.default_model == "qwen/qwen3-vl-8b-instruct"
    assert thinking.base_url == "https://openrouter.ai/api/v1"
    assert thinking.api_key_env == "OPENROUTER_API_KEY"
    assert thinking.default_model == "qwen/qwen3-vl-8b-thinking"

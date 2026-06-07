from pathlib import Path
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
    render_prompt,
    resolve_prompt_type,
    resolve_sample_image_path,
    run_inference,
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
    output.write_text('{"sample_id":"x"}\n', encoding="utf-8")

    with pytest.raises(FileExistsError):
        _prepare_output_rows(output, overwrite=False, resume=False)

    rows, sample_ids = _prepare_output_rows(output, overwrite=False, resume=True)
    assert rows == [{"sample_id": "x"}]
    assert sample_ids == {"x"}

    rows, sample_ids = _prepare_output_rows(output, overwrite=True, resume=False)
    assert rows == []
    assert sample_ids == set()
    assert output.read_text(encoding="utf-8") == ""


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
        def __init__(self, config: Any) -> None:
            self.config = config

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
                "inference_metadata": {},
            }
        ],
    )

    class UnexpectedClient:
        def __init__(self, config: Any) -> None:
            self.config = config

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
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code

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


def test_provider_configs():
    gemini = get_provider_config("gemini_local")
    instruct = get_provider_config("openrouter_qwen3_vl_instruct")
    thinking = get_provider_config("openrouter_qwen3_vl_thinking")

    assert gemini.default_model == "gemini-2.5-flash"
    assert instruct.base_url == "https://openrouter.ai/api/v1"
    assert instruct.api_key_env == "OPENROUTER_API_KEY"
    assert instruct.default_model == "qwen/qwen3-vl-8b-instruct"
    assert thinking.base_url == "https://openrouter.ai/api/v1"
    assert thinking.api_key_env == "OPENROUTER_API_KEY"
    assert thinking.default_model == "qwen/qwen3-vl-8b-thinking"

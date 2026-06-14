"""运行多模态模型回答生成，支持并发、失败记录和 resume 恢复。"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import json
from pathlib import Path
import tempfile
from typing import Any, cast

import requests

from src.datasets.jsonl import read_json_records, write_jsonl
from src.datasets.schema import ModelResponse, PromptType
from src.models.openai_compatible import (
    PROVIDERS,
    OpenAICompatibleClient,
    extract_message_text,
    extract_native_reasoning,
    get_provider_config,
)
from src.models.response_parser import parse_response
from src.pipelines.compatibility import is_response_row_compatible

_DEFAULT_MAX_IMAGE_BYTES = 20 * 1024 * 1024


def run_inference(
    *,
    dataset_path: str | Path,
    prompt_path: str | Path,
    output_path: str | Path,
    provider: str,
    run_id: str,
    prompt_type: PromptType,
    model: str | None = None,
    prompt_version: str = "v1",
    limit: int | None = None,
    offset: int = 0,
    temperature: float = 0,
    max_tokens: int = 512,
    path_base: str | Path = ".",
    image_root: str | Path = "data/raw",
    max_image_bytes: int = _DEFAULT_MAX_IMAGE_BYTES,
    native_reasoning: bool = False,
    reasoning_max_tokens: int | None = None,
    overwrite: bool = False,
    resume: bool = False,
    concurrency: int = 1,
    record_failures: bool = False,
    request_timeout_seconds: int = 120,
    request_max_retries: int = 2,
    resume_provider_model_agnostic: bool = False,
) -> None:
    """生成目标样本的模型回答，已存在且兼容的行会在 resume 时跳过。"""

    if overwrite and resume:
        raise ValueError("Use either overwrite or resume, not both.")
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    if request_timeout_seconds < 1:
        raise ValueError("request_timeout_seconds must be at least 1")
    if request_max_retries < 0:
        raise ValueError("request_max_retries must be at least 0")

    provider_config = get_provider_config(provider)
    prompt_template = Path(prompt_path).read_text(encoding="utf-8")
    output = Path(output_path)
    expected_model = model or provider_config.default_model
    rows, existing_sample_ids = _prepare_output_rows(
        output,
        overwrite=overwrite,
        resume=resume,
        run_id=run_id,
        dataset=_sample_dataset(dataset_path),
        model=expected_model,
        model_type=provider_config.model_type,
        prompt_type=prompt_type,
        prompt_version=prompt_version,
        provider=provider,
        max_tokens=max_tokens,
        temperature=temperature,
        provider_model_agnostic=resume_provider_model_agnostic,
    )
    allowed_image_root = (Path(path_base) / image_root).resolve()

    target_sample_ids = _target_sample_ids(dataset_path, offset=offset, limit=limit)
    # 目标样本都已存在时仍做完整校验，防止旧配置结果被误复用。
    if target_sample_ids <= existing_sample_ids:
        _validate_completed_output(
            output,
            target_sample_ids,
            run_id=run_id,
            dataset=_sample_dataset(dataset_path),
            model=expected_model,
            model_type=provider_config.model_type,
            prompt_type=prompt_type,
            prompt_version=prompt_version,
            provider=provider,
            max_tokens=max_tokens,
            temperature=temperature,
            provider_model_agnostic=resume_provider_model_agnostic,
        )
        return

    sample_indices = _sample_indices(dataset_path)
    indexed_rows = [
        (sample_indices.get(str(row["sample_id"]), len(sample_indices)), row)
        for row in rows
    ]
    pending_samples = _collect_pending_samples(
        dataset_path=dataset_path,
        offset=offset,
        limit=limit,
        existing_sample_ids=existing_sample_ids,
    )
    if concurrency == 1:
        for sample_index, sample in pending_samples:
            sample_id = str(sample["sample_id"])
            try:
                response = _infer_sample(
                    sample=sample,
                    provider_config=provider_config,
                    provider=provider,
                    prompt_template=prompt_template,
                    path_base=path_base,
                    allowed_image_root=allowed_image_root,
                    run_id=run_id,
                    prompt_type=prompt_type,
                    prompt_version=prompt_version,
                    model=expected_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_image_bytes=max_image_bytes,
                    native_reasoning=native_reasoning,
                    reasoning_max_tokens=reasoning_max_tokens,
                    request_timeout_seconds=request_timeout_seconds,
                    request_max_retries=request_max_retries,
                )
            except Exception as exc:
                if not _should_record_failure(record_failures, exc):
                    raise
                response = _build_failed_response_row(
                    sample=sample,
                    provider_config=provider_config,
                    provider=provider,
                    run_id=run_id,
                    prompt_type=prompt_type,
                    prompt_version=prompt_version,
                    model=expected_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    error=exc,
                )
            indexed_rows.append((sample_index, response))
            existing_sample_ids.add(sample_id)
            _write_indexed_rows(output, indexed_rows)
        _validate_completed_output(
            output,
            target_sample_ids,
            run_id=run_id,
            dataset=_sample_dataset(dataset_path),
            model=expected_model,
            model_type=provider_config.model_type,
            prompt_type=prompt_type,
            prompt_version=prompt_version,
            provider=provider,
            max_tokens=max_tokens,
            temperature=temperature,
            provider_model_agnostic=resume_provider_model_agnostic,
        )
        return

    _run_inference_concurrently(
        pending_samples=pending_samples,
        concurrency=concurrency,
        provider_config=provider_config,
        provider=provider,
        prompt_template=prompt_template,
        path_base=path_base,
        allowed_image_root=allowed_image_root,
        run_id=run_id,
        prompt_type=prompt_type,
        prompt_version=prompt_version,
        model=expected_model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_image_bytes=max_image_bytes,
        native_reasoning=native_reasoning,
        reasoning_max_tokens=reasoning_max_tokens,
        existing_sample_ids=existing_sample_ids,
        indexed_rows=indexed_rows,
        output=output,
        record_failures=record_failures,
        request_timeout_seconds=request_timeout_seconds,
        request_max_retries=request_max_retries,
    )
    _validate_completed_output(
        output,
        target_sample_ids,
        run_id=run_id,
        dataset=_sample_dataset(dataset_path),
        model=expected_model,
        model_type=provider_config.model_type,
        prompt_type=prompt_type,
        prompt_version=prompt_version,
        provider=provider,
        max_tokens=max_tokens,
        temperature=temperature,
        provider_model_agnostic=resume_provider_model_agnostic,
    )


def _run_inference_concurrently(
    *,
    pending_samples: list[tuple[int, dict[str, Any]]],
    concurrency: int,
    provider_config: Any,
    provider: str,
    prompt_template: str,
    path_base: str | Path,
    allowed_image_root: Path,
    run_id: str,
    prompt_type: PromptType,
    prompt_version: str,
    model: str | None,
    temperature: float,
    max_tokens: int,
    max_image_bytes: int,
    native_reasoning: bool,
    reasoning_max_tokens: int | None,
    existing_sample_ids: set[str],
    indexed_rows: list[tuple[int, dict[str, Any]]],
    output: Path,
    record_failures: bool,
    request_timeout_seconds: int,
    request_max_retries: int,
) -> None:
    """保持固定数量的在途请求，任一请求完成后立即补交下一个样本。"""

    failures = []
    next_sample = 0
    stop_submitting = False
    futures: dict[Future[dict[str, Any]], tuple[int, dict[str, Any]]] = {}

    def submit_next(executor: ThreadPoolExecutor) -> None:
        nonlocal next_sample
        if stop_submitting or next_sample >= len(pending_samples):
            return
        sample_index, sample = pending_samples[next_sample]
        next_sample += 1
        future = executor.submit(
            _infer_sample,
            sample=sample,
            provider_config=provider_config,
            provider=provider,
            prompt_template=prompt_template,
            path_base=path_base,
            allowed_image_root=allowed_image_root,
            run_id=run_id,
            prompt_type=prompt_type,
            prompt_version=prompt_version,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_image_bytes=max_image_bytes,
            native_reasoning=native_reasoning,
            reasoning_max_tokens=reasoning_max_tokens,
            request_timeout_seconds=request_timeout_seconds,
            request_max_retries=request_max_retries,
        )
        futures[future] = (sample_index, sample)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        for _ in range(min(concurrency, len(pending_samples))):
            submit_next(executor)
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            done = done | {future for future in futures if future.done()}
            completed_count = len(done)
            for future in done:
                sample_index, sample = futures.pop(future)
                sample_id = str(sample["sample_id"])
                try:
                    response = future.result()
                except Exception as exc:
                    if (
                        _should_record_failure(record_failures, exc)
                        and sample_id not in existing_sample_ids
                    ):
                        indexed_rows.append(
                            (
                                sample_index,
                                _build_failed_response_row(
                                    sample=sample,
                                    provider_config=provider_config,
                                    provider=provider,
                                    run_id=run_id,
                                    prompt_type=prompt_type,
                                    prompt_version=prompt_version,
                                    model=model,
                                    temperature=temperature,
                                    max_tokens=max_tokens,
                                    error=exc,
                                ),
                            )
                        )
                        existing_sample_ids.add(sample_id)
                        _write_indexed_rows(output, indexed_rows)
                    else:
                        failures.append((sample_id, exc))
                        stop_submitting = True
                else:
                    if sample_id not in existing_sample_ids:
                        indexed_rows.append((sample_index, response))
                        existing_sample_ids.add(sample_id)
                        _write_indexed_rows(output, indexed_rows)
            for _ in range(completed_count):
                submit_next(executor)

    if failures:
        failed_preview = "; ".join(
            f"{sample_id}: {exc}" for sample_id, exc in failures[:10]
        )
        suffix = "" if len(failures) <= 10 else f" ... and {len(failures) - 10} more"
        raise RuntimeError(f"Failed inference for samples: {failed_preview}{suffix}")


def _should_record_failure(record_failures: bool, error: Exception) -> bool:
    """只把可重试的网络/API失败写成临时失败行，供下一次 resume 清理重跑。"""

    if not record_failures:
        return False
    current: BaseException | None = error
    while current is not None:
        if isinstance(current, (requests.ConnectionError, requests.Timeout)):
            return True
        current = current.__cause__ or current.__context__

    text = str(error).lower()
    return (
        "status 429" in text
        or "429" in text
        and "rate" in text
        or "status 5" in text
        or "read timed out" in text
        or "timed out" in text
        or "connectionerror" in text
        or "connection error" in text
        or "connection aborted" in text
        or "connection refused" in text
    )


def _build_failed_response_row(
    *,
    sample: dict[str, Any],
    provider_config: Any,
    provider: str,
    run_id: str,
    prompt_type: PromptType,
    prompt_version: str,
    model: str | None,
    temperature: float,
    max_tokens: int,
    error: Exception,
) -> dict[str, Any]:
    """把可恢复的 API 失败记录为空回答行，下一次 resume 会重新尝试。"""
    error_text = f"Inference failed after retries: {error}"
    response = ModelResponse(
        run_id=run_id,
        sample_id=str(sample["sample_id"]),
        dataset=sample["dataset"],
        model=model or provider_config.default_model,
        model_type=provider_config.model_type,
        prompt_type=prompt_type,
        prompt_version=prompt_version,
        raw_response="",
        parsed=parse_response("", prompt_type),
        inference_metadata={
            "provider": provider,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "error": error_text,
            "status": "failed",
        },
    )
    return response.to_dict()


def _collect_pending_samples(
    *,
    dataset_path: str | Path,
    offset: int,
    limit: int | None,
    existing_sample_ids: set[str],
) -> list[tuple[int, dict[str, Any]]]:
    samples = []
    for index, sample in enumerate(read_json_records(dataset_path)):
        if index < offset:
            continue
        if limit is not None and index >= offset + limit:
            break
        if str(sample["sample_id"]) in existing_sample_ids:
            continue
        samples.append((index, sample))
    return samples


def _sample_dataset(dataset_path: str | Path) -> str:
    rows = list(read_json_records(dataset_path))
    if not rows:
        raise ValueError(f"Dataset is empty: {dataset_path}")
    return str(rows[0].get("dataset", ""))


def _sample_indices(dataset_path: str | Path) -> dict[str, int]:
    return {
        str(sample["sample_id"]): index
        for index, sample in enumerate(read_json_records(dataset_path))
    }


def _target_sample_ids(
    dataset_path: str | Path, *, offset: int, limit: int | None
) -> set[str]:
    return {
        str(sample["sample_id"])
        for index, sample in enumerate(read_json_records(dataset_path))
        if index >= offset and (limit is None or index < offset + limit)
    }


def model_response_invalid_reason(row: dict[str, Any]) -> str | None:
    """解释模型回答为何不能复用，供 resume 清理和结果校验使用。"""
    metadata = row.get("inference_metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    parsed = row.get("parsed")
    if not isinstance(parsed, dict):
        parsed = {}

    if metadata.get("status") == "failed":
        return "model inference failed"
    finish_reason = str(metadata.get("finish_reason", "")).strip().lower()
    if finish_reason in {"length", "max_tokens", "token_limit"}:
        return "model output was truncated by the token limit"
    if not str(row.get("raw_response", "")).strip():
        return "raw response is empty"
    prompt_type = str(row.get("prompt_type", ""))
    if parsed.get("parse_status") != "ok" and prompt_type == "evidence_grounded_cot":
        # 历史结果可能由旧解析器生成，校验时用当前解析器再尝试一次。
        reparsed = parse_response(str(row.get("raw_response", "")), prompt_type)
        if reparsed.parse_status == "ok":
            parsed = reparsed.to_dict()
    if parsed.get("parse_status") == "failed":
        return "response parsing failed"
    if not str(parsed.get("final_answer", "")).strip():
        return "parsed final answer is empty"
    if prompt_type == "evidence_grounded_cot" and parsed.get("parse_status") != "ok":
        return "CoT response is missing an explicit Final Answer section"
    return None


def _is_invalid_response_row(row: dict[str, Any]) -> bool:
    return model_response_invalid_reason(row) is not None


def _validate_completed_output(
    output: Path,
    target_sample_ids: set[str],
    *,
    run_id: str,
    dataset: str,
    model: str,
    model_type: str,
    prompt_type: PromptType,
    prompt_version: str,
    provider: str,
    max_tokens: int,
    temperature: float,
    provider_model_agnostic: bool = False,
) -> None:
    rows = list(read_json_records(output)) if output.exists() else []
    rows_by_sample = {str(row.get("sample_id", "")): row for row in rows}
    missing = sorted(target_sample_ids - set(rows_by_sample))
    invalid = sorted(
        sample_id
        for sample_id in target_sample_ids & set(rows_by_sample)
        if _is_invalid_response_row(rows_by_sample[sample_id])
        or not _is_compatible_existing_response_row(
            rows_by_sample[sample_id],
            run_id=run_id,
            dataset=dataset,
            model=model,
            model_type=model_type,
            prompt_type=prompt_type,
            prompt_version=prompt_version,
            provider=provider,
            max_tokens=max_tokens,
            temperature=temperature,
            provider_model_agnostic=provider_model_agnostic,
        )
    )
    if missing or invalid:
        parts = []
        if missing:
            parts.append(f"missing={missing[:10]}")
        if invalid:
            parts.append(f"invalid={invalid[:10]}")
        raise RuntimeError(
            "Model inference output is incomplete or invalid: " + "; ".join(parts)
        )


def _write_indexed_rows(
    output: Path, indexed_rows: list[tuple[int, dict[str, Any]]]
) -> None:
    rows_by_sample_id = {
        str(row["sample_id"]): (index, row) for index, row in indexed_rows
    }
    rows = [
        row for _, row in sorted(rows_by_sample_id.values(), key=lambda item: item[0])
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=output.parent,
        delete=False,
    ) as file:
        temp_path = Path(file.name)
        try:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
            file.flush()
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
    temp_path.replace(output)


def _extract_finish_reason(api_response: dict[str, Any]) -> str:
    choices = api_response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return ""
    return str(first_choice.get("finish_reason", ""))


def _build_client(
    provider_config: Any,
    *,
    request_timeout_seconds: int,
    request_max_retries: int,
) -> OpenAICompatibleClient:
    try:
        return OpenAICompatibleClient(
            provider_config,
            timeout=request_timeout_seconds,
            max_retries=request_max_retries,
        )
    except TypeError:
        try:
            return OpenAICompatibleClient(
                provider_config, timeout=request_timeout_seconds
            )
        except TypeError:
            return OpenAICompatibleClient(provider_config)


def _infer_sample(
    *,
    sample: dict[str, Any],
    provider_config: Any,
    provider: str,
    prompt_template: str,
    path_base: str | Path,
    allowed_image_root: Path,
    run_id: str,
    prompt_type: PromptType,
    prompt_version: str,
    model: str | None,
    temperature: float,
    max_tokens: int,
    max_image_bytes: int,
    native_reasoning: bool,
    reasoning_max_tokens: int | None,
    request_timeout_seconds: int,
    request_max_retries: int,
) -> dict[str, Any]:
    sample_id = str(sample["sample_id"])
    client = _build_client(
        provider_config,
        request_timeout_seconds=request_timeout_seconds,
        request_max_retries=request_max_retries,
    )
    prompt = render_prompt(prompt_template, sample)
    image_path = resolve_sample_image_path(sample["image_path"], path_base=path_base)
    try:
        api_response = client.chat_completion(
            prompt=prompt,
            image_path=image_path,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            allowed_image_root=allowed_image_root,
            max_image_bytes=max_image_bytes,
            native_reasoning=native_reasoning,
            reasoning_max_tokens=reasoning_max_tokens,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed inference for sample_id={sample_id}: {exc}"
        ) from exc

    raw_response = extract_message_text(api_response)
    native_reasoning_payload = extract_native_reasoning(api_response)
    finish_reason = _extract_finish_reason(api_response)
    inference_metadata: dict[str, Any] = {
        "provider": provider,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "api_usage": api_response.get("usage", {}),
    }
    if finish_reason:
        inference_metadata["finish_reason"] = finish_reason
    if native_reasoning:
        inference_metadata["native_reasoning_enabled"] = True
        inference_metadata["native_reasoning"] = native_reasoning_payload
        if reasoning_max_tokens is not None:
            inference_metadata["reasoning_max_tokens"] = reasoning_max_tokens
    response = ModelResponse(
        run_id=run_id,
        sample_id=sample_id,
        dataset=sample["dataset"],
        model=model or provider_config.default_model,
        model_type=provider_config.model_type,
        prompt_type=prompt_type,
        prompt_version=prompt_version,
        raw_response=raw_response,
        parsed=parse_response(raw_response, prompt_type),
        inference_metadata=inference_metadata,
    )
    return response.to_dict()


def resolve_prompt_type(
    prompt_path: str | Path, prompt_type: str | None = None
) -> PromptType:
    """根据 prompt 文件名推断 direct 或 evidence_grounded_cot 类型。"""
    inferred = _infer_prompt_type_from_path(prompt_path)
    if prompt_type is None:
        if inferred is None:
            raise ValueError(
                f"Cannot infer prompt type from prompt path: {prompt_path}. Use --prompt-type."
            )
        return inferred
    if prompt_type not in {"direct", "evidence_grounded_cot"}:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    resolved = cast(PromptType, prompt_type)
    if inferred is not None and inferred != resolved:
        raise ValueError(
            f"Prompt path suggests prompt_type={inferred}, but got prompt_type={resolved}."
        )
    return resolved


def _infer_prompt_type_from_path(prompt_path: str | Path) -> PromptType | None:
    name = Path(prompt_path).name.lower()
    if name.startswith("evidence_grounded_cot"):
        return "evidence_grounded_cot"
    if name.startswith("direct"):
        return "direct"
    return None


def render_prompt(template: str, sample: dict[str, Any]) -> str:
    """将样本字段填入回答生成 prompt 模板。"""
    choices = sample.get("choices")
    choices_text = ""
    if choices:
        if isinstance(choices, dict):
            choices_text = "\n".join(
                f"({key}) {value}" for key, value in choices.items()
            )
        else:
            choices_text = "\n".join(
                f"({chr(65 + index)}) {value}" for index, value in enumerate(choices)
            )
    return template.replace("{question}", str(sample.get("question", ""))).replace(
        "{choices}", choices_text
    )


def resolve_sample_image_path(
    image_path: str | Path, *, path_base: str | Path = "."
) -> Path:
    """把样本中的相对图片路径解析到真实文件路径。"""
    path = Path(image_path)
    if path.is_absolute():
        return path.resolve()
    return (Path(path_base) / path).resolve()


def _prepare_output_rows(
    output_path: Path,
    *,
    overwrite: bool,
    resume: bool,
    run_id: str | None = None,
    dataset: str | None = None,
    model: str | None = None,
    model_type: str | None = None,
    prompt_type: PromptType | None = None,
    prompt_version: str | None = None,
    provider: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    provider_model_agnostic: bool = False,
) -> tuple[list[dict[str, Any]], set[str]]:
    if overwrite:
        write_jsonl(output_path, [])
        return [], set()
    if not output_path.exists():
        return [], set()
    if not resume:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite or --resume."
        )
    rows = [
        row
        for row in read_json_records(output_path)
        if not _is_invalid_response_row(row)
        and _is_compatible_existing_response_row(
            row,
            run_id=run_id,
            dataset=dataset,
            model=model,
            model_type=model_type,
            prompt_type=prompt_type,
            prompt_version=prompt_version,
            provider=provider,
            max_tokens=max_tokens,
            temperature=temperature,
            provider_model_agnostic=provider_model_agnostic,
        )
    ]
    _write_indexed_rows(
        output_path,
        [(index, row) for index, row in enumerate(rows)],
    )
    return rows, {str(row["sample_id"]) for row in rows}


def _is_compatible_existing_response_row(
    row: dict[str, Any],
    *,
    run_id: str | None,
    dataset: str | None,
    model: str | None,
    model_type: str | None,
    prompt_type: PromptType | None,
    prompt_version: str | None,
    provider: str | None,
    max_tokens: int | None,
    temperature: float | None,
    provider_model_agnostic: bool = False,
) -> bool:
    if provider_model_agnostic:
        return _matches_response_row_without_provider_model(
            row,
            run_id=run_id,
            dataset=dataset,
            prompt_type=prompt_type,
            prompt_version=prompt_version,
            temperature=temperature,
        )
    if None in {
        run_id,
        dataset,
        model,
        model_type,
        prompt_type,
        prompt_version,
        provider,
        max_tokens,
        temperature,
    }:
        return True
    return is_response_row_compatible(
        row,
        run_id=run_id,
        dataset=dataset,
        model=model,
        model_type=model_type,
        prompt_type=prompt_type,
        prompt_version=prompt_version,
        provider=provider,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def _matches_response_row_without_provider_model(
    row: dict[str, Any],
    *,
    run_id: str | None,
    dataset: str | None,
    prompt_type: PromptType | None,
    prompt_version: str | None,
    temperature: float | None,
) -> bool:
    if None in {run_id, dataset, prompt_type, prompt_version, temperature}:
        return True
    if row.get("run_id") != run_id:
        return False
    if row.get("dataset") != dataset:
        return False
    if row.get("prompt_type") != prompt_type:
        return False
    if row.get("prompt_version") != prompt_version:
        return False
    metadata = row.get("inference_metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    return metadata.get("temperature") == temperature


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multimodal model inference with OpenAI-compatible APIs."
    )
    parser.add_argument(
        "--dataset", required=True, help="Unified eval JSONL dataset path."
    )
    parser.add_argument("--prompt", required=True, help="Prompt template path.")
    parser.add_argument(
        "--output", required=True, help="Output model response JSONL path."
    )
    parser.add_argument("--provider", required=True, choices=sorted(PROVIDERS))
    parser.add_argument(
        "--model",
        default=None,
        help="Provider model name. Defaults to provider config.",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--prompt-type",
        default=None,
        choices=["direct", "evidence_grounded_cot"],
        help="Prompt format. If omitted, inferred from the prompt file name.",
    )
    parser.add_argument("--prompt-version", default="v1")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--native-reasoning",
        action="store_true",
        help="Request provider-native reasoning/thought summaries when supported.",
    )
    parser.add_argument(
        "--reasoning-max-tokens",
        type=int,
        default=None,
        help="Optional provider-native reasoning token budget when supported.",
    )
    parser.add_argument("--path-base", default=".")
    parser.add_argument("--image-root", default="data/raw")
    parser.add_argument("--max-image-mb", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--record-failures",
        action="store_true",
        help="Write retryable model failures as failed response rows instead of aborting.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent model requests. File writes remain serialized.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=int,
        default=120,
        help="Per-request HTTP timeout for model API calls.",
    )
    parser.add_argument(
        "--request-max-retries",
        type=int,
        default=2,
        help="Maximum HTTP retries per model request.",
    )
    args = parser.parse_args()

    run_inference(
        dataset_path=args.dataset,
        prompt_path=args.prompt,
        output_path=args.output,
        provider=args.provider,
        run_id=args.run_id,
        prompt_type=resolve_prompt_type(args.prompt, args.prompt_type),
        model=args.model,
        prompt_version=args.prompt_version,
        limit=args.limit,
        offset=args.offset,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        path_base=args.path_base,
        image_root=args.image_root,
        max_image_bytes=args.max_image_mb * 1024 * 1024,
        native_reasoning=args.native_reasoning,
        reasoning_max_tokens=args.reasoning_max_tokens,
        overwrite=args.overwrite,
        resume=args.resume,
        concurrency=args.concurrency,
        record_failures=args.record_failures,
        request_timeout_seconds=args.request_timeout_seconds,
        request_max_retries=args.request_max_retries,
    )


if __name__ == "__main__":
    main()

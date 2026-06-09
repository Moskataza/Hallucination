"""使用 zero-shot LLM judge 将模型回答转成结构化幻觉检测结果。"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Protocol, cast

from src.datasets.jsonl import read_json_records, write_jsonl
from src.datasets.schema import (
    CoarseTaxonomy,
    DetectorResult,
    EvalSample,
    FineTaxonomy,
    ModelResponse,
    TaxonomyLabel,
)
from src.detectors.prompt_rendering import render_prompt_template
from src.detectors.taxonomy import (
    FACTUAL_TYPES,
    infer_coarse_from_fine,
    normalize_fine_label,
)
from src.models.openai_compatible import (
    OpenAICompatibleClient,
    extract_message_text,
    get_provider_config,
)
from src.models.run_inference import (
    model_response_invalid_reason,
    resolve_sample_image_path,
)
from src.pipelines.compatibility import is_detector_row_compatible

_LABEL_ORDER = ["OBJ", "ATT", "SPA", "IR", "CI", "INC", "SO"]
_CLAIM_TYPES = {
    "object_claim",
    "attribute_claim",
    "spatial_claim",
    "reasoning_claim",
    "causal_claim",
    "inconsistency_claim",
    "semantic_claim",
    "answer_claim",
}
_RELEVANCE_VALUES = {
    "directly_relevant",
    "supporting_reasoning",
    "extra_but_verifiable",
    "irrelevant_extra",
    "required_but_missing",
}
_SUPPORT_STATUSES = {"supported", "contradicted", "unverifiable"}
_CLAIM_TYPE_LABELS = {
    "object_claim": "OBJ",
    "attribute_claim": "ATT",
    "spatial_claim": "SPA",
    "reasoning_claim": "IR",
    "causal_claim": "SO",
    "inconsistency_claim": "INC",
    "semantic_claim": "SO",
    "answer_claim": "None",
}
_CONFIDENCE_VALUES = {"high", "medium", "low"}
_DEFAULT_JUDGE_MODEL = "gpt-5.4-mini"
_DEFAULT_MAX_IMAGE_BYTES = 20 * 1024 * 1024
_JUDGE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "hallucination_judge_result",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answer_correct": {"type": ["boolean", "null"]},
                "is_hallucination": {"type": "boolean"},
                "label_order": {
                    "type": "array",
                    "items": {"type": "string", "enum": _LABEL_ORDER},
                },
                "hallucination_vector": {
                    "type": "array",
                    "items": {"type": "integer", "enum": [0, 1]},
                },
                "hallucination_labels": {
                    "type": "array",
                    "items": {"type": "string", "enum": _LABEL_ORDER},
                },
                "primary_label": {
                    "type": "string",
                    "enum": [*_LABEL_ORDER, "None", "Unclear"],
                },
                "coarse_labels": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "Factual Hallucination",
                            "Logical Hallucination",
                            "None",
                            "Unclear",
                        ],
                    },
                },
                "unsupported_visual_claim": {"type": ["boolean", "null"]},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                "claim_checks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim_id": {"type": "string"},
                            "claim": {"type": "string"},
                            "source": {"type": "string"},
                            "claim_type": {
                                "type": "string",
                                "enum": sorted(_CLAIM_TYPES),
                            },
                            "relevance_to_question": {
                                "type": "string",
                                "enum": sorted(_RELEVANCE_VALUES),
                            },
                            "support_status": {
                                "type": "string",
                                "enum": sorted(_SUPPORT_STATUSES),
                            },
                            "fine_label": {
                                "type": "string",
                                "enum": [*_LABEL_ORDER, "None", "Unclear"],
                            },
                            "reason": {"type": "string"},
                        },
                        "required": [
                            "claim_id",
                            "claim",
                            "source",
                            "claim_type",
                            "relevance_to_question",
                            "support_status",
                            "fine_label",
                            "reason",
                        ],
                        "additionalProperties": False,
                    },
                },
                "explanation": {"type": "string"},
            },
            "required": [
                "answer_correct",
                "is_hallucination",
                "label_order",
                "hallucination_vector",
                "hallucination_labels",
                "primary_label",
                "coarse_labels",
                "unsupported_visual_claim",
                "confidence",
                "claim_checks",
                "explanation",
            ],
            "additionalProperties": False,
        },
    },
}


class JudgeClient(Protocol):
    def chat_completion(
        self,
        *,
        prompt: str,
        image_path: str | Path,
        model: str | None = None,
        temperature: float = 0,
        max_tokens: int = 512,
        allowed_image_root: str | Path | None = None,
        max_image_bytes: int = _DEFAULT_MAX_IMAGE_BYTES,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...


def render_zero_shot_judge_prompt(
    sample: EvalSample,
    response: ModelResponse,
    taxonomy_definition: str,
    template_path: str | Path = "prompts/judge/zero_shot_judge.txt",
) -> str:
    """把样本、模型回答和 taxonomy 定义填入 judge prompt 模板。"""

    template = Path(template_path).read_text(encoding="utf-8")
    return render_prompt_template(
        template,
        {
            "dataset": sample.dataset,
            "task_type": sample.task_type,
            "question": sample.question,
            "choices": json.dumps(sample.choices, ensure_ascii=False),
            "reference_answer": sample.reference_answer,
            "metadata": json.dumps(sample.metadata, ensure_ascii=False),
            "model_response": response.raw_response,
            "visual_evidence": response.parsed.visual_evidence,
            "reasoning": response.parsed.reasoning,
            "final_answer": response.parsed.final_answer,
            "taxonomy_definition": taxonomy_definition,
        },
    )


def parse_judge_output(raw_text: str) -> dict[str, Any]:
    """解析 judge JSON，并统一成 detector 后续校验需要的字段。"""

    payload = _load_judge_json(raw_text)
    if payload is None:
        return _fallback_details("Could not parse judge response as JSON.")
    return normalize_judge_payload(payload)


def normalize_judge_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """把 judge 返回的多种标签表达统一成主标签、向量和 claim 检查列表。"""

    raw_claim_checks = payload.get("claim_checks")
    has_claim_checks = isinstance(raw_claim_checks, list)
    claim_checks = _normalize_claim_checks(raw_claim_checks)
    labels = _labels_from_payload(payload, claim_checks, has_claim_checks)
    vector = [1 if label in labels else 0 for label in _LABEL_ORDER]
    raw_is_hallucination = _coerce_optional_bool(payload.get("is_hallucination"))
    primary_label = _select_primary_label(labels)
    coarse_labels = _coarse_labels(labels)
    unsupported_visual_claim = _coerce_optional_bool(
        payload.get("unsupported_visual_claim")
    )
    if labels & FACTUAL_TYPES:
        unsupported_visual_claim = True
    if not labels and unsupported_visual_claim is None:
        unsupported_visual_claim = False

    return {
        "answer_correct": _coerce_optional_bool(payload.get("answer_correct")),
        "is_hallucination": bool(labels),
        "raw_is_hallucination": raw_is_hallucination,
        "label_order": list(_LABEL_ORDER),
        "hallucination_vector": vector,
        "hallucination_labels": [label for label in _LABEL_ORDER if label in labels],
        "primary_label": primary_label,
        "coarse_labels": coarse_labels,
        "unsupported_visual_claim": unsupported_visual_claim,
        "confidence": _normalize_confidence(payload.get("confidence")),
        "claim_checks": claim_checks,
        "explanation": str(payload.get("explanation", "")).strip(),
    }


def details_to_detector_result(
    *,
    sample: EvalSample,
    response: ModelResponse,
    details: dict[str, Any],
    raw_judge_response: str,
    run_id: str,
    judge_provider: str | None = None,
) -> DetectorResult:
    details = dict(details)
    if judge_provider is not None:
        details["judge_provider"] = judge_provider
    primary_label = cast(
        FineTaxonomy,
        normalize_fine_label(str(details.get("primary_label", "Unclear"))),
    )
    taxonomy = TaxonomyLabel(
        coarse=cast(CoarseTaxonomy, infer_coarse_from_fine(primary_label)),
        fine=primary_label,
    )
    return DetectorResult(
        run_id=run_id,
        sample_id=sample.sample_id,
        model_response_id=_model_response_id(response),
        detector="response_claim_zero_shot_judge",
        answer_correct=details.get("answer_correct"),
        is_hallucination=details.get("is_hallucination"),
        taxonomy=taxonomy,
        unsupported_visual_claim=details.get("unsupported_visual_claim"),
        confidence=details.get("confidence"),
        explanation=str(details.get("explanation", "")),
        raw_judge_response=raw_judge_response,
        dataset=sample.dataset,
        model=response.model,
        prompt_type=response.prompt_type,
        details=details,
    )


def _unclear_detector_result(
    sample: EvalSample,
    response: ModelResponse,
    *,
    run_id: str,
    explanation: str,
    fallback_source: str,
    raw_judge_response: str = "",
) -> DetectorResult:
    details = {
        "answer_correct": None,
        "is_hallucination": None,
        "label_order": _LABEL_ORDER,
        "hallucination_vector": [0] * len(_LABEL_ORDER),
        "hallucination_labels": [],
        "primary_label": "Unclear",
        "coarse_labels": ["Unclear"],
        "unsupported_visual_claim": None,
        "confidence": "low",
        "claim_checks": [],
        "fallback_source": fallback_source,
        "explanation": explanation,
    }
    return details_to_detector_result(
        sample=sample,
        response=response,
        details=details,
        raw_judge_response=raw_judge_response,
        run_id=run_id,
    )


def build_failed_model_response_detector_result(
    sample: EvalSample,
    response: ModelResponse,
    *,
    run_id: str,
) -> DetectorResult:
    error = str(response.inference_metadata.get("error", "Model inference failed."))
    return _unclear_detector_result(
        sample,
        response,
        run_id=run_id,
        fallback_source="model_inference_failed",
        explanation=f"Model response is unavailable because inference failed: {error}",
    )


def build_failed_judge_detector_result(
    sample: EvalSample,
    response: ModelResponse,
    *,
    run_id: str,
    error: Exception,
) -> DetectorResult:
    return _unclear_detector_result(
        sample,
        response,
        run_id=run_id,
        fallback_source="judge_inference_failed",
        explanation=f"Judge inference failed: {error}",
        raw_judge_response="",
    )


def _build_client(
    provider_config: Any,
    *,
    request_timeout_seconds: int,
    request_max_retries: int,
) -> JudgeClient:
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


def _invalid_model_response_reason(response: ModelResponse) -> str | None:
    return model_response_invalid_reason(response.to_dict())


def _has_reference_answer(sample: EvalSample) -> bool:
    return sample.reference_answer.strip() not in {"", "UNAVAILABLE"}


def _judge_response_with_retries(
    sample: EvalSample,
    response: ModelResponse,
    *,
    client: JudgeClient,
    run_id: str,
    provider: str | None = None,
    model: str | None = _DEFAULT_JUDGE_MODEL,
    taxonomy_path: str | Path = "prompts/judge/taxonomy_definition.txt",
    template_path: str | Path = "prompts/judge/zero_shot_judge.txt",
    path_base: str | Path = ".",
    image_root: str | Path = "data/raw",
    temperature: float = 0,
    max_tokens: int = 1024,
    max_image_bytes: int = _DEFAULT_MAX_IMAGE_BYTES,
    max_attempts: int = 3,
) -> DetectorResult:
    """对同一条模型回答重复调用 judge，直到拿到可校验的结构化结果。"""

    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return judge_response(
                sample,
                response,
                client=client,
                run_id=run_id,
                provider=provider,
                model=model,
                taxonomy_path=taxonomy_path,
                template_path=template_path,
                path_base=path_base,
                image_root=image_root,
                temperature=temperature,
                max_tokens=max_tokens,
                max_image_bytes=max_image_bytes,
            )
        except Exception as exc:
            last_error = exc
            if attempt < max_attempts - 1:
                time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(
        f"Judge failed after {max_attempts} attempts for {_model_response_id(response)}: {last_error}"
    ) from last_error


def judge_response(
    sample: EvalSample,
    response: ModelResponse,
    *,
    client: JudgeClient,
    run_id: str,
    provider: str | None = None,
    model: str | None = _DEFAULT_JUDGE_MODEL,
    taxonomy_path: str | Path = "prompts/judge/taxonomy_definition.txt",
    template_path: str | Path = "prompts/judge/zero_shot_judge.txt",
    path_base: str | Path = ".",
    image_root: str | Path = "data/raw",
    temperature: float = 0,
    max_tokens: int = 1024,
    max_image_bytes: int = _DEFAULT_MAX_IMAGE_BYTES,
) -> DetectorResult:
    """调用 judge 模型并把 JSON 输出转换成 DetectorResult。"""

    taxonomy_definition = Path(taxonomy_path).read_text(encoding="utf-8")
    prompt = render_zero_shot_judge_prompt(
        sample, response, taxonomy_definition, template_path
    )
    image_path = resolve_sample_image_path(sample.image_path, path_base=path_base)
    allowed_image_root = (Path(path_base) / image_root).resolve()
    api_response = _call_judge_model(
        client=client,
        prompt=prompt,
        image_path=image_path,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        allowed_image_root=allowed_image_root,
        max_image_bytes=max_image_bytes,
    )
    raw_judge_response = extract_message_text(api_response)
    details = parse_judge_output(raw_judge_response)
    if details.get("fallback_source") == "judge_output_parse_failed":
        raise ValueError("Could not parse judge response as JSON.")
    if details.get("raw_is_hallucination") is None:
        raise ValueError("Judge response has null required detector fields.")
    if details.get("answer_correct") is None and _has_reference_answer(sample):
        raise ValueError(
            "Judge response has null answer correctness despite available reference answer."
        )
    return details_to_detector_result(
        sample=sample,
        response=response,
        details=details,
        raw_judge_response=raw_judge_response,
        run_id=run_id,
        judge_provider=provider,
    )


def _call_judge_model(
    *,
    client: JudgeClient,
    prompt: str,
    image_path: Path,
    model: str | None,
    temperature: float,
    max_tokens: int,
    allowed_image_root: Path,
    max_image_bytes: int,
) -> dict[str, Any]:
    kwargs = {
        "prompt": prompt,
        "image_path": image_path,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "allowed_image_root": allowed_image_root,
        "max_image_bytes": max_image_bytes,
    }
    try:
        return client.chat_completion(
            **kwargs,
            response_format=_JUDGE_RESPONSE_FORMAT,
        )
    except TypeError as exc:
        if _is_response_format_unsupported_error(exc):
            return client.chat_completion(**kwargs)
        raise
    except RuntimeError as exc:
        if _is_response_format_unsupported_error(exc):
            return client.chat_completion(**kwargs)
        raise


def _is_response_format_unsupported_error(error: Exception) -> bool:
    message = str(error).lower()
    mentions_structured_output = any(
        marker in message
        for marker in (
            "response_format",
            "json_schema",
            "structured output",
            "structured outputs",
        )
    )
    mentions_unsupported_parameter = any(
        marker in message
        for marker in (
            "unsupported",
            "not supported",
            "unknown parameter",
            "unrecognized request argument",
            "unrecognized parameter",
            "unexpected keyword argument",
        )
    )
    return mentions_structured_output and mentions_unsupported_parameter


def detect_file(
    samples_path: str | Path,
    responses_path: str | Path,
    output_path: str | Path,
    *,
    provider: str,
    run_id: str,
    model: str | None = _DEFAULT_JUDGE_MODEL,
    taxonomy_path: str | Path = "prompts/judge/taxonomy_definition.txt",
    template_path: str | Path = "prompts/judge/zero_shot_judge.txt",
    path_base: str | Path = ".",
    image_root: str | Path = "data/raw",
    limit: int | None = None,
    offset: int = 0,
    temperature: float = 0,
    max_tokens: int = 1024,
    max_image_bytes: int = _DEFAULT_MAX_IMAGE_BYTES,
    allow_missing: bool = False,
    overwrite: bool = False,
    resume: bool = False,
    concurrency: int = 1,
    judge_max_attempts: int = 3,
    request_timeout_seconds: int = 120,
    request_max_retries: int = 2,
) -> None:
    """对模型回答文件运行 detector，resume 时只补齐缺失或无效的判定行。"""

    if overwrite and resume:
        raise ValueError("Use either overwrite or resume, not both.")
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    if judge_max_attempts < 1:
        raise ValueError("judge_max_attempts must be at least 1")
    if request_timeout_seconds < 1:
        raise ValueError("request_timeout_seconds must be at least 1")
    if request_max_retries < 0:
        raise ValueError("request_max_retries must be at least 0")

    samples = {
        row["sample_id"]: EvalSample.from_dict(row)
        for row in read_json_records(samples_path)
    }
    output = Path(output_path)
    response_rows = _response_rows_by_id(responses_path)
    rows, existing_ids = _prepare_output_rows(
        output,
        samples=samples,
        overwrite=overwrite,
        resume=resume,
        run_id=run_id,
        provider=provider,
        response_rows=response_rows,
    )
    response_indices = _response_indices(responses_path)
    indexed_rows = [
        (
            response_indices.get(
                str(row["model_response_id"]), len(response_indices) + index
            ),
            row,
        )
        for index, row in enumerate(rows)
    ]
    target_response_ids = _target_response_ids(
        responses_path=responses_path, offset=offset, limit=limit
    )
    if target_response_ids and target_response_ids <= existing_ids:
        _validate_completed_detector_output(
            output,
            samples,
            target_response_ids,
            run_id=run_id,
            provider=provider,
            response_rows=response_rows,
        )
        return
    pending, missing_sample_ids = _collect_pending_judgements(
        responses_path=responses_path,
        samples=samples,
        existing_ids=existing_ids,
        offset=offset,
        limit=limit,
    )

    if concurrency == 1:
        client: JudgeClient | None = None
        for response_index, sample, response in pending:
            invalid_reason = _invalid_model_response_reason(response)
            if invalid_reason is not None:
                raise RuntimeError(
                    f"Cannot run detector for invalid model response {_model_response_id(response)}: "
                    f"{invalid_reason}. Rerun model inference with --resume until the response succeeds."
                )
            if client is None:
                client = _build_client(
                    get_provider_config(provider),
                    request_timeout_seconds=request_timeout_seconds,
                    request_max_retries=request_max_retries,
                )
            result = _judge_response_with_retries(
                sample,
                response,
                client=client,
                run_id=run_id,
                provider=provider,
                model=model,
                taxonomy_path=taxonomy_path,
                template_path=template_path,
                path_base=path_base,
                image_root=image_root,
                temperature=temperature,
                max_tokens=max_tokens,
                max_image_bytes=max_image_bytes,
                max_attempts=judge_max_attempts,
            )
            indexed_rows.append((response_index, result.to_dict()))
            existing_ids.add(_model_response_id(response))
            _write_indexed_rows(output, indexed_rows)
    else:
        _detect_pending_concurrently(
            pending=pending,
            concurrency=concurrency,
            provider=provider,
            run_id=run_id,
            model=model,
            taxonomy_path=taxonomy_path,
            template_path=template_path,
            path_base=path_base,
            image_root=image_root,
            temperature=temperature,
            max_tokens=max_tokens,
            max_image_bytes=max_image_bytes,
            judge_max_attempts=judge_max_attempts,
            request_timeout_seconds=request_timeout_seconds,
            request_max_retries=request_max_retries,
            existing_ids=existing_ids,
            indexed_rows=indexed_rows,
            output=output,
        )

    if missing_sample_ids and not allow_missing:
        preview = ", ".join(missing_sample_ids[:5])
        raise ValueError(
            f"{len(missing_sample_ids)} responses did not match samples: {preview}"
        )
    _validate_completed_detector_output(
        output,
        samples,
        target_response_ids,
        run_id=run_id,
        provider=provider,
        response_rows=response_rows,
    )


def _response_rows_by_id(responses_path: str | Path) -> dict[str, dict[str, Any]]:
    return {
        _model_response_id(ModelResponse.from_dict(row)): row
        for row in read_json_records(responses_path)
    }


def _response_indices(responses_path: str | Path) -> dict[str, int]:
    return {
        _model_response_id(ModelResponse.from_dict(row)): index
        for index, row in enumerate(read_json_records(responses_path))
    }


def _target_response_ids(
    *, responses_path: str | Path, offset: int, limit: int | None
) -> set[str]:
    return {
        _model_response_id(ModelResponse.from_dict(row))
        for index, row in enumerate(read_json_records(responses_path))
        if index >= offset and (limit is None or index < offset + limit)
    }


def _collect_pending_judgements(
    *,
    responses_path: str | Path,
    samples: dict[str, EvalSample],
    existing_ids: set[str],
    offset: int,
    limit: int | None,
) -> tuple[list[tuple[int, EvalSample, ModelResponse]], list[str]]:
    pending = []
    missing_sample_ids = []
    for index, row in enumerate(read_json_records(responses_path)):
        if index < offset:
            continue
        if limit is not None and index >= offset + limit:
            break
        response = ModelResponse.from_dict(row)
        response_id = _model_response_id(response)
        if response_id in existing_ids:
            continue
        sample = samples.get(response.sample_id)
        if sample is None:
            missing_sample_ids.append(response.sample_id)
            continue
        pending.append((index, sample, response))
    return pending, missing_sample_ids


def _detect_pending_concurrently(
    *,
    pending: list[tuple[int, EvalSample, ModelResponse]],
    concurrency: int,
    provider: str,
    run_id: str,
    model: str | None,
    taxonomy_path: str | Path,
    template_path: str | Path,
    path_base: str | Path,
    image_root: str | Path,
    temperature: float,
    max_tokens: int,
    max_image_bytes: int,
    judge_max_attempts: int,
    request_timeout_seconds: int,
    request_max_retries: int,
    existing_ids: set[str],
    indexed_rows: list[tuple[int, dict[str, Any]]],
    output: Path,
) -> None:
    """在固定并发窗口内执行 judge；失败会抛给外层 group resume 继续重试。"""

    failures: list[tuple[str, Exception]] = []
    next_item = 0
    futures: dict[Future[DetectorResult], tuple[int, EvalSample, ModelResponse]] = {}
    provider_config = get_provider_config(provider)

    def submit_next(executor: ThreadPoolExecutor) -> None:
        nonlocal next_item
        if next_item >= len(pending):
            return
        response_index, sample, response = pending[next_item]
        next_item += 1
        invalid_reason = _invalid_model_response_reason(response)
        if invalid_reason is not None:
            raise RuntimeError(
                f"Cannot run detector for invalid model response {_model_response_id(response)}: "
                f"{invalid_reason}. Rerun model inference with --resume until the response succeeds."
            )
        client = _build_client(
            provider_config,
            request_timeout_seconds=request_timeout_seconds,
            request_max_retries=request_max_retries,
        )
        future = executor.submit(
            _judge_response_with_retries,
            sample,
            response,
            client=client,
            run_id=run_id,
            provider=provider,
            model=model,
            taxonomy_path=taxonomy_path,
            template_path=template_path,
            path_base=path_base,
            image_root=image_root,
            temperature=temperature,
            max_tokens=max_tokens,
            max_image_bytes=max_image_bytes,
            max_attempts=judge_max_attempts,
        )
        futures[future] = (response_index, sample, response)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        for _ in range(min(concurrency, len(pending))):
            submit_next(executor)
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                response_index, _, response = futures.pop(future)
                response_id = _model_response_id(response)
                try:
                    result = future.result()
                except Exception as exc:
                    failures.append((response_id, exc))
                else:
                    if response_id not in existing_ids:
                        indexed_rows.append((response_index, result.to_dict()))
                        existing_ids.add(response_id)
                        _write_indexed_rows(output, indexed_rows)
                submit_next(executor)

    if failures:
        failed_items = ", ".join(
            f"{response_id}: {error}" for response_id, error in failures[:10]
        )
        suffix = "" if len(failures) <= 10 else f" ... and {len(failures) - 10} more"
        raise RuntimeError(
            f"Failed judge inference for responses: {failed_items}{suffix}"
        )


def _write_indexed_rows(
    output: Path, indexed_rows: list[tuple[int, dict[str, Any]]]
) -> None:
    rows_by_response_id = {
        str(row["model_response_id"]): (index, row) for index, row in indexed_rows
    }
    rows = [
        row for _, row in sorted(rows_by_response_id.values(), key=lambda item: item[0])
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
    for attempt in range(5):
        try:
            temp_path.replace(output)
            return
        except PermissionError:
            if attempt == 4:
                temp_path.unlink(missing_ok=True)
                raise
            time.sleep(0.1 * (attempt + 1))


def _prepare_output_rows(
    output_path: Path,
    *,
    samples: dict[str, EvalSample],
    overwrite: bool,
    resume: bool,
    run_id: str | None = None,
    provider: str | None = None,
    response_rows: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], set[str]]:
    """读取旧 detector 输出，只保留当前配置下仍然有效且兼容的结果。"""

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
        if not _should_retry_detector_row(row, samples)
        and _is_compatible_existing_detector_row(
            row,
            run_id=run_id,
            provider=provider,
            response_rows=response_rows,
        )
    ]
    _write_indexed_rows(
        output_path,
        [(index, row) for index, row in enumerate(rows)],
    )
    return rows, {str(row["model_response_id"]) for row in rows}


def _is_compatible_existing_detector_row(
    row: dict[str, Any],
    *,
    run_id: str | None,
    provider: str | None,
    response_rows: dict[str, dict[str, Any]] | None,
) -> bool:
    if run_id is None or provider is None or response_rows is None:
        return True
    response_id = str(row.get("model_response_id", ""))
    response = response_rows.get(response_id)
    if response is None:
        return False
    if model_response_invalid_reason(response) is not None:
        return False
    if str(row.get("sample_id", "")) != str(response.get("sample_id", "")):
        return False
    if response_id != f"{response.get('run_id')}:{response.get('sample_id')}":
        return False
    return is_detector_row_compatible(
        row,
        run_id=run_id,
        detector="zero_shot",
        dataset=str(response.get("dataset", "")),
        model=str(response.get("model", "")),
        prompt_type=str(response.get("prompt_type", "")),
        judge_provider=provider,
    )


def _should_retry_detector_row(
    row: dict[str, Any], samples: dict[str, EvalSample]
) -> bool:
    details = row.get("details")
    if not isinstance(details, dict):
        details = {}
    fallback_source = str(details.get("fallback_source", ""))
    if fallback_source in {
        "judge_inference_failed",
        "judge_output_parse_failed",
        "model_inference_failed",
    }:
        return True
    if row.get("is_hallucination") is None:
        return True
    sample = samples.get(str(row.get("sample_id", "")))
    if sample is None:
        return True
    if row.get("answer_correct") is None and _has_reference_answer(sample):
        return True
    explanation = str(row.get("explanation", details.get("explanation", "")))
    return (
        explanation.startswith("Judge inference failed:")
        or explanation.startswith(
            "Model response is unavailable because inference failed:"
        )
        or explanation == "Could not parse judge response as JSON."
    )


def _validate_completed_detector_output(
    output: Path,
    samples: dict[str, EvalSample],
    target_response_ids: set[str],
    *,
    run_id: str | None = None,
    provider: str | None = None,
    response_rows: dict[str, dict[str, Any]] | None = None,
) -> None:
    rows = list(read_json_records(output)) if output.exists() else []
    rows_by_response_id = {str(row.get("model_response_id", "")): row for row in rows}
    missing = sorted(target_response_ids - set(rows_by_response_id))
    invalid = sorted(
        response_id
        for response_id in target_response_ids & set(rows_by_response_id)
        if _should_retry_detector_row(rows_by_response_id[response_id], samples)
        or not _is_compatible_existing_detector_row(
            rows_by_response_id[response_id],
            run_id=run_id,
            provider=provider,
            response_rows=response_rows,
        )
    )
    if missing or invalid:
        parts = []
        if missing:
            parts.append(f"missing={missing[:10]}")
        if invalid:
            parts.append(f"invalid={invalid[:10]}")
        raise RuntimeError(
            "Detector output is incomplete or invalid: " + "; ".join(parts)
        )


def _model_response_id(response: ModelResponse) -> str:
    return f"{response.run_id}:{response.sample_id}"


def _load_judge_json(raw_text: str) -> dict[str, Any] | None:
    candidates = [raw_text.strip(), _strip_json_fence(raw_text.strip())]
    extracted = _extract_first_json_object(raw_text)
    if extracted:
        candidates.append(extracted)
    for candidate in candidates:
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _strip_json_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_first_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return ""
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return ""


def _normalize_claim_checks(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    checks = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            continue
        claim_type = _normalize_member(
            item.get("claim_type"), _CLAIM_TYPES, "reasoning_claim"
        )
        support_status = _normalize_member(
            item.get("support_status"), _SUPPORT_STATUSES, "unverifiable"
        )
        fine_label = _derive_claim_label(
            claim_type,
            support_status,
            normalize_fine_label(str(item.get("fine_label", "None"))),
        )
        checks.append(
            {
                "claim_id": str(item.get("claim_id") or f"c{index}"),
                "claim": str(item.get("claim", "")),
                "source": str(item.get("source", "")),
                "claim_type": claim_type,
                "relevance_to_question": _normalize_member(
                    item.get("relevance_to_question"),
                    _RELEVANCE_VALUES,
                    "supporting_reasoning",
                ),
                "support_status": support_status,
                "fine_label": fine_label,
                "reason": str(item.get("reason", "")),
            }
        )
    return checks


def _labels_from_payload(
    payload: dict[str, Any],
    claim_checks: list[dict[str, Any]],
    has_claim_checks: bool,
) -> set[str]:
    labels = set()
    if has_claim_checks:
        for check in claim_checks:
            label = normalize_fine_label(str(check.get("fine_label", "None")))
            if label in _LABEL_ORDER and check.get("support_status") != "supported":
                labels.add(label)
        return labels

    vector = payload.get("hallucination_vector")
    if isinstance(vector, list):
        for label, value in zip(_LABEL_ORDER, vector):
            if value == 1 or value is True:
                labels.add(label)
    raw_labels = payload.get("hallucination_labels")
    if isinstance(raw_labels, list):
        for raw_label in raw_labels:
            label = normalize_fine_label(str(raw_label))
            if label in _LABEL_ORDER:
                labels.add(label)
    return labels


def _derive_claim_label(claim_type: str, support_status: str, fine_label: str) -> str:
    if support_status == "supported":
        return "None"
    if claim_type in {"causal_claim", "semantic_claim"}:
        return "SO"
    if claim_type == "inconsistency_claim":
        return "INC"
    if claim_type == "answer_claim" and fine_label not in {"CI", "INC"}:
        return "None"
    if fine_label in _LABEL_ORDER:
        return fine_label
    return _CLAIM_TYPE_LABELS.get(claim_type, "Unclear")


def _select_primary_label(labels: set[str]) -> str:
    if not labels:
        return "None"
    for label in ("OBJ", "ATT", "SPA", "IR", "CI", "INC", "SO"):
        if label in labels:
            return label
    return "Unclear"


def _coarse_labels(labels: set[str]) -> list[str]:
    if not labels:
        return ["None"]
    coarse = []
    if labels & FACTUAL_TYPES:
        coarse.append("Factual Hallucination")
    if labels - FACTUAL_TYPES:
        coarse.append("Logical Hallucination")
    return coarse


def _fallback_details(explanation: str) -> dict[str, Any]:
    return {
        "answer_correct": None,
        "is_hallucination": None,
        "label_order": list(_LABEL_ORDER),
        "hallucination_vector": [0, 0, 0, 0, 0, 0, 0],
        "hallucination_labels": [],
        "primary_label": "Unclear",
        "coarse_labels": ["Unclear"],
        "unsupported_visual_claim": None,
        "confidence": "low",
        "claim_checks": [],
        "fallback_source": "judge_output_parse_failed",
        "explanation": explanation,
    }


def _coerce_optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered in {"", "null", "none", "unclear"}:
            return None
    return None


def _normalize_confidence(value: Any) -> str:
    if isinstance(value, str) and value.strip().lower() in _CONFIDENCE_VALUES:
        return value.strip().lower()
    return "low"


def _normalize_member(value: Any, allowed: set[str], fallback: str) -> str:
    if isinstance(value, str) and value.strip() in allowed:
        return value.strip()
    return fallback


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run response-driven claim-level zero-shot hallucination judging."
    )
    parser.add_argument("--samples", required=True, help="Unified eval JSONL path.")
    parser.add_argument(
        "--responses", required=True, help="Model responses JSONL path."
    )
    parser.add_argument("--output", required=True, help="Detector results JSONL path.")
    parser.add_argument("--provider", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", default=_DEFAULT_JUDGE_MODEL)
    parser.add_argument("--taxonomy", default="prompts/judge/taxonomy_definition.txt")
    parser.add_argument("--template", default="prompts/judge/zero_shot_judge.txt")
    parser.add_argument("--path-base", default=".")
    parser.add_argument("--image-root", default="data/raw")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-image-mb", type=int, default=20)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent judge requests. File writes remain serialized.",
    )
    parser.add_argument(
        "--judge-max-attempts",
        type=int,
        default=3,
        help="Maximum attempts per response for judge API or malformed-output failures.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=int,
        default=120,
        help="Per-request HTTP timeout for judge API calls.",
    )
    parser.add_argument(
        "--request-max-retries",
        type=int,
        default=2,
        help="Maximum HTTP retries per judge request.",
    )
    args = parser.parse_args()

    detect_file(
        samples_path=args.samples,
        responses_path=args.responses,
        output_path=args.output,
        provider=args.provider,
        run_id=args.run_id,
        model=args.model,
        taxonomy_path=args.taxonomy,
        template_path=args.template,
        path_base=args.path_base,
        image_root=args.image_root,
        limit=args.limit,
        offset=args.offset,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_image_bytes=args.max_image_mb * 1024 * 1024,
        allow_missing=args.allow_missing,
        overwrite=args.overwrite,
        resume=args.resume,
        concurrency=args.concurrency,
        judge_max_attempts=args.judge_max_attempts,
        request_timeout_seconds=args.request_timeout_seconds,
        request_max_retries=args.request_max_retries,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
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
    FINE_TYPES,
    infer_coarse_from_fine,
    normalize_fine_label,
)
from src.models.openai_compatible import (
    OpenAICompatibleClient,
    extract_message_text,
    get_provider_config,
)
from src.models.run_inference import resolve_sample_image_path

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
    "causal_claim": "CI",
    "inconsistency_claim": "INC",
    "semantic_claim": "SO",
    "answer_claim": "None",
}
_CONFIDENCE_VALUES = {"high", "medium", "low"}
_DEFAULT_JUDGE_MODEL = "gpt-5.4-mini"
_DEFAULT_MAX_IMAGE_BYTES = 20 * 1024 * 1024


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
    ) -> dict[str, Any]: ...


def render_zero_shot_judge_prompt(
    sample: EvalSample,
    response: ModelResponse,
    taxonomy_definition: str,
    template_path: str | Path = "prompts/judge/zero_shot_judge.txt",
) -> str:
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
    payload = _load_judge_json(raw_text)
    if payload is None:
        return _fallback_details("Could not parse judge response as JSON.")
    return normalize_judge_payload(payload)


def normalize_judge_payload(payload: dict[str, Any]) -> dict[str, Any]:
    raw_claim_checks = payload.get("claim_checks")
    has_claim_checks = isinstance(raw_claim_checks, list)
    claim_checks = _normalize_claim_checks(raw_claim_checks)
    labels = _labels_from_payload(payload, claim_checks, has_claim_checks)
    vector = [1 if label in labels else 0 for label in _LABEL_ORDER]
    is_hallucination = bool(labels)
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
        "is_hallucination": is_hallucination,
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
) -> DetectorResult:
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


def judge_response(
    sample: EvalSample,
    response: ModelResponse,
    *,
    client: JudgeClient,
    run_id: str,
    model: str | None = _DEFAULT_JUDGE_MODEL,
    taxonomy_path: str | Path = "prompts/judge/taxonomy_definition.txt",
    template_path: str | Path = "prompts/judge/zero_shot_judge.txt",
    path_base: str | Path = ".",
    image_root: str | Path = "data/raw",
    temperature: float = 0,
    max_tokens: int = 1024,
    max_image_bytes: int = _DEFAULT_MAX_IMAGE_BYTES,
) -> DetectorResult:
    taxonomy_definition = Path(taxonomy_path).read_text(encoding="utf-8")
    prompt = render_zero_shot_judge_prompt(
        sample, response, taxonomy_definition, template_path
    )
    image_path = resolve_sample_image_path(sample.image_path, path_base=path_base)
    allowed_image_root = (Path(path_base) / image_root).resolve()
    api_response = client.chat_completion(
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
    return details_to_detector_result(
        sample=sample,
        response=response,
        details=details,
        raw_judge_response=raw_judge_response,
        run_id=run_id,
    )


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
) -> None:
    if overwrite and resume:
        raise ValueError("Use either overwrite or resume, not both.")

    output = Path(output_path)
    rows, existing_ids = _prepare_output_rows(
        output, overwrite=overwrite, resume=resume
    )
    samples = {
        row["sample_id"]: EvalSample.from_dict(row)
        for row in read_json_records(samples_path)
    }
    client: JudgeClient | None = None
    missing_sample_ids = []
    processed_count = 0

    for index, row in enumerate(read_json_records(responses_path)):
        if index < offset:
            continue
        if limit is not None and processed_count >= limit:
            break
        response = ModelResponse.from_dict(row)
        response_id = _model_response_id(response)
        if response_id in existing_ids:
            continue
        sample = samples.get(response.sample_id)
        if sample is None:
            missing_sample_ids.append(response.sample_id)
            continue

        if client is None:
            client = OpenAICompatibleClient(get_provider_config(provider))
        result = judge_response(
            sample,
            response,
            client=client,
            run_id=run_id,
            model=model,
            taxonomy_path=taxonomy_path,
            template_path=template_path,
            path_base=path_base,
            image_root=image_root,
            temperature=temperature,
            max_tokens=max_tokens,
            max_image_bytes=max_image_bytes,
        )
        rows.append(result.to_dict())
        existing_ids.add(response_id)
        processed_count += 1
        write_jsonl(output, rows)

    if missing_sample_ids and not allow_missing:
        preview = ", ".join(missing_sample_ids[:5])
        raise ValueError(
            f"{len(missing_sample_ids)} responses did not match samples: {preview}"
        )


def _prepare_output_rows(
    output_path: Path, *, overwrite: bool, resume: bool
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
    rows = list(read_json_records(output_path))
    return rows, {str(row["model_response_id"]) for row in rows}


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
    )


if __name__ == "__main__":
    main()

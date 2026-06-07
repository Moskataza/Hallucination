from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

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
) -> None:
    if overwrite and resume:
        raise ValueError("Use either overwrite or resume, not both.")

    provider_config = get_provider_config(provider)
    client = OpenAICompatibleClient(provider_config)
    prompt_template = Path(prompt_path).read_text(encoding="utf-8")
    output = Path(output_path)
    rows, existing_sample_ids = _prepare_output_rows(
        output, overwrite=overwrite, resume=resume
    )
    allowed_image_root = (Path(path_base) / image_root).resolve()

    target_sample_ids: set[str] | None = None
    if limit is not None:
        target_sample_ids = {
            str(sample["sample_id"])
            for index, sample in enumerate(read_json_records(dataset_path))
            if index >= offset and index < offset + limit
        }
        if target_sample_ids <= existing_sample_ids:
            return

    processed_count = 0
    for index, sample in enumerate(read_json_records(dataset_path)):
        if index < offset:
            continue
        if limit is not None and index >= offset + limit:
            break
        sample_id = str(sample["sample_id"])
        if sample_id in existing_sample_ids:
            continue

        prompt = render_prompt(prompt_template, sample)
        image_path = resolve_sample_image_path(
            sample["image_path"], path_base=path_base
        )
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
        inference_metadata: dict[str, Any] = {
            "provider": provider,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "api_usage": api_response.get("usage", {}),
        }
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
        rows.append(response.to_dict())
        existing_sample_ids.add(sample_id)
        processed_count += 1
        write_jsonl(output, rows)


def resolve_prompt_type(
    prompt_path: str | Path, prompt_type: str | None = None
) -> PromptType:
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
    path = Path(image_path)
    if path.is_absolute():
        return path.resolve()
    return (Path(path_base) / path).resolve()


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
    return rows, {str(row["sample_id"]) for row in rows}


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
    )


if __name__ == "__main__":
    main()

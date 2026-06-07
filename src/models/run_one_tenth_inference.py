from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.datasets.jsonl import read_json_records
from src.models.run_inference import resolve_prompt_type, run_inference

DatasetName = Literal["pope", "mathvista"]
PromptName = Literal["direct", "cot"]
ModelName = Literal["gemini", "qwen"]


@dataclass(frozen=True)
class InferenceGroup:
    run_id: str
    dataset_path: str
    prompt_path: str
    output_path: str
    provider: str
    limit: int
    max_tokens: int
    dataset: DatasetName
    model: ModelName
    prompt: PromptName
    offset: int = 0


ONE_TENTH_GROUPS: tuple[InferenceGroup, ...] = (
    InferenceGroup(
        run_id="one_tenth_pope_gemini_direct_v1",
        dataset_path="data/processed/pope_eval.jsonl",
        prompt_path="prompts/answer/direct_pope.txt",
        output_path="outputs/model_responses/one_tenth_pope_gemini_direct.jsonl",
        provider="gemini_local",
        limit=900,
        max_tokens=256,
        dataset="pope",
        model="gemini",
        prompt="direct",
    ),
    InferenceGroup(
        run_id="one_tenth_pope_gemini_cot_v1",
        dataset_path="data/processed/pope_eval.jsonl",
        prompt_path="prompts/answer/evidence_grounded_cot_pope.txt",
        output_path="outputs/model_responses/one_tenth_pope_gemini_cot.jsonl",
        provider="gemini_local",
        limit=900,
        max_tokens=512,
        dataset="pope",
        model="gemini",
        prompt="cot",
    ),
    InferenceGroup(
        run_id="one_tenth_pope_qwen_direct_v1",
        dataset_path="data/processed/pope_eval.jsonl",
        prompt_path="prompts/answer/direct_pope.txt",
        output_path="outputs/model_responses/one_tenth_pope_qwen_direct.jsonl",
        provider="openrouter_qwen3_vl_instruct",
        limit=900,
        max_tokens=256,
        dataset="pope",
        model="qwen",
        prompt="direct",
    ),
    InferenceGroup(
        run_id="one_tenth_pope_qwen_cot_v1",
        dataset_path="data/processed/pope_eval.jsonl",
        prompt_path="prompts/answer/evidence_grounded_cot_pope.txt",
        output_path="outputs/model_responses/one_tenth_pope_qwen_cot.jsonl",
        provider="openrouter_qwen3_vl_instruct",
        limit=900,
        max_tokens=512,
        dataset="pope",
        model="qwen",
        prompt="cot",
    ),
    InferenceGroup(
        run_id="one_tenth_mathvista_gemini_direct_v1",
        dataset_path="data/processed/mathvista_eval.jsonl",
        prompt_path="prompts/answer/direct_mathvista.txt",
        output_path="outputs/model_responses/one_tenth_mathvista_gemini_direct.jsonl",
        provider="gemini_local",
        limit=514,
        max_tokens=256,
        dataset="mathvista",
        model="gemini",
        prompt="direct",
    ),
    InferenceGroup(
        run_id="one_tenth_mathvista_gemini_cot_v1",
        dataset_path="data/processed/mathvista_eval.jsonl",
        prompt_path="prompts/answer/evidence_grounded_cot_mathvista.txt",
        output_path="outputs/model_responses/one_tenth_mathvista_gemini_cot.jsonl",
        provider="gemini_local",
        limit=514,
        max_tokens=512,
        dataset="mathvista",
        model="gemini",
        prompt="cot",
    ),
    InferenceGroup(
        run_id="one_tenth_mathvista_qwen_direct_v1",
        dataset_path="data/processed/mathvista_eval.jsonl",
        prompt_path="prompts/answer/direct_mathvista.txt",
        output_path="outputs/model_responses/one_tenth_mathvista_qwen_direct.jsonl",
        provider="openrouter_qwen3_vl_instruct",
        limit=514,
        max_tokens=256,
        dataset="mathvista",
        model="qwen",
        prompt="direct",
    ),
    InferenceGroup(
        run_id="one_tenth_mathvista_qwen_cot_v1",
        dataset_path="data/processed/mathvista_eval.jsonl",
        prompt_path="prompts/answer/evidence_grounded_cot_mathvista.txt",
        output_path="outputs/model_responses/one_tenth_mathvista_qwen_cot.jsonl",
        provider="openrouter_qwen3_vl_instruct",
        limit=514,
        max_tokens=512,
        dataset="mathvista",
        model="qwen",
        prompt="cot",
    ),
)


def count_completed_target_samples(group: InferenceGroup) -> int:
    target_ids = set(_target_sample_ids(group))
    existing_ids = _existing_sample_ids(group)
    return len(target_ids & existing_ids)


def count_completed_target_prefix(group: InferenceGroup) -> int:
    existing_ids = _existing_sample_ids(group)
    completed = 0
    for sample_id in _target_sample_ids(group):
        if sample_id not in existing_ids:
            break
        completed += 1
    return completed


def _target_sample_ids(group: InferenceGroup) -> list[str]:
    return [
        str(sample["sample_id"])
        for index, sample in enumerate(read_json_records(group.dataset_path))
        if index >= group.offset and index < group.offset + group.limit
    ]


def _existing_sample_ids(group: InferenceGroup) -> set[str]:
    output = Path(group.output_path)
    if not output.exists():
        return set()
    return {str(row["sample_id"]) for row in read_json_records(output)}


def select_groups(
    *,
    datasets: set[str] | None = None,
    models: set[str] | None = None,
    prompts: set[str] | None = None,
) -> list[InferenceGroup]:
    groups = []
    for group in ONE_TENTH_GROUPS:
        if datasets is not None and group.dataset not in datasets:
            continue
        if models is not None and group.model not in models:
            continue
        if prompts is not None and group.prompt not in prompts:
            continue
        groups.append(group)
    return groups


def resume_groups(groups: list[InferenceGroup], *, chunk_size: int) -> None:
    for group in groups:
        completed = count_completed_target_prefix(group)
        if completed >= group.limit:
            total_completed = count_completed_target_samples(group)
            print(f"SKIP {group.run_id} {total_completed}/{group.limit}", flush=True)
            continue

        while completed < group.limit:
            target_limit = min(group.limit, completed + chunk_size)
            print(
                f"RUN {group.run_id} target={target_limit} completed={completed}/{group.limit}",
                flush=True,
            )
            run_inference(
                dataset_path=group.dataset_path,
                prompt_path=group.prompt_path,
                output_path=group.output_path,
                provider=group.provider,
                run_id=group.run_id,
                prompt_type=resolve_prompt_type(group.prompt_path),
                limit=target_limit,
                offset=group.offset,
                max_tokens=group.max_tokens,
                resume=True,
            )
            next_completed = count_completed_target_prefix(group)
            total_completed = count_completed_target_samples(group)
            print(f"OK {group.run_id} {total_completed}/{group.limit}", flush=True)
            if next_completed <= completed:
                raise RuntimeError(
                    f"Resume made no prefix progress for {group.run_id}: "
                    f"{next_completed}/{group.limit}"
                )
            completed = next_completed


def _parse_filter(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    return set(values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume the one-tenth model inference experiment from existing outputs."
    )
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--dataset", action="append", choices=["pope", "mathvista"])
    parser.add_argument("--model", action="append", choices=["gemini", "qwen"])
    parser.add_argument("--prompt", action="append", choices=["direct", "cot"])
    args = parser.parse_args()

    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be at least 1")

    groups = select_groups(
        datasets=_parse_filter(args.dataset),
        models=_parse_filter(args.model),
        prompts=_parse_filter(args.prompt),
    )
    resume_groups(groups, chunk_size=args.chunk_size)
    print("ONE_TENTH_INFERENCE_DONE", flush=True)


if __name__ == "__main__":
    main()

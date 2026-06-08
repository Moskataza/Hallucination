"""稳定实验流水线 CLI：选择实验组并运行 responses、detectors 或 validate。"""

from __future__ import annotations

import argparse

from src.models.openai_compatible import PROVIDERS
from src.pipelines.experiment_groups import (
    select_detector_groups,
    select_inference_groups,
)
from src.pipelines.result_store import inspect_detector_group, inspect_inference_group
from src.pipelines.resume_groups import resume_detector_groups, resume_inference_groups


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or validate stable experiment pipeline groups.")
    parser.add_argument("--experiment", action="append", dest="experiments")
    parser.add_argument(
        "--stage",
        choices=["responses", "detectors", "all", "validate"],
        default="validate",
    )
    parser.add_argument("--dataset", action="append", choices=["pope", "mathvista"])
    parser.add_argument("--model", action="append", choices=["gemini", "qwen"])
    parser.add_argument("--prompt", action="append", choices=["direct", "cot"])
    parser.add_argument("--detector", action="append", choices=["zero_shot"])
    parser.add_argument("--version", action="append", dest="versions")
    parser.add_argument("--responses-from", default=None)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--request-timeout-seconds", type=int, default=180)
    parser.add_argument("--request-max-retries", type=int, default=2)
    parser.add_argument("--judge-max-attempts", type=int, default=3)
    parser.add_argument("--max-chunk-attempts", type=int, default=3)
    parser.add_argument("--overwrite-detectors", action="store_true")
    parser.add_argument("--detector-provider", choices=sorted(PROVIDERS), default=None)
    args = parser.parse_args()

    experiments = _parse_filter(args.experiments) or {"one_tenth"}
    datasets = _parse_filter(args.dataset)
    models = _parse_filter(args.model)
    prompts = _parse_filter(args.prompt)
    versions = _parse_filter(args.versions)

    inference_groups = select_inference_groups(
        experiments=experiments,
        datasets=datasets,
        models=models,
        prompts=prompts,
        versions=versions,
    )
    detector_groups = select_detector_groups(
        experiments=experiments,
        datasets=datasets,
        models=models,
        prompts=prompts,
        detectors=_parse_filter(args.detector),
        versions=versions,
        responses_from=args.responses_from,
    )
    if args.detector_provider is not None:
        detector_groups = [
            group.__class__(**{**group.__dict__, "provider": args.detector_provider})
            for group in detector_groups
        ]

    if args.stage == "validate":
        _print_validation(inference_groups, detector_groups)
        return
    if args.stage in {"responses", "all"}:
        resume_inference_groups(
            inference_groups,
            chunk_size=args.chunk_size,
            concurrency=args.concurrency,
            request_timeout_seconds=args.request_timeout_seconds,
            request_max_retries=args.request_max_retries,
            max_chunk_attempts=args.max_chunk_attempts,
        )
    if args.stage in {"detectors", "all"}:
        resume_detector_groups(
            detector_groups,
            chunk_size=args.chunk_size,
            concurrency=args.concurrency,
            request_timeout_seconds=args.request_timeout_seconds,
            request_max_retries=args.request_max_retries,
            judge_max_attempts=args.judge_max_attempts,
            max_chunk_attempts=args.max_chunk_attempts,
            overwrite=args.overwrite_detectors,
        )
    _print_final_validation(args.stage, inference_groups, detector_groups)


def _print_final_validation(stage: str, inference_groups, detector_groups) -> None:
    print("FINAL_VALIDATE")
    if stage == "responses":
        _print_validation(inference_groups, [])
    elif stage == "detectors":
        _print_validation([], detector_groups)
    else:
        _print_validation(inference_groups, detector_groups)


def _print_validation(inference_groups, detector_groups) -> None:
    for group in inference_groups:
        status = inspect_inference_group(group).status
        print(
            f"RESPONSES {group.run_id} valid={status.valid} missing={status.missing} "
            f"invalid={status.invalid} duplicates={status.duplicates} complete={status.complete}"
        )
        if status.missing_examples:
            print(f"  missing_examples={list(status.missing_examples)}")
        if status.invalid_examples:
            print(f"  invalid_examples={list(status.invalid_examples)}")
    for group in detector_groups:
        status = inspect_detector_group(group).status
        print(
            f"DETECTORS {group.run_id} valid={status.valid} missing={status.missing} "
            f"invalid={status.invalid} duplicates={status.duplicates} complete={status.complete}"
        )
        if status.missing_examples:
            print(f"  missing_examples={list(status.missing_examples)}")
        if status.invalid_examples:
            print(f"  invalid_examples={list(status.invalid_examples)}")


def _parse_filter(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    return set(values)


if __name__ == "__main__":
    main()

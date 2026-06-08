from __future__ import annotations

import argparse
from pathlib import Path

from src.datasets.jsonl import read_json_records
from src.models.run_inference import (
    _is_invalid_response_row,
    resolve_prompt_type,
    run_inference,
)
from src.pipelines.experiment_groups import (
    InferenceGroup,
    ONE_TENTH_INFERENCE_GROUPS,
    select_inference_groups,
)

ONE_TENTH_GROUPS = ONE_TENTH_INFERENCE_GROUPS


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
    return {
        str(row["sample_id"])
        for row in read_json_records(output)
        if not _is_invalid_response_row(row)
    }


def select_groups(
    *,
    datasets: set[str] | None = None,
    models: set[str] | None = None,
    prompts: set[str] | None = None,
) -> list[InferenceGroup]:
    return select_inference_groups(
        experiments={"one_tenth"}, datasets=datasets, models=models, prompts=prompts
    )


def resume_groups(
    groups: list[InferenceGroup],
    *,
    chunk_size: int,
    concurrency: int = 10,
    request_timeout_seconds: int = 180,
    request_max_retries: int = 2,
    max_chunk_attempts: int = 3,
) -> None:
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    if request_timeout_seconds < 1:
        raise ValueError("request_timeout_seconds must be at least 1")
    if request_max_retries < 0:
        raise ValueError("request_max_retries must be at least 0")
    if max_chunk_attempts < 1:
        raise ValueError("max_chunk_attempts must be at least 1")

    chunk_attempts = {group.run_id: 0 for group in groups}
    completed_groups: set[str] = set()

    while True:
        pending_groups = []
        ran_chunk = False
        stalled_groups = []

        for group in groups:
            prefix_completed = count_completed_target_prefix(group)
            total_completed = count_completed_target_samples(group)
            if total_completed >= group.limit:
                if group.run_id not in completed_groups:
                    print(
                        f"SKIP {group.run_id} prefix={prefix_completed}/{group.limit} "
                        f"total={total_completed}/{group.limit}",
                        flush=True,
                    )
                    completed_groups.add(group.run_id)
                continue

            pending_groups.append(group)
            if chunk_attempts[group.run_id] >= max_chunk_attempts:
                stalled_groups.append(
                    f"{group.run_id} prefix={prefix_completed}/{group.limit} "
                    f"total={total_completed}/{group.limit}"
                )
                continue

            ran_chunk = True
            target_limit = min(group.limit, total_completed + chunk_size)
            print(
                f"RUN {group.run_id} target={target_limit} "
                f"prefix={prefix_completed}/{group.limit} "
                f"total={total_completed}/{group.limit}",
                flush=True,
            )
            run_error: Exception | None = None
            try:
                run_inference(
                    dataset_path=group.dataset_path,
                    prompt_path=group.prompt_path,
                    output_path=group.output_path,
                    provider=group.provider,
                    run_id=group.run_id,
                    prompt_type=resolve_prompt_type(group.prompt_path),
                    prompt_version=group.version,
                    limit=target_limit,
                    offset=group.offset,
                    max_tokens=group.max_tokens,
                    resume=True,
                    concurrency=concurrency,
                    record_failures=True,
                    request_timeout_seconds=request_timeout_seconds,
                    request_max_retries=request_max_retries,
                )
            except Exception as exc:
                run_error = exc

            next_prefix_completed = count_completed_target_prefix(group)
            next_total_completed = count_completed_target_samples(group)
            if next_total_completed > total_completed:
                print(
                    f"OK {group.run_id} prefix={next_prefix_completed}/{group.limit} "
                    f"total={next_total_completed}/{group.limit}",
                    flush=True,
                )
                chunk_attempts[group.run_id] = 0
                continue

            chunk_attempts[group.run_id] += 1
            reason = run_error if run_error is not None else "no completed samples"
            print(
                f"RETRY {group.run_id} attempt={chunk_attempts[group.run_id]}/"
                f"{max_chunk_attempts} prefix={next_prefix_completed}/{group.limit} "
                f"total={next_total_completed}/{group.limit}: {reason}",
                flush=True,
            )

        if not pending_groups:
            return
        if not ran_chunk:
            stalled = "; ".join(stalled_groups)
            raise RuntimeError(f"Resume stalled for all pending groups: {stalled}")


def _parse_filter(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    return set(values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume the one-tenth model inference experiment from existing outputs."
    )
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent model requests per chunk.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=int,
        default=180,
        help="Per-request HTTP timeout for model API calls.",
    )
    parser.add_argument(
        "--request-max-retries",
        type=int,
        default=2,
        help="Maximum HTTP retries per model request.",
    )
    parser.add_argument(
        "--max-chunk-attempts",
        type=int,
        default=3,
        help="Maximum no-progress retry attempts before failing a chunk.",
    )
    parser.add_argument("--dataset", action="append", choices=["pope", "mathvista"])
    parser.add_argument("--model", action="append", choices=["gemini", "qwen"])
    parser.add_argument("--prompt", action="append", choices=["direct", "cot"])
    args = parser.parse_args()

    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be at least 1")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")
    if args.request_timeout_seconds < 1:
        raise ValueError("--request-timeout-seconds must be at least 1")
    if args.request_max_retries < 0:
        raise ValueError("--request-max-retries must be at least 0")
    if args.max_chunk_attempts < 1:
        raise ValueError("--max-chunk-attempts must be at least 1")

    groups = select_groups(
        datasets=_parse_filter(args.dataset),
        models=_parse_filter(args.model),
        prompts=_parse_filter(args.prompt),
    )
    resume_groups(
        groups,
        chunk_size=args.chunk_size,
        concurrency=args.concurrency,
        request_timeout_seconds=args.request_timeout_seconds,
        request_max_retries=args.request_max_retries,
        max_chunk_attempts=args.max_chunk_attempts,
    )
    print("ONE_TENTH_INFERENCE_DONE", flush=True)


if __name__ == "__main__":
    main()

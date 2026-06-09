"""按实验组恢复模型回答和 detector 结果，负责分批推进与失败重试。"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

from src.datasets.jsonl import read_json_records
from src.detectors.zero_shot_judge import detect_file
from src.models.openai_compatible import get_provider_config
from src.models.run_inference import (
    _is_compatible_existing_response_row,
    _is_invalid_response_row,
    resolve_prompt_type,
    run_inference,
)
from src.pipelines.experiment_groups import DetectorGroup, INFERENCE_GROUPS, InferenceGroup
from src.pipelines.jsonl_outputs import OutputStatus
from src.pipelines.result_store import (
    inspect_detector_group,
    inspect_inference_group,
    target_response_ids,
    target_sample_ids,
)


def count_completed_target_samples(group: InferenceGroup) -> int:
    return inspect_inference_group(group).status.valid


def count_completed_target_prefix(group: InferenceGroup) -> int:
    if not Path(group.output_path).exists():
        return 0
    rows_by_sample = {
        str(row.get("sample_id", "")): row for row in read_json_records(group.output_path)
    }
    completed = 0
    for sample_id in target_sample_ids(group):
        row = rows_by_sample.get(sample_id)
        if row is None or _is_invalid_response_row(row):
            break
        completed += 1
    return completed


def resume_inference_groups(
    groups: list[InferenceGroup],
    *,
    chunk_size: int,
    concurrency: int = 10,
    request_timeout_seconds: int = 180,
    request_max_retries: int = 2,
    max_chunk_attempts: int = 3,
) -> None:
    """先锁定 response 输出文件，再按 chunk 推进每个 inference group。"""

    with _inference_output_locks(groups):
        _resume_inference_groups_locked(
            groups,
            chunk_size=chunk_size,
            concurrency=concurrency,
            request_timeout_seconds=request_timeout_seconds,
            request_max_retries=request_max_retries,
            max_chunk_attempts=max_chunk_attempts,
        )


def _resume_inference_groups_locked(
    groups: list[InferenceGroup],
    *,
    chunk_size: int,
    concurrency: int = 10,
    request_timeout_seconds: int = 180,
    request_max_retries: int = 2,
    max_chunk_attempts: int = 3,
) -> None:
    """按 chunk 推进每个 inference group，只有产生新有效行才视为成功推进。"""

    _validate_positive("chunk_size", chunk_size)
    _validate_positive("concurrency", concurrency)
    _validate_positive("request_timeout_seconds", request_timeout_seconds)
    _validate_positive("request_max_retries", request_max_retries + 1)
    _validate_positive("max_chunk_attempts", max_chunk_attempts)

    chunk_attempts = {group.run_id: 0 for group in groups}
    completed_groups: set[str] = set()

    while True:
        pending_groups = []
        ran_chunk = False
        stalled_groups = []

        for group in groups:
            prefix_completed = count_completed_target_prefix(group)
            status = inspect_inference_group(group).status
            total_completed = status.valid
            if status.complete:
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
                    f"total={total_completed}/{group.limit} missing={status.missing} "
                    f"invalid={status.invalid} duplicates={status.duplicates}"
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
                    resume_provider_model_agnostic=group.model == "qwen",
                )
            except Exception as exc:
                run_error = exc

            next_prefix_completed = count_completed_target_prefix(group)
            next_status = inspect_inference_group(group).status
            next_total_completed = next_status.valid
            if next_status.complete or next_total_completed > total_completed:
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
            print("RESPONSES_DONE", flush=True)
            return
        if not ran_chunk:
            stalled = "; ".join(stalled_groups)
            raise RuntimeError(f"Resume stalled for all pending response groups: {stalled}")


def resume_detector_groups(
    groups: list[DetectorGroup],
    *,
    chunk_size: int,
    concurrency: int = 10,
    request_timeout_seconds: int = 180,
    request_max_retries: int = 2,
    judge_max_attempts: int = 3,
    max_chunk_attempts: int = 3,
    overwrite: bool = False,
) -> None:
    """先锁定 detector 输出文件，再按 group 恢复判定结果。"""

    with _detector_output_locks(groups):
        _resume_detector_groups_locked(
            groups,
            chunk_size=chunk_size,
            concurrency=concurrency,
            request_timeout_seconds=request_timeout_seconds,
            request_max_retries=request_max_retries,
            judge_max_attempts=judge_max_attempts,
            max_chunk_attempts=max_chunk_attempts,
            overwrite=overwrite,
        )


@contextmanager
def _inference_output_locks(groups: list[InferenceGroup]):
    with _output_locks([group.output_path for group in groups], kind="Response"):
        yield


@contextmanager
def _detector_output_locks(groups: list[DetectorGroup]):
    with _output_locks([group.output_path for group in groups], kind="Detector"):
        yield


@contextmanager
def _output_locks(output_paths: list[str], *, kind: str):
    acquired: list[Path] = []
    try:
        for output_path in sorted({Path(path) for path in output_paths}):
            lock_path = output_path.with_suffix(output_path.suffix + ".lock")
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError as exc:
                raise RuntimeError(
                    f"{kind} output is already locked by another run: {output_path}"
                ) from exc
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                file.write(f"pid={os.getpid()}\n")
            acquired.append(lock_path)
        yield
    finally:
        for lock_path in reversed(acquired):
            lock_path.unlink(missing_ok=True)


def _resume_detector_groups_locked(
    groups: list[DetectorGroup],
    *,
    chunk_size: int,
    concurrency: int = 10,
    request_timeout_seconds: int = 180,
    request_max_retries: int = 2,
    judge_max_attempts: int = 3,
    max_chunk_attempts: int = 3,
    overwrite: bool = False,
) -> None:
    """先确认模型回答完整，再按 chunk 恢复 detector 判定。"""

    _validate_positive("chunk_size", chunk_size)
    _validate_positive("concurrency", concurrency)
    _validate_positive("request_timeout_seconds", request_timeout_seconds)
    _validate_positive("request_max_retries", request_max_retries + 1)
    _validate_positive("judge_max_attempts", judge_max_attempts)
    _validate_positive("max_chunk_attempts", max_chunk_attempts)

    chunk_attempts = {group.run_id: 0 for group in groups}
    overwritten_groups: set[str] = set()
    completed_groups: set[str] = set()

    while True:
        pending_groups = []
        ran_chunk = False
        stalled_groups = []

        for group in groups:
            _assert_response_file_ready(group)
            group_overwrite = overwrite and group.run_id not in overwritten_groups
            status = _detector_group_status(group, overwrite=group_overwrite)
            target_count = status.valid + status.invalid + status.missing
            if target_count > 0 and status.complete:
                if group.run_id not in completed_groups:
                    print(
                        f"SKIP {group.run_id} total={status.valid}/{target_count}",
                        flush=True,
                    )
                    completed_groups.add(group.run_id)
                continue

            pending_groups.append(group)
            if chunk_attempts[group.run_id] >= max_chunk_attempts:
                stalled_groups.append(
                    f"{group.run_id} total={status.valid}/{target_count} "
                    f"missing={status.missing} invalid={status.invalid}"
                )
                continue

            ran_chunk = True
            target_limit = min(target_count, status.valid + chunk_size)
            print(
                f"RUN {group.run_id} target={target_limit} total={status.valid}/{target_count}",
                flush=True,
            )
            run_error: Exception | None = None
            try:
                detect_file(
                    samples_path=group.samples_path,
                    responses_path=group.responses_path,
                    output_path=group.output_path,
                    provider=group.provider,
                    run_id=group.run_id,
                    limit=target_limit,
                    offset=group.offset,
                    resume=not group_overwrite,
                    overwrite=group_overwrite,
                    concurrency=concurrency,
                    judge_max_attempts=judge_max_attempts,
                    request_timeout_seconds=request_timeout_seconds,
                    request_max_retries=request_max_retries,
                )
            except Exception as exc:
                run_error = exc

            next_status = inspect_detector_group(group).status
            if next_status.valid > status.valid:
                print(
                    f"OK {group.run_id} total={next_status.valid}/{target_count}",
                    flush=True,
                )
                chunk_attempts[group.run_id] = 0
                overwritten_groups.add(group.run_id)
                continue

            chunk_attempts[group.run_id] += 1
            reason = run_error if run_error is not None else "no completed judgements"
            print(
                f"RETRY {group.run_id} attempt={chunk_attempts[group.run_id]}/"
                f"{max_chunk_attempts} total={next_status.valid}/{target_count}: {reason}",
                flush=True,
            )
            overwritten_groups.add(group.run_id)

        if not pending_groups:
            print("DETECTORS_DONE", flush=True)
            return
        if not ran_chunk:
            stalled = "; ".join(stalled_groups)
            raise RuntimeError(f"Resume stalled for all pending detector groups: {stalled}")


def _detector_group_status(group: DetectorGroup, *, overwrite: bool) -> OutputStatus:
    if not overwrite:
        return inspect_detector_group(group).status
    target_count = len(target_response_ids(group))
    return OutputStatus(
        path=group.output_path,
        total=0,
        valid=0,
        invalid=0,
        missing=target_count,
        duplicates=0,
    )


def _assert_response_file_ready(group: DetectorGroup) -> None:
    """detector 运行前的强前置条件：目标模型回答必须存在、有效且配置兼容。"""

    if not Path(group.responses_path).exists():
        raise RuntimeError(
            f"Model responses are missing for {group.run_id}: {group.responses_path}. "
            "Run the responses stage first."
        )
    expected_group = _expected_response_group(group)
    response_status = inspect_inference_group(expected_group).status
    if response_status.missing or response_status.duplicates:
        parts = [
            f"valid={response_status.valid}",
            f"missing={response_status.missing}",
            f"invalid={response_status.invalid}",
            f"duplicates={response_status.duplicates}",
        ]
        if response_status.missing_examples:
            parts.append(f"missing_examples={list(response_status.missing_examples)}")
        if response_status.invalid_examples:
            parts.append(f"invalid_examples={list(response_status.invalid_examples)}")
        raise RuntimeError(
            f"Model responses are not complete for detector {group.run_id}: "
            + "; ".join(parts)
            + ". Run the responses stage first."
        )
    provider_config = get_provider_config(expected_group.provider)
    expected_prompt_type = resolve_prompt_type(expected_group.prompt_path)
    invalid = []
    missing_sample_id = []
    incompatible = []
    for index, row in enumerate(read_json_records(group.responses_path)):
        if index < group.offset:
            continue
        if group.limit is not None and index >= group.offset + group.limit:
            break
        response_id = f"{row.get('run_id')}:{row.get('sample_id')}"
        if _is_invalid_response_row(row):
            invalid.append(response_id)
        if not str(row.get("sample_id", "")):
            missing_sample_id.append(str(index))
        elif not _is_compatible_existing_response_row(
            row,
            run_id=expected_group.run_id,
            dataset=expected_group.dataset,
            model=provider_config.default_model,
            model_type=provider_config.model_type,
            prompt_type=expected_prompt_type,
            prompt_version=expected_group.version,
            provider=expected_group.provider,
            max_tokens=expected_group.max_tokens,
            temperature=0,
            provider_model_agnostic=expected_group.model == "qwen",
        ):
            incompatible.append(response_id)
    if invalid or missing_sample_id or incompatible:
        parts = []
        if invalid:
            parts.append(f"invalid={invalid[:10]}")
        if missing_sample_id:
            parts.append(f"missing_sample_id={missing_sample_id[:10]}")
        if incompatible:
            parts.append(f"incompatible={incompatible[:10]}")
        raise RuntimeError(
            f"Model responses are not ready for detector {group.run_id}: "
            + "; ".join(parts)
            + ". Run the responses stage first."
        )


def _expected_response_group(group: DetectorGroup) -> InferenceGroup:
    for inference_group in INFERENCE_GROUPS:
        if inference_group.output_path == group.responses_path:
            return inference_group
    raise RuntimeError(
        f"No registered inference group produces responses for detector {group.run_id}: "
        f"{group.responses_path}. Register the response source before running detectors."
    )


def _validate_positive(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be at least 1")

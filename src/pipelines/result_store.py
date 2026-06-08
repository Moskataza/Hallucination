"""检查已生成结果是否可复用，并汇总每个 pipeline 组的完成状态。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.datasets.jsonl import read_json_records
from src.datasets.schema import EvalSample
from src.detectors.zero_shot_judge import _should_retry_detector_row
from src.models.openai_compatible import get_provider_config
from src.models.run_inference import _is_invalid_response_row, resolve_prompt_type
from src.pipelines.compatibility import (
    is_detector_row_compatible,
    is_response_row_compatible,
)
from src.pipelines.experiment_groups import DetectorGroup, InferenceGroup
from src.pipelines.jsonl_outputs import OutputStatus, inspect_output


@dataclass(frozen=True)
class PipelineStatus:
    group_id: str
    stage: str
    status: OutputStatus


def target_sample_ids(group: InferenceGroup) -> list[str]:
    return [
        str(sample["sample_id"])
        for index, sample in enumerate(read_json_records(group.dataset_path))
        if index >= group.offset and index < group.offset + group.limit
    ]


def target_response_ids(group: DetectorGroup) -> list[str]:
    return [
        f"{row['run_id']}:{row['sample_id']}"
        for index, row in enumerate(read_json_records(group.responses_path))
        if index >= group.offset
        and (group.limit is None or index < group.offset + group.limit)
    ]


def inspect_inference_group(group: InferenceGroup) -> PipelineStatus:
    """检查模型回答文件是否覆盖目标样本且与当前 group 配置一致。"""

    target_ids = set(target_sample_ids(group))
    provider_config = get_provider_config(group.provider)
    prompt_type = _group_prompt_type(group.prompt_path, group.prompt)
    status = inspect_output(
        group.output_path,
        target_ids=target_ids,
        key_fn=lambda row: str(row.get("sample_id", "")),
        invalid_row=lambda row: _is_invalid_response_row(row)
        or not is_response_row_compatible(
            row,
            run_id=group.run_id,
            dataset=group.dataset,
            model=provider_config.default_model,
            model_type=provider_config.model_type,
            prompt_type=prompt_type,
            prompt_version=group.version,
            provider=group.provider,
            max_tokens=group.max_tokens,
            temperature=0,
        ),
    )
    return PipelineStatus(group_id=group.run_id, stage="responses", status=status)


def inspect_detector_group(group: DetectorGroup) -> PipelineStatus:
    """检查 detector 输出是否覆盖目标 response，并验证 sample/response 身份匹配。"""

    samples = _load_samples(group.samples_path)
    target_ids = set(target_response_ids(group)) if Path(group.responses_path).exists() else set()
    response_rows = {
        f"{row['run_id']}:{row['sample_id']}": row
        for row in read_json_records(group.responses_path)
    } if Path(group.responses_path).exists() else {}
    status = inspect_output(
        group.output_path,
        target_ids=target_ids,
        key_fn=lambda row: str(row.get("model_response_id", "")),
        invalid_row=lambda row: _should_retry_detector_row(row, samples)
        or not _is_detector_output_compatible(row, group, response_rows, samples),
    )
    return PipelineStatus(group_id=group.run_id, stage="detectors", status=status)


def _group_prompt_type(prompt_path: str, prompt: str) -> str:
    try:
        return resolve_prompt_type(prompt_path)
    except ValueError:
        if prompt == "cot":
            return "evidence_grounded_cot"
        return prompt


def _is_detector_output_compatible(
    row: dict[str, object],
    group: DetectorGroup,
    response_rows: dict[str, dict[str, object]],
    samples: dict[str, EvalSample],
) -> bool:
    response_id = str(row.get("model_response_id", ""))
    response = response_rows.get(response_id)
    if response is None:
        return False
    sample_id = str(row.get("sample_id", ""))
    if sample_id not in samples:
        return False
    if sample_id != str(response.get("sample_id", "")):
        return False
    if response_id != f"{response.get('run_id')}:{response.get('sample_id')}":
        return False
    return is_detector_row_compatible(
        row,
        run_id=group.run_id,
        detector=group.detector,
        dataset=group.dataset,
        model=str(response.get("model", "")),
        prompt_type=str(response.get("prompt_type", "")),
        judge_provider=group.provider,
    )


def _load_samples(samples_path: str) -> dict[str, EvalSample]:
    return {
        str(row["sample_id"]): EvalSample.from_dict(row)
        for row in read_json_records(samples_path)
    }

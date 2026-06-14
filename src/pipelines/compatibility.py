"""严格判断历史 JSONL 行是否属于当前实验配置，避免混用旧结果。"""

from __future__ import annotations

from typing import Any


_DETECTOR_RESULT_NAMES = {
    "zero_shot": "response_claim_zero_shot_judge",
}


def detector_result_name(detector: str) -> str:
    """把命令行 detector 名称映射到结果文件中的 detector 字段。"""
    return _DETECTOR_RESULT_NAMES.get(detector, detector)


def is_response_row_compatible(
    row: dict[str, Any],
    *,
    run_id: str,
    dataset: str,
    model: str,
    model_type: str,
    prompt_type: str,
    prompt_version: str,
    provider: str,
    max_tokens: int,
    temperature: float,
) -> bool:
    """检查旧模型回答行是否与当前推理配置完全一致。"""
    if not _matches_optional(row.get("run_id"), run_id):
        return False
    if not _matches_optional(row.get("dataset"), dataset):
        return False
    if not _matches_optional(row.get("model"), model):
        return False
    if not _matches_optional(row.get("model_type"), model_type):
        return False
    if not _matches_optional(row.get("prompt_type"), prompt_type):
        return False
    if not _matches_optional(row.get("prompt_version"), prompt_version):
        return False

    metadata = row.get("inference_metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    if not _matches_optional(metadata.get("provider"), provider):
        return False
    if not _matches_optional(metadata.get("max_tokens"), max_tokens):
        return False
    if not _matches_optional(metadata.get("temperature"), temperature):
        return False
    return True


def is_detector_row_compatible(
    row: dict[str, Any],
    *,
    run_id: str,
    detector: str,
    dataset: str,
    model: str,
    prompt_type: str,
    judge_provider: str | None = None,
) -> bool:
    """检查旧 detector 行是否与当前 detector 配置和 judge provider 一致。"""
    if not _matches_optional(row.get("run_id"), run_id):
        return False
    if not _matches_optional(row.get("detector"), detector_result_name(detector)):
        return False
    if not _matches_optional(row.get("dataset"), dataset):
        return False
    if not _matches_optional(row.get("model"), model):
        return False
    if not _matches_optional(row.get("prompt_type"), prompt_type):
        return False

    if judge_provider is None:
        return True

    details = row.get("details")
    if not isinstance(details, dict):
        details = {}
    return _matches_optional(details.get("judge_provider"), judge_provider)


def _matches_optional(actual: Any, expected: Any) -> bool:
    return actual == expected

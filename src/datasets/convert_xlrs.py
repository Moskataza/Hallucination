"""将 XLRS-Bench 风格记录转换为统一评测样本格式。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.datasets.jsonl import read_jsonl, write_jsonl
from src.datasets.schema import EvalSample

_QUESTION_KEYS = ("question", "query", "prompt", "instruction")
_ANSWER_KEYS = ("answer", "label", "gt", "ground_truth")
_IMAGE_KEYS = ("image", "image_path", "img", "file_name")
_CHOICES_KEYS = ("choices", "options", "candidates")


def convert_xlrs_record(record: dict[str, Any], image_root: str | Path = "") -> EvalSample:
    """把 XLRS-Bench 单条记录映射到统一 EvalSample。"""
    image_value = _first_present(record, _IMAGE_KEYS, "")
    image_path = str(Path(image_root) / str(image_value)) if image_root and image_value else str(image_value)
    sample_id = record.get("sample_id", record.get("id", record.get("qid", image_value)))
    # 保留未映射字段，便于后续追溯 XLRS 原始元数据。
    metadata = {k: v for k, v in record.items() if k not in {*_QUESTION_KEYS, *_ANSWER_KEYS, *_IMAGE_KEYS, *_CHOICES_KEYS}}
    return EvalSample(
        sample_id=f"xlrs_{sample_id}",
        dataset="xlrs_bench",
        task_type="remote_sensing_vqa",
        image_path=image_path,
        question=str(_first_present(record, _QUESTION_KEYS, "")),
        reference_answer=str(_first_present(record, _ANSWER_KEYS, "")),
        choices=_first_present(record, _CHOICES_KEYS, None),
        metadata=metadata,
    )


def convert_xlrs_file(input_path: str | Path, output_path: str | Path, image_root: str | Path = "") -> None:
    """批量转换 XLRS-Bench JSONL 并写出统一样本文件。"""
    samples = (convert_xlrs_record(record, image_root).to_dict() for record in read_jsonl(input_path))
    write_jsonl(output_path, samples)


def _first_present(record: dict[str, Any], keys: tuple[str, ...], default: Any) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return default


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert XLRS-Bench-like JSONL records to the unified eval schema.")
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("--image-root", default="")
    args = parser.parse_args()
    convert_xlrs_file(args.input_path, args.output_path, args.image_root)


if __name__ == "__main__":
    main()

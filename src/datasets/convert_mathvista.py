"""将 MathVista 风格记录转换为统一评测样本格式。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.datasets.jsonl import read_jsonl, write_jsonl
from src.datasets.schema import EvalSample


def convert_mathvista_record(
    record: dict[str, Any], image_root: str | Path = ""
) -> EvalSample:
    """把 MathVista 单条记录映射到统一 EvalSample。"""
    image_value = str(record.get("image", record.get("image_path", "")))
    image_path = (
        (Path(image_root) / image_value).as_posix()
        if image_root and image_value
        else image_value
    )
    pid = record.get("pid", record.get("id", image_value))
    metadata = dict(record.get("metadata", {}))
    for key in ("question_type", "answer_type", "unit", "precision", "query"):
        if key in record:
            metadata[key] = record[key]
    raw_answer = record.get("answer")
    answer = "" if raw_answer is None else str(raw_answer).strip()
    metadata["answer_available"] = bool(answer)
    return EvalSample(
        sample_id=f"mathvista_{pid}",
        dataset="mathvista",
        task_type="visual_math_reasoning",
        image_path=image_path,
        question=str(record.get("question", record.get("query", ""))),
        reference_answer=answer or "UNAVAILABLE",
        choices=record.get("choices"),
        metadata=metadata,
    )


def convert_mathvista_file(
    input_path: str | Path, output_path: str | Path, image_root: str | Path = ""
) -> None:
    """批量转换 MathVista JSONL 并写出统一样本文件。"""
    samples = (
        convert_mathvista_record(record, image_root).to_dict()
        for record in read_jsonl(input_path)
    )
    write_jsonl(output_path, samples)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert MathVista-like JSONL records to the unified eval schema."
    )
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("--image-root", default="")
    args = parser.parse_args()
    convert_mathvista_file(args.input_path, args.output_path, args.image_root)


if __name__ == "__main__":
    main()

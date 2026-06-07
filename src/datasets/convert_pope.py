from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.datasets.jsonl import read_json_records, write_jsonl
from src.datasets.schema import EvalSample, TaxonomyLabel


def convert_pope_record(
    record: dict[str, Any],
    image_root: str | Path = "",
    split: str | None = None,
) -> EvalSample:
    image_name = str(record.get("image", ""))
    image_path = _join_image_path(image_root, image_name)
    question_id = record.get("question_id", record.get("id", image_name))
    resolved_split = split or record.get("split") or record.get("category") or record.get("pope_split")
    sample_id = f"pope_{resolved_split}_{question_id}" if resolved_split else f"pope_{question_id}"
    return EvalSample(
        sample_id=sample_id,
        dataset="pope",
        task_type="vqa_yes_no",
        image_path=image_path,
        question=str(record.get("text", record.get("question", ""))),
        reference_answer=str(record.get("label", record.get("answer", ""))).strip(),
        choices=None,
        metadata={
            "split": resolved_split,
            "source_record": {
                k: v for k, v in record.items() if k not in {"text", "question", "label", "answer"}
            },
        },
        taxonomy_hint=TaxonomyLabel(coarse="Factual", fine="OBJ"),
    )


def convert_pope_file(
    input_path: str | Path,
    output_path: str | Path,
    image_root: str | Path = "",
    split: str | None = None,
) -> None:
    resolved_split = split or _infer_split_from_path(input_path)
    samples = (
        convert_pope_record(record, image_root, resolved_split).to_dict()
        for record in read_json_records(input_path)
    )
    write_jsonl(output_path, samples)


def _join_image_path(image_root: str | Path, image_name: str) -> str:
    if not image_root or not image_name:
        return image_name
    return (Path(image_root) / image_name).as_posix()


def _infer_split_from_path(input_path: str | Path) -> str | None:
    stem = Path(input_path).stem.lower()
    for split in ("random", "popular", "adversarial"):
        if split in stem:
            return split
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert POPE JSON, JSON list, or JSONL records to the unified eval schema."
    )
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("--image-root", default="")
    parser.add_argument("--split", default=None)
    args = parser.parse_args()
    convert_pope_file(args.input_path, args.output_path, args.image_root, args.split)


if __name__ == "__main__":
    main()

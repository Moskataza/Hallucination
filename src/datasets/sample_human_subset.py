"""按数据集、模型和 prompt 分组抽取人工标注子集。"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from typing import Any

from src.datasets.jsonl import read_jsonl, write_jsonl


def sample_human_subset(rows: list[dict[str, Any]], per_group: int, seed: int = 42) -> list[dict[str, Any]]:
    """在每个数据集-模型-prompt 组内随机抽取固定数量样本。"""
    rng = random.Random(seed)
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("dataset", "")), str(row.get("model", "")), str(row.get("prompt_type", "")))].append(row)

    selected: list[dict[str, Any]] = []
    for group_rows in groups.values():
        rng.shuffle(group_rows)
        selected.extend(group_rows[:per_group])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample a balanced human annotation subset from model response JSONL.")
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("--per-group", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = list(read_jsonl(args.input_path))
    write_jsonl(args.output_path, sample_human_subset(rows, args.per_group, args.seed))


if __name__ == "__main__":
    main()

"""JSONL 输出文件的原子写入、resume 清理和完整性检查工具。"""

from __future__ import annotations

import json
import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.datasets.jsonl import read_json_records, write_jsonl

InvalidRowPredicate = Callable[[dict[str, Any]], bool]
KeyFunction = Callable[[dict[str, Any]], str]


@dataclass(frozen=True)
class OutputStatus:
    """描述输出文件中目标 key 的有效、缺失、无效和重复数量。"""
    path: str
    total: int
    valid: int
    invalid: int
    missing: int
    duplicates: int
    invalid_examples: tuple[str, ...] = ()
    missing_examples: tuple[str, ...] = ()
    duplicate_examples: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return self.missing == 0 and self.invalid == 0 and self.duplicates == 0


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    """安全读取可选存在的 JSON/JSONL 输出文件。"""
    output = Path(path)
    if not output.exists():
        return []
    return list(read_json_records(output))


def write_indexed_rows(
    output: str | Path,
    indexed_rows: Iterable[tuple[int, dict[str, Any]]],
    *,
    key_fn: KeyFunction,
) -> None:
    """按原始样本顺序写回 JSONL，并用 key 去重保留最新行。"""

    output_path = Path(output)
    rows_by_key = {key_fn(row): (index, row) for index, row in indexed_rows}
    rows = [row for _, row in sorted(rows_by_key.values(), key=lambda item: item[0])]
    atomic_write_jsonl(output_path, rows)


def atomic_write_jsonl(output: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """先写临时文件再替换目标文件，降低中断导致的半写风险。"""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output_path.parent, delete=False
    ) as file:
        temp_path = Path(file.name)
        try:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
            file.flush()
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
    for attempt in range(5):
        try:
            temp_path.replace(output_path)
            return
        except PermissionError:
            if attempt == 4:
                temp_path.unlink(missing_ok=True)
                raise
            time.sleep(0.1 * (attempt + 1))


def prepare_resume_rows(
    output: str | Path,
    *,
    overwrite: bool,
    resume: bool,
    key_fn: KeyFunction,
    invalid_row: InvalidRowPredicate,
) -> tuple[list[dict[str, Any]], set[str]]:
    """resume 前清理无效旧行，返回仍可复用的行和已完成 key。"""

    output_path = Path(output)
    if overwrite:
        write_jsonl(output_path, [])
        return [], set()
    if not output_path.exists():
        return [], set()
    if not resume:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite or --resume."
        )
    # resume 前剔除无效旧行，避免失败占位或过期配置阻塞重跑。
    rows = [row for row in read_json_records(output_path) if not invalid_row(row)]
    write_indexed_rows(
        output_path,
        [(index, row) for index, row in enumerate(rows)],
        key_fn=key_fn,
    )
    return rows, {key_fn(row) for row in rows}


def inspect_output(
    output: str | Path,
    *,
    target_ids: set[str],
    key_fn: KeyFunction,
    invalid_row: InvalidRowPredicate,
    example_limit: int = 10,
) -> OutputStatus:
    """统计目标 key 的 valid、missing、invalid 和 duplicate 情况。"""

    output_path = Path(output)
    rows = read_rows(output_path)
    rows_by_key: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    invalid: list[str] = []
    for row in rows:
        key = key_fn(row)
        if key in rows_by_key:
            duplicates.append(key)
        rows_by_key[key] = row
    for key in sorted(target_ids & set(rows_by_key)):
        if invalid_row(rows_by_key[key]):
            invalid.append(key)
    missing = sorted(target_ids - set(rows_by_key))
    return OutputStatus(
        path=str(output_path),
        total=len(rows),
        valid=len(target_ids & set(rows_by_key)) - len(invalid),
        invalid=len(invalid),
        missing=len(missing),
        duplicates=len(duplicates),
        invalid_examples=tuple(invalid[:example_limit]),
        missing_examples=tuple(missing[:example_limit]),
        duplicate_examples=tuple(duplicates[:example_limit]),
    )


def validate_complete_output(
    output: str | Path,
    *,
    target_ids: set[str],
    key_fn: KeyFunction,
    invalid_row: InvalidRowPredicate,
    label: str,
) -> None:
    """校验输出覆盖所有目标 key，否则报告缺失、无效或重复示例。"""
    status = inspect_output(
        output, target_ids=target_ids, key_fn=key_fn, invalid_row=invalid_row
    )
    if status.complete:
        return
    parts = []
    if status.missing:
        parts.append(f"missing={list(status.missing_examples)}")
    if status.invalid:
        parts.append(f"invalid={list(status.invalid_examples)}")
    if status.duplicates:
        parts.append(f"duplicates={list(status.duplicate_examples)}")
    raise RuntimeError(f"{label} output is incomplete or invalid: " + "; ".join(parts))

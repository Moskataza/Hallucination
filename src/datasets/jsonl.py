from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def read_json_records(path: str | Path) -> Iterator[dict[str, Any]]:
    input_path = Path(path)
    text = input_path.read_text(encoding="utf-8").strip()
    if not text:
        return

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        yield from _read_line_delimited_json(input_path, text)
        return

    if isinstance(payload, list):
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                raise ValueError(f"Expected object at {input_path}[{index}], got {type(item).__name__}")
            yield item
        return

    if isinstance(payload, dict):
        yield payload
        return

    raise ValueError(f"Expected JSON object, JSON list, or JSONL records in {input_path}")


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    yield from read_json_records(path)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_line_delimited_json(path: Path, text: str) -> Iterator[dict[str, Any]]:
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON record at {path}:{line_number}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Expected object at {path}:{line_number}, got {type(payload).__name__}")
        yield payload

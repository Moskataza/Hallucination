from __future__ import annotations

from typing import Any


def select_failure_cases(rows: list[dict[str, Any]], limit: int = 4) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for row in rows:
        if row.get("is_hallucination") is True:
            _append_if_new(selected, seen_ids, row, limit)
        if len(selected) >= limit:
            return selected

    for row in rows:
        _append_if_new(selected, seen_ids, row, limit)
        if len(selected) >= limit:
            return selected

    return selected


def _append_if_new(selected: list[dict[str, Any]], seen_ids: set[str], row: dict[str, Any], limit: int) -> None:
    row_id = str(row.get("model_response_id") or row.get("sample_id") or id(row))
    if row_id in seen_ids or len(selected) >= limit:
        return
    seen_ids.add(row_id)
    selected.append(row)

"""把分析结果行渲染成安全的 Markdown 表格。"""

from __future__ import annotations

from typing import Any


def rows_to_markdown(rows: list[dict[str, Any]], columns: list[str]) -> str:
    """按指定列顺序把字典行转换为 Markdown 表格。"""
    header = "| " + " | ".join(_escape_cell(column) for column in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(_escape_cell(row.get(column, "")) for column in columns) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _escape_cell(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\r", " ").replace("\n", " ")

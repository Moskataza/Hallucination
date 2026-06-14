"""提供简单占位符替换的 prompt 模板渲染。"""

from __future__ import annotations

_PLACEHOLDERS = (
    "dataset",
    "task_type",
    "question",
    "choices",
    "reference_answer",
    "metadata",
    "model_response",
    "visual_evidence",
    "reasoning",
    "final_answer",
    "taxonomy_definition",
)


def render_prompt_template(template: str, values: dict[str, str]) -> str:
    """用空字符串兜底替换模板中的固定占位符。"""
    rendered = template
    for key in _PLACEHOLDERS:
        rendered = rendered.replace("{" + key + "}", values.get(key, ""))
    return rendered

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
    rendered = template
    for key in _PLACEHOLDERS:
        rendered = rendered.replace("{" + key + "}", values.get(key, ""))
    return rendered

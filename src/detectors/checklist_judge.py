"""渲染 checklist judge 的提示词输入。"""

from __future__ import annotations

from pathlib import Path

from src.datasets.schema import EvalSample, ModelResponse
from src.detectors.prompt_rendering import render_prompt_template


def render_checklist_judge_prompt(
    sample: EvalSample,
    response: ModelResponse,
    taxonomy_definition: str,
    template_path: str | Path = "prompts/judge/checklist_judge.txt",
) -> str:
    """把样本、回答和 taxonomy 定义填入 checklist judge 模板。"""
    template = Path(template_path).read_text(encoding="utf-8")
    return render_prompt_template(
        template,
        {
            "dataset": sample.dataset,
            "question": sample.question,
            "reference_answer": sample.reference_answer,
            "model_response": response.raw_response,
            "visual_evidence": response.parsed.visual_evidence,
            "reasoning": response.parsed.reasoning,
            "final_answer": response.parsed.final_answer,
            "taxonomy_definition": taxonomy_definition,
        },
    )

"""把统一样本和模型回答转换为 EasyDetect 所需输入。"""

from __future__ import annotations

from src.datasets.schema import EvalSample, ModelResponse


def build_easydetect_input(sample: EvalSample, response: ModelResponse) -> dict[str, str]:
    """抽取 EasyDetect 需要的样本、问题、参考答案和模型回答字段。"""
    return {
        "sample_id": sample.sample_id,
        "image_path": sample.image_path,
        "question": sample.question,
        "reference_answer": sample.reference_answer,
        "model_response": response.raw_response,
    }

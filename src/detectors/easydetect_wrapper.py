from __future__ import annotations

from src.datasets.schema import EvalSample, ModelResponse


def build_easydetect_input(sample: EvalSample, response: ModelResponse) -> dict[str, str]:
    return {
        "sample_id": sample.sample_id,
        "image_path": sample.image_path,
        "question": sample.question,
        "reference_answer": sample.reference_answer,
        "model_response": response.raw_response,
    }

"""实验样本、模型回答和 detector 结果的统一 JSONL 数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from src.detectors.taxonomy import normalize_coarse_label, normalize_fine_label

DatasetName = Literal["pope", "mathvista", "xlrs_bench"]
CoarseTaxonomy = Literal["Factual", "Logical", "None", "Unclear"]
FineTaxonomy = Literal["OBJ", "ATT", "SPA", "IR", "CI", "INC", "SO", "None", "Unclear"]
PromptType = Literal["direct", "evidence_grounded_cot"]

_DATASETS = {"pope", "mathvista", "xlrs_bench"}
_MODEL_TYPES = {"closed", "open", "unknown"}
_PROMPT_TYPES = {"direct", "evidence_grounded_cot"}
_PARSE_STATUSES = {"ok", "fallback", "failed"}
_CONFIDENCE_VALUES = {"high", "medium", "low"}


@dataclass(frozen=True)
class TaxonomyLabel:
    """Detector 输出的粗粒度和细粒度幻觉类型标签。"""

    coarse: CoarseTaxonomy = "None"
    fine: FineTaxonomy = "None"

    def __post_init__(self) -> None:
        fine = normalize_fine_label(self.fine)
        coarse = normalize_coarse_label(self.coarse, fine)
        object.__setattr__(self, "fine", fine)
        object.__setattr__(self, "coarse", coarse)

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "TaxonomyLabel":
        if not value:
            return cls()
        return cls(
            coarse=value.get("coarse", "None"),
            fine=value.get("fine", "None"),
        )

    def to_dict(self) -> dict[str, str]:
        return {"coarse": self.coarse, "fine": self.fine}


@dataclass(frozen=True)
class EvalSample:
    """评测样本的规范格式，承载题目、图片路径和参考答案。"""

    sample_id: str
    dataset: DatasetName
    task_type: str
    image_path: str
    question: str
    reference_answer: str
    choices: list[str] | dict[str, str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    taxonomy_hint: TaxonomyLabel = field(default_factory=TaxonomyLabel)

    def __post_init__(self) -> None:
        _require_nonempty("sample_id", self.sample_id)
        _validate_member("dataset", self.dataset, _DATASETS)
        _require_nonempty("question", self.question)
        _require_nonempty("reference_answer", self.reference_answer)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvalSample":
        return cls(
            sample_id=str(value["sample_id"]),
            dataset=value["dataset"],
            task_type=str(value["task_type"]),
            image_path=str(value.get("image_path", "")),
            question=str(value["question"]),
            reference_answer=str(value["reference_answer"]),
            choices=value.get("choices"),
            metadata=dict(value.get("metadata", {})),
            taxonomy_hint=TaxonomyLabel.from_dict(value.get("taxonomy_hint")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "dataset": self.dataset,
            "task_type": self.task_type,
            "image_path": self.image_path,
            "question": self.question,
            "reference_answer": self.reference_answer,
            "choices": self.choices,
            "metadata": self.metadata,
            "taxonomy_hint": self.taxonomy_hint.to_dict(),
        }


@dataclass(frozen=True)
class ParsedResponse:
    """从模型原始回答中解析出的视觉证据、推理过程和最终答案。"""

    visual_evidence: str = ""
    reasoning: str = ""
    final_answer: str = ""
    parse_status: Literal["ok", "fallback", "failed"] = "failed"

    def __post_init__(self) -> None:
        _validate_member("parse_status", self.parse_status, _PARSE_STATUSES)

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "ParsedResponse":
        if not value:
            return cls()
        return cls(
            visual_evidence=str(value.get("visual_evidence", "")),
            reasoning=str(value.get("reasoning", "")),
            final_answer=str(value.get("final_answer", "")),
            parse_status=value.get("parse_status", "failed"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "visual_evidence": self.visual_evidence,
            "reasoning": self.reasoning,
            "final_answer": self.final_answer,
            "parse_status": self.parse_status,
        }


@dataclass(frozen=True)
class ModelResponse:
    """一次模型回答的完整记录，用于后续 detector 和结果复用校验。"""

    run_id: str
    sample_id: str
    dataset: DatasetName
    model: str
    model_type: Literal["closed", "open", "unknown"]
    prompt_type: PromptType
    prompt_version: str
    raw_response: str
    parsed: ParsedResponse = field(default_factory=ParsedResponse)
    inference_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonempty("run_id", self.run_id)
        _require_nonempty("sample_id", self.sample_id)
        _validate_member("dataset", self.dataset, _DATASETS)
        _validate_member("model_type", self.model_type, _MODEL_TYPES)
        _validate_member("prompt_type", self.prompt_type, _PROMPT_TYPES)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ModelResponse":
        return cls(
            run_id=str(value["run_id"]),
            sample_id=str(value["sample_id"]),
            dataset=value["dataset"],
            model=str(value["model"]),
            model_type=value.get("model_type", "unknown"),
            prompt_type=value["prompt_type"],
            prompt_version=str(value.get("prompt_version", "v1")),
            raw_response=str(value.get("raw_response", "")),
            parsed=ParsedResponse.from_dict(value.get("parsed")),
            inference_metadata=dict(value.get("inference_metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "sample_id": self.sample_id,
            "dataset": self.dataset,
            "model": self.model,
            "model_type": self.model_type,
            "prompt_type": self.prompt_type,
            "prompt_version": self.prompt_version,
            "raw_response": self.raw_response,
            "parsed": self.parsed.to_dict(),
            "inference_metadata": self.inference_metadata,
        }


@dataclass(frozen=True)
class DetectorResult:
    """zero-shot judge 等 detector 对单条模型回答的结构化判定结果。"""

    run_id: str
    sample_id: str
    model_response_id: str
    detector: str
    answer_correct: bool | None
    is_hallucination: bool | None
    taxonomy: TaxonomyLabel = field(default_factory=TaxonomyLabel)
    unsupported_visual_claim: bool | None = None
    confidence: Literal["high", "medium", "low"] | None = None
    explanation: str = ""
    raw_judge_response: str = ""
    dataset: DatasetName | None = None
    model: str = ""
    prompt_type: PromptType | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonempty("run_id", self.run_id)
        _require_nonempty("sample_id", self.sample_id)
        _require_nonempty("detector", self.detector)
        if self.confidence is not None:
            _validate_member("confidence", self.confidence, _CONFIDENCE_VALUES)
        if self.dataset is not None:
            _validate_member("dataset", self.dataset, _DATASETS)
        if self.prompt_type is not None:
            _validate_member("prompt_type", self.prompt_type, _PROMPT_TYPES)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DetectorResult":
        return cls(
            run_id=str(value["run_id"]),
            sample_id=str(value["sample_id"]),
            model_response_id=str(value.get("model_response_id", value["sample_id"])),
            detector=str(value["detector"]),
            answer_correct=value.get("answer_correct"),
            is_hallucination=value.get("is_hallucination"),
            taxonomy=TaxonomyLabel.from_dict(value.get("taxonomy")),
            unsupported_visual_claim=value.get("unsupported_visual_claim"),
            confidence=value.get("confidence"),
            explanation=str(value.get("explanation", "")),
            raw_judge_response=str(value.get("raw_judge_response", "")),
            dataset=value.get("dataset"),
            model=str(value.get("model", "")),
            prompt_type=value.get("prompt_type"),
            details=dict(value.get("details", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "sample_id": self.sample_id,
            "model_response_id": self.model_response_id,
            "detector": self.detector,
            "answer_correct": self.answer_correct,
            "is_hallucination": self.is_hallucination,
            "taxonomy": self.taxonomy.to_dict(),
            "unsupported_visual_claim": self.unsupported_visual_claim,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "raw_judge_response": self.raw_judge_response,
            "dataset": self.dataset,
            "model": self.model,
            "prompt_type": self.prompt_type,
            "details": self.details,
        }


def _require_nonempty(name: str, value: str) -> None:
    if not str(value).strip():
        raise ValueError(f"{name} must be non-empty")


def _validate_member(name: str, value: str, allowed: set[str]) -> None:
    if value not in allowed:
        raise ValueError(
            f"Invalid {name}: {value!r}. Expected one of {sorted(allowed)}"
        )

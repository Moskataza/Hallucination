"""集中登记实验组，统一描述模型回答与 detector 输出的路径和配置。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

DatasetName = Literal["pope", "mathvista", "xlrs_bench"]
PromptName = Literal["direct", "cot"]
ModelName = Literal["gemini", "qwen", "gpt54"]
StageName = Literal["responses", "detectors", "all", "validate"]


@dataclass(frozen=True)
class InferenceGroup:
    """一组可恢复的模型回答任务，定义输入样本、prompt、provider 和输出文件。"""

    run_id: str
    dataset_path: str
    prompt_path: str
    output_path: str
    provider: str
    limit: int
    max_tokens: int
    dataset: DatasetName
    model: ModelName
    prompt: PromptName
    experiment: str = "one_tenth"
    version: str = "v1"
    offset: int = 0


@dataclass(frozen=True)
class DetectorGroup:
    """一组 detector 任务，定义要读取的模型回答文件和写入的判定文件。"""

    run_id: str
    samples_path: str
    responses_path: str
    output_path: str
    provider: str
    detector: str
    dataset: DatasetName
    model: ModelName
    prompt: PromptName
    experiment: str = "one_tenth"
    version: str = "v1"
    response_version: str = "v1"
    limit: int | None = None
    offset: int = 0


ONE_TENTH_INFERENCE_GROUPS: tuple[InferenceGroup, ...] = (
    InferenceGroup(
        experiment="one_tenth",
        run_id="one_tenth_pope_gemini_direct_v1",
        dataset_path="data/processed/pope_eval.jsonl",
        prompt_path="prompts/answer/direct_pope.txt",
        output_path="outputs/model_responses/one_tenth_pope_gemini_direct.jsonl",
        provider="gemini_local",
        limit=900,
        max_tokens=512,
        dataset="pope",
        model="gemini",
        prompt="direct",
    ),
    InferenceGroup(
        experiment="one_tenth",
        run_id="one_tenth_pope_gemini_cot_v1",
        dataset_path="data/processed/pope_eval.jsonl",
        prompt_path="prompts/answer/evidence_grounded_cot_pope.txt",
        output_path="outputs/model_responses/one_tenth_pope_gemini_cot.jsonl",
        provider="gemini_local",
        limit=900,
        max_tokens=512,
        dataset="pope",
        model="gemini",
        prompt="cot",
    ),
    InferenceGroup(
        experiment="one_tenth",
        run_id="one_tenth_pope_qwen_direct_v1",
        dataset_path="data/processed/pope_eval.jsonl",
        prompt_path="prompts/answer/direct_pope.txt",
        output_path="outputs/model_responses/one_tenth_pope_qwen_direct.jsonl",
        provider="qwen",
        limit=900,
        max_tokens=256,
        dataset="pope",
        model="qwen",
        prompt="direct",
    ),
    InferenceGroup(
        experiment="one_tenth",
        run_id="one_tenth_pope_qwen_cot_v1",
        dataset_path="data/processed/pope_eval.jsonl",
        prompt_path="prompts/answer/evidence_grounded_cot_pope.txt",
        output_path="outputs/model_responses/one_tenth_pope_qwen_cot.jsonl",
        provider="qwen",
        limit=900,
        max_tokens=512,
        dataset="pope",
        model="qwen",
        prompt="cot",
    ),
    InferenceGroup(
        experiment="one_tenth",
        run_id="one_tenth_mathvista_gemini_direct_v1",
        dataset_path="data/processed/mathvista_eval.jsonl",
        prompt_path="prompts/answer/direct_mathvista.txt",
        output_path="outputs/model_responses/one_tenth_mathvista_gemini_direct.jsonl",
        provider="gemini_local",
        limit=514,
        max_tokens=256,
        dataset="mathvista",
        model="gemini",
        prompt="direct",
    ),
    InferenceGroup(
        experiment="one_tenth",
        run_id="one_tenth_mathvista_gemini_cot_v1",
        dataset_path="data/processed/mathvista_eval.jsonl",
        prompt_path="prompts/answer/evidence_grounded_cot_mathvista.txt",
        output_path="outputs/model_responses/one_tenth_mathvista_gemini_cot.jsonl",
        provider="gemini_local",
        limit=514,
        max_tokens=512,
        dataset="mathvista",
        model="gemini",
        prompt="cot",
    ),
    InferenceGroup(
        experiment="one_tenth",
        run_id="one_tenth_mathvista_qwen_direct_v1",
        dataset_path="data/processed/mathvista_eval.jsonl",
        prompt_path="prompts/answer/direct_mathvista.txt",
        output_path="outputs/model_responses/one_tenth_mathvista_qwen_direct.jsonl",
        provider="qwen",
        limit=514,
        max_tokens=256,
        dataset="mathvista",
        model="qwen",
        prompt="direct",
    ),
    InferenceGroup(
        experiment="one_tenth",
        run_id="one_tenth_mathvista_qwen_cot_v1",
        dataset_path="data/processed/mathvista_eval.jsonl",
        prompt_path="prompts/answer/evidence_grounded_cot_mathvista.txt",
        output_path="outputs/model_responses/one_tenth_mathvista_qwen_cot.jsonl",
        provider="qwen",
        limit=514,
        max_tokens=512,
        dataset="mathvista",
        model="qwen",
        prompt="cot",
    ),
)

XLRS_PILOT_INFERENCE_GROUPS: tuple[InferenceGroup, ...] = (
    InferenceGroup(
        experiment="xlrs_pilot",
        run_id="xlrs_pilot_xlrs_bench_gpt54_direct_v1",
        dataset_path="data/processed/xlrs_eval.jsonl",
        prompt_path="prompts/answer/direct_xlrs.txt",
        output_path="outputs/model_responses/xlrs_pilot_xlrs_bench_gpt54_direct.jsonl",
        provider="gpt54_local",
        limit=100,
        max_tokens=512,
        dataset="xlrs_bench",
        model="gpt54",
        prompt="direct",
    ),
    InferenceGroup(
        experiment="xlrs_pilot",
        run_id="xlrs_pilot_xlrs_bench_gpt54_cot_v1",
        dataset_path="data/processed/xlrs_eval.jsonl",
        prompt_path="prompts/answer/evidence_grounded_cot_xlrs.txt",
        output_path="outputs/model_responses/xlrs_pilot_xlrs_bench_gpt54_cot.jsonl",
        provider="gpt54_local",
        limit=100,
        max_tokens=512,
        dataset="xlrs_bench",
        model="gpt54",
        prompt="cot",
    ),
    InferenceGroup(
        experiment="xlrs_pilot",
        run_id="xlrs_pilot_xlrs_bench_qwen_direct_v1",
        dataset_path="data/processed/xlrs_eval.jsonl",
        prompt_path="prompts/answer/direct_xlrs.txt",
        output_path="outputs/model_responses/xlrs_pilot_xlrs_bench_qwen_direct.jsonl",
        provider="qwen",
        limit=100,
        max_tokens=512,
        dataset="xlrs_bench",
        model="qwen",
        prompt="direct",
    ),
    InferenceGroup(
        experiment="xlrs_pilot",
        run_id="xlrs_pilot_xlrs_bench_qwen_cot_v1",
        dataset_path="data/processed/xlrs_eval.jsonl",
        prompt_path="prompts/answer/evidence_grounded_cot_xlrs.txt",
        output_path="outputs/model_responses/xlrs_pilot_xlrs_bench_qwen_cot.jsonl",
        provider="qwen",
        limit=100,
        max_tokens=512,
        dataset="xlrs_bench",
        model="qwen",
        prompt="cot",
    ),
)


XLRS_SR_VARIANTS: tuple[str, ...] = ("original", "sr", "paired")
XLRS_SR_MODELS: tuple[ModelName, ...] = ("gpt54", "qwen")
XLRS_SR_PROMPTS: tuple[PromptName, ...] = ("direct", "cot")

XLRS_SR_INFERENCE_GROUPS: tuple[InferenceGroup, ...] = tuple(
    InferenceGroup(
        experiment="xlrs_sr",
        run_id=f"xlrs_sr_{variant}_xlrs_bench_{model}_{prompt}_v1",
        dataset_path=f"data/processed/xlrs_eval_{variant}.jsonl",
        prompt_path=(
            "prompts/answer/direct_xlrs_sr.txt"
            if prompt == "direct"
            else "prompts/answer/evidence_grounded_cot_xlrs_sr.txt"
        ),
        output_path=(
            f"outputs/model_responses/xlrs_sr_{variant}_xlrs_bench_"
            f"{model}_{prompt}.jsonl"
        ),
        provider="gpt54_local" if model == "gpt54" else "qwen",
        limit=100,
        max_tokens=512,
        dataset="xlrs_bench",
        model=model,
        prompt=prompt,
    )
    for variant in XLRS_SR_VARIANTS
    for model in XLRS_SR_MODELS
    for prompt in XLRS_SR_PROMPTS
)

INFERENCE_GROUPS: tuple[InferenceGroup, ...] = (
    ONE_TENTH_INFERENCE_GROUPS + XLRS_PILOT_INFERENCE_GROUPS + XLRS_SR_INFERENCE_GROUPS
)


def _detector_output_path(group: InferenceGroup) -> str:
    return f"outputs/detector_results/{_run_stem(group.run_id)}_zero_shot_v2.jsonl"


def _run_stem(run_id: str) -> str:
    if run_id.endswith("_v1"):
        return run_id[:-3]
    return run_id


DETECTOR_GROUPS: tuple[DetectorGroup, ...] = tuple(
    DetectorGroup(
        experiment=group.experiment,
        run_id=f"{group.run_id.replace('_v1', '')}_zero_shot_gpt54_v2",
        samples_path=group.dataset_path,
        responses_path=group.output_path,
        output_path=_detector_output_path(group),
        provider="gpt54_local",
        detector="zero_shot",
        dataset=group.dataset,
        model=group.model,
        prompt=group.prompt,
        version="v2",
        response_version=group.version,
        limit=group.limit,
        offset=group.offset,
    )
    for group in INFERENCE_GROUPS
)


def select_inference_groups(
    *,
    experiments: set[str] | None = None,
    datasets: set[str] | None = None,
    models: set[str] | None = None,
    prompts: set[str] | None = None,
    versions: set[str] | None = None,
) -> list[InferenceGroup]:
    """按实验、数据集、模型、prompt 和版本筛选 inference 组。"""
    return [
        group
        for group in INFERENCE_GROUPS
        if _matches(group.experiment, experiments)
        and _matches(group.dataset, datasets)
        and _matches(group.model, models)
        and _matches(group.prompt, prompts)
        and _matches(group.version, versions)
    ]


def select_detector_groups(
    *,
    experiments: set[str] | None = None,
    datasets: set[str] | None = None,
    models: set[str] | None = None,
    prompts: set[str] | None = None,
    detectors: set[str] | None = None,
    versions: set[str] | None = None,
    responses_from: str | None = None,
) -> list[DetectorGroup]:
    """按过滤条件筛选 detector 组，并可复用其他实验的 response 文件。"""
    groups = [
        group
        for group in DETECTOR_GROUPS
        if _matches(group.experiment, experiments)
        and _matches(group.dataset, datasets)
        and _matches(group.model, models)
        and _matches(group.prompt, prompts)
        and _matches(group.detector, detectors)
        and _matches(group.version, versions)
    ]
    if responses_from is None:
        return groups

    source_groups = select_inference_groups(
        experiments={responses_from}, datasets=datasets, models=models, prompts=prompts
    )
    source_by_key = {_response_reuse_key(group): group for group in source_groups}
    reused = []
    for group in groups:
        matching_inference = _matching_inference_group(group)
        key = _response_reuse_key(matching_inference)
        source = source_by_key.get(key)
        if source is None:
            raise ValueError(
                f"No reusable response group from experiment {responses_from!r} for {group.run_id}."
            )
        reused.append(replace(group, responses_path=source.output_path))
    return reused


def _matching_inference_group(detector_group: DetectorGroup) -> InferenceGroup:
    for group in INFERENCE_GROUPS:
        if (
            group.experiment == detector_group.experiment
            and group.dataset == detector_group.dataset
            and group.model == detector_group.model
            and group.prompt == detector_group.prompt
            and group.limit == detector_group.limit
            and group.offset == detector_group.offset
            and group.version == detector_group.response_version
        ):
            return group
    raise ValueError(
        f"No matching inference group for detector group {detector_group.run_id}."
    )


def _response_reuse_key(group: InferenceGroup) -> tuple[object, ...]:
    return (
        group.dataset,
        group.model,
        group.prompt,
        group.dataset_path,
        group.prompt_path,
        group.provider,
        group.limit,
        group.offset,
        group.max_tokens,
        group.version,
    )


def _matches(value: str, allowed: set[str] | None) -> bool:
    return allowed is None or value in allowed

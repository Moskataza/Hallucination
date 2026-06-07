from __future__ import annotations

import argparse
from pathlib import Path

from src.datasets.jsonl import read_jsonl, write_jsonl
from src.datasets.schema import DetectorResult, EvalSample, ModelResponse, TaxonomyLabel
from src.models.response_parser import normalize_yes_no


def detect_pope_hallucination(
    sample: EvalSample,
    response: ModelResponse,
    run_id: str = "pope_rule_based_v1",
) -> DetectorResult:
    reference = normalize_yes_no(sample.reference_answer)
    prediction = normalize_yes_no(response.parsed.final_answer or response.raw_response)

    if reference == "no" and prediction == "yes":
        return DetectorResult(
            run_id=run_id,
            sample_id=sample.sample_id,
            model_response_id=response.run_id,
            detector="pope_rule_based",
            answer_correct=False,
            is_hallucination=True,
            taxonomy=TaxonomyLabel(coarse="Factual", fine="OBJ"),
            unsupported_visual_claim=True,
            confidence="high",
            explanation="POPE negative object-existence question: the model answered yes although the reference answer is no.",
            dataset=sample.dataset,
            model=response.model,
            prompt_type=response.prompt_type,
        )

    answer_correct = None if reference == "unclear" or prediction == "unclear" else reference == prediction
    return DetectorResult(
        run_id=run_id,
        sample_id=sample.sample_id,
        model_response_id=response.run_id,
        detector="pope_rule_based",
        answer_correct=answer_correct,
        is_hallucination=False if answer_correct is not None else None,
        taxonomy=TaxonomyLabel(
            coarse="None" if answer_correct is not None else "Unclear",
            fine="None" if answer_correct is not None else "Unclear",
        ),
        unsupported_visual_claim=False if answer_correct is not None else None,
        confidence="high" if answer_correct is not None else "low",
        explanation="Rule did not detect a POPE object hallucination.",
        dataset=sample.dataset,
        model=response.model,
        prompt_type=response.prompt_type,
    )


def detect_file(
    samples_path: str | Path,
    responses_path: str | Path,
    output_path: str | Path,
    allow_missing: bool = False,
) -> None:
    samples = {row["sample_id"]: EvalSample.from_dict(row) for row in read_jsonl(samples_path)}
    results = []
    missing_sample_ids = []
    for row in read_jsonl(responses_path):
        response = ModelResponse.from_dict(row)
        sample = samples.get(response.sample_id)
        if not sample:
            missing_sample_ids.append(response.sample_id)
            continue
        results.append(detect_pope_hallucination(sample, response).to_dict())

    if missing_sample_ids and not allow_missing:
        preview = ", ".join(missing_sample_ids[:5])
        raise ValueError(f"{len(missing_sample_ids)} responses did not match samples: {preview}")

    write_jsonl(output_path, results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run POPE rule-based hallucination detection.")
    parser.add_argument("samples_path")
    parser.add_argument("responses_path")
    parser.add_argument("output_path")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    detect_file(args.samples_path, args.responses_path, args.output_path, args.allow_missing)


if __name__ == "__main__":
    main()

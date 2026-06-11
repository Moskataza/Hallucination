from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.datasets.jsonl import write_jsonl
from src.pipelines import experiment_groups, resume_groups, run_stable_pipeline
from src.pipelines.experiment_groups import (
    DetectorGroup,
    InferenceGroup,
    select_detector_groups,
    select_inference_groups,
)
from src.pipelines.result_store import inspect_detector_group, inspect_inference_group
from src.pipelines.resume_groups import resume_detector_groups, resume_inference_groups


def test_select_inference_groups_filters_registered_experiment() -> None:
    groups = select_inference_groups(
        experiments={"one_tenth"}, datasets={"mathvista"}, models={"gemini"}
    )

    assert {group.run_id for group in groups} == {
        "one_tenth_mathvista_gemini_direct_v1",
        "one_tenth_mathvista_gemini_cot_v1",
    }


def test_final_validation_reports_only_completed_stage(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    class Status:
        valid = 1
        missing = 0
        invalid = 0
        duplicates = 0
        complete = True
        missing_examples = ()
        invalid_examples = ()

    class PipelineStatus:
        status = Status()

    samples = tmp_path / "samples.jsonl"
    responses = tmp_path / "responses.jsonl"
    write_jsonl(samples, [_sample_row("sample_0")])
    write_jsonl(responses, [_response_row(sample_id="sample_0")])
    detector_calls = []

    inference_group = InferenceGroup(
        experiment="tmp",
        run_id="response_run",
        dataset_path=str(samples),
        prompt_path="prompts/answer/direct_mathvista.txt",
        output_path=str(responses),
        provider="gemini_local",
        limit=1,
        max_tokens=256,
        dataset="mathvista",
        model="gemini",
        prompt="direct",
    )
    detector_group = DetectorGroup(
        experiment="tmp",
        run_id="detector_run",
        samples_path=str(samples),
        responses_path=str(responses),
        output_path="judge.jsonl",
        provider="gpt54_local",
        detector="zero_shot",
        dataset="mathvista",
        model="gemini",
        prompt="direct",
        limit=1,
    )
    monkeypatch.setattr(
        run_stable_pipeline, "inspect_inference_group", lambda group: PipelineStatus()
    )

    def fake_inspect_detector_group(group: DetectorGroup) -> PipelineStatus:
        detector_calls.append(group.run_id)
        return PipelineStatus()

    monkeypatch.setattr(
        run_stable_pipeline, "inspect_detector_group", fake_inspect_detector_group
    )

    run_stable_pipeline._print_final_validation(
        "detectors", [inference_group], [detector_group]
    )

    output = capsys.readouterr().out
    assert "FINAL_VALIDATE" in output
    assert "DETECTORS detector_run" in output
    assert "RESPONSES response_run" not in output
    assert detector_calls == ["detector_run"]


def test_select_detector_groups_can_reuse_response_paths_from_source_experiment() -> (
    None
):
    groups = select_detector_groups(
        experiments={"one_tenth"},
        datasets={"mathvista"},
        models={"gemini"},
        prompts={"direct"},
        responses_from="one_tenth",
    )

    assert len(groups) == 1
    assert (
        groups[0].responses_path
        == "outputs/model_responses/one_tenth_mathvista_gemini_direct.jsonl"
    )
    assert (
        groups[0].output_path
        == "outputs/detector_results/one_tenth_mathvista_gemini_direct_zero_shot_v2.jsonl"
    )
    assert groups[0].version == "v2"
    assert groups[0].response_version == "v1"


def test_select_detector_groups_refuses_incompatible_response_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = InferenceGroup(
        experiment="target",
        run_id="target_mathvista_gemini_direct_v1",
        dataset_path="data/processed/mathvista_eval.jsonl",
        prompt_path="prompts/answer/direct_mathvista.txt",
        output_path="outputs/model_responses/target.jsonl",
        provider="gemini_local",
        limit=514,
        max_tokens=512,
        dataset="mathvista",
        model="gemini",
        prompt="direct",
    )
    source = InferenceGroup(
        experiment="source",
        run_id="source_mathvista_gemini_direct_v1",
        dataset_path="data/processed/mathvista_eval.jsonl",
        prompt_path="prompts/answer/direct_mathvista.txt",
        output_path="outputs/model_responses/source.jsonl",
        provider="gemini_local",
        limit=514,
        max_tokens=256,
        dataset="mathvista",
        model="gemini",
        prompt="direct",
    )
    monkeypatch.setattr(experiment_groups, "INFERENCE_GROUPS", (target, source))
    monkeypatch.setattr(
        experiment_groups,
        "DETECTOR_GROUPS",
        (
            DetectorGroup(
                experiment="target",
                run_id="target_judge",
                samples_path=target.dataset_path,
                responses_path=target.output_path,
                output_path="outputs/detector_results/target.jsonl",
                provider="gpt54_local",
                detector="zero_shot",
                dataset="mathvista",
                model="gemini",
                prompt="direct",
                limit=514,
            ),
        ),
    )

    with pytest.raises(ValueError, match="No reusable response group"):
        select_detector_groups(experiments={"target"}, responses_from="source")


def test_inspect_inference_group_reports_missing_and_invalid_rows(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "samples.jsonl"
    output = tmp_path / "responses.jsonl"
    write_jsonl(
        dataset,
        [
            {"sample_id": "sample_0"},
            {"sample_id": "sample_1"},
            {"sample_id": "sample_2"},
        ],
    )
    write_jsonl(
        output,
        [
            _response_row(sample_id="sample_0", run_id="run_1"),
            {
                "sample_id": "sample_1",
                "raw_response": "",
                "parsed": {"parse_status": "failed", "final_answer": ""},
                "inference_metadata": {"status": "failed"},
            },
        ],
    )
    group = InferenceGroup(
        run_id="run_1",
        dataset_path=str(dataset),
        prompt_path="prompt.txt",
        output_path=str(output),
        provider="gemini_local",
        limit=3,
        max_tokens=256,
        dataset="mathvista",
        model="gemini",
        prompt="direct",
        experiment="tmp",
    )

    status = inspect_inference_group(group).status

    assert status.valid == 1
    assert status.invalid == 1
    assert status.missing == 1
    assert status.complete is False
    assert status.invalid_examples == ("sample_1",)
    assert status.missing_examples == ("sample_2",)


def test_inspect_inference_group_rejects_incompatible_response_rows(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "samples.jsonl"
    output = tmp_path / "responses.jsonl"
    write_jsonl(dataset, [{"sample_id": "sample_0"}])
    write_jsonl(
        output,
        [
            _response_row(
                sample_id="sample_0",
                run_id="other_run",
                dataset="mathvista",
                model="gemini-2.5-flash",
                model_type="closed",
                prompt_type="direct",
                provider="gemini_local",
                max_tokens=256,
            )
        ],
    )
    group = InferenceGroup(
        run_id="run_1",
        dataset_path=str(dataset),
        prompt_path="prompt.txt",
        output_path=str(output),
        provider="gemini_local",
        limit=1,
        max_tokens=256,
        dataset="mathvista",
        model="gemini",
        prompt="direct",
        experiment="tmp",
    )

    status = inspect_inference_group(group).status

    assert status.valid == 0
    assert status.invalid == 1
    assert status.invalid_examples == ("sample_0",)


def test_inspect_detector_group_rejects_incompatible_detector_rows(
    tmp_path: Path,
) -> None:
    samples = tmp_path / "samples.jsonl"
    responses = tmp_path / "responses.jsonl"
    output = tmp_path / "judge.jsonl"
    write_jsonl(samples, [_sample_row("sample_0")])
    write_jsonl(responses, [_response_row(sample_id="sample_0")])
    write_jsonl(
        output,
        [
            _detector_row(
                run_id="other_judge_run",
                sample_id="sample_0",
                model_response_id="model_run:sample_0",
            )
        ],
    )
    group = DetectorGroup(
        run_id="judge_run",
        samples_path=str(samples),
        responses_path=str(responses),
        output_path=str(output),
        provider="gpt54_local",
        detector="zero_shot",
        dataset="mathvista",
        model="gemini",
        prompt="direct",
        experiment="tmp",
        limit=1,
    )

    status = inspect_detector_group(group).status

    assert status.valid == 0
    assert status.invalid == 1
    assert status.invalid_examples == ("model_run:sample_0",)


def test_inspect_detector_group_rejects_mismatched_sample_identity(
    tmp_path: Path,
) -> None:
    samples = tmp_path / "samples.jsonl"
    responses = tmp_path / "responses.jsonl"
    output = tmp_path / "judge.jsonl"
    write_jsonl(samples, [_sample_row("sample_0")])
    write_jsonl(responses, [_response_row(sample_id="sample_0")])
    write_jsonl(
        output,
        [
            _detector_row(
                run_id="judge_run",
                sample_id="sample_1",
                model_response_id="model_run:sample_0",
            )
        ],
    )
    group = DetectorGroup(
        run_id="judge_run",
        samples_path=str(samples),
        responses_path=str(responses),
        output_path=str(output),
        provider="gpt54_local",
        detector="zero_shot",
        dataset="mathvista",
        model="gemini",
        prompt="direct",
        experiment="tmp",
        limit=1,
    )

    status = inspect_detector_group(group).status

    assert status.valid == 0
    assert status.invalid == 1
    assert status.invalid_examples == ("model_run:sample_0",)


def test_inspect_detector_group_rejects_sample_not_in_samples(tmp_path: Path) -> None:
    samples = tmp_path / "samples.jsonl"
    responses = tmp_path / "responses.jsonl"
    output = tmp_path / "judge.jsonl"
    write_jsonl(samples, [_sample_row("sample_0")])
    write_jsonl(responses, [_response_row(sample_id="sample_1")])
    write_jsonl(
        output,
        [
            _detector_row(
                run_id="judge_run",
                sample_id="sample_1",
                model_response_id="model_run:sample_1",
            )
        ],
    )
    group = DetectorGroup(
        run_id="judge_run",
        samples_path=str(samples),
        responses_path=str(responses),
        output_path=str(output),
        provider="gpt54_local",
        detector="zero_shot",
        dataset="mathvista",
        model="gemini",
        prompt="direct",
        experiment="tmp",
        limit=1,
    )

    status = inspect_detector_group(group).status

    assert status.valid == 0
    assert status.invalid == 1
    assert status.invalid_examples == ("model_run:sample_1",)


def test_inspect_detector_group_rejects_missing_judge_provider(tmp_path: Path) -> None:
    samples = tmp_path / "samples.jsonl"
    responses = tmp_path / "responses.jsonl"
    output = tmp_path / "judge.jsonl"
    write_jsonl(samples, [_sample_row("sample_0")])
    write_jsonl(responses, [_response_row(sample_id="sample_0")])
    row = _detector_row(
        run_id="judge_run",
        sample_id="sample_0",
        model_response_id="model_run:sample_0",
    )
    row["details"] = {}
    write_jsonl(output, [row])
    group = DetectorGroup(
        run_id="judge_run",
        samples_path=str(samples),
        responses_path=str(responses),
        output_path=str(output),
        provider="gpt54_local",
        detector="zero_shot",
        dataset="mathvista",
        model="gemini",
        prompt="direct",
        experiment="tmp",
        limit=1,
    )

    status = inspect_detector_group(group).status

    assert status.valid == 0
    assert status.invalid == 1
    assert status.invalid_examples == ("model_run:sample_0",)


def test_detector_resume_refuses_partial_model_responses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    samples = tmp_path / "samples.jsonl"
    responses = tmp_path / "responses.jsonl"
    output = tmp_path / "judge.jsonl"
    write_jsonl(samples, [_sample_row("sample_0"), _sample_row("sample_1")])
    write_jsonl(responses, [_response_row(sample_id="sample_0")])
    group = DetectorGroup(
        run_id="judge_run",
        samples_path=str(samples),
        responses_path=str(responses),
        output_path=str(output),
        provider="gpt54_local",
        detector="zero_shot",
        dataset="mathvista",
        model="gemini",
        prompt="direct",
        experiment="tmp",
        limit=2,
    )
    monkeypatch.setattr(
        resume_groups,
        "INFERENCE_GROUPS",
        (
            InferenceGroup(
                run_id="model_run",
                dataset_path=str(samples),
                prompt_path="prompts/answer/direct_mathvista.txt",
                output_path=str(responses),
                provider="gemini_local",
                limit=2,
                max_tokens=256,
                dataset="mathvista",
                model="gemini",
                prompt="direct",
                experiment="tmp",
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="responses are not complete"):
        resume_detector_groups([group], chunk_size=1)


def test_detector_resume_refuses_invalid_model_responses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    samples = tmp_path / "samples.jsonl"
    responses = tmp_path / "responses.jsonl"
    output = tmp_path / "judge.jsonl"
    write_jsonl(samples, [_sample_row("sample_0")])
    write_jsonl(
        responses,
        [
            {
                "run_id": "model_run",
                "sample_id": "sample_0",
                "dataset": "mathvista",
                "model": "gemini",
                "model_type": "closed",
                "prompt_type": "direct",
                "prompt_version": "v1",
                "raw_response": "",
                "parsed": {"parse_status": "failed", "final_answer": ""},
                "inference_metadata": {"status": "failed"},
            }
        ],
    )
    group = DetectorGroup(
        run_id="judge_run",
        samples_path=str(samples),
        responses_path=str(responses),
        output_path=str(output),
        provider="gpt54_local",
        detector="zero_shot",
        dataset="mathvista",
        model="gemini",
        prompt="direct",
        experiment="tmp",
        limit=1,
    )
    _register_response_source(monkeypatch, responses=responses, samples=samples)

    with pytest.raises(RuntimeError, match="responses are not ready"):
        resume_detector_groups([group], chunk_size=1)


def test_detector_resume_refuses_incompatible_registered_responses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    samples = tmp_path / "samples.jsonl"
    responses = tmp_path / "responses.jsonl"
    output = tmp_path / "judge.jsonl"
    write_jsonl(samples, [_sample_row("sample_0")])
    write_jsonl(responses, [_response_row(sample_id="sample_0", max_tokens=512)])
    group = DetectorGroup(
        run_id="judge_run",
        samples_path=str(samples),
        responses_path=str(responses),
        output_path=str(output),
        provider="gpt54_local",
        detector="zero_shot",
        dataset="mathvista",
        model="gemini",
        prompt="direct",
        experiment="tmp",
        limit=1,
    )
    _register_response_source(monkeypatch, responses=responses, samples=samples)

    with pytest.raises(RuntimeError, match="incompatible"):
        resume_detector_groups([group], chunk_size=1)


def test_detector_resume_accepts_qwen_provider_model_variants(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    samples = tmp_path / "samples.jsonl"
    responses = tmp_path / "responses.jsonl"
    output = tmp_path / "judge.jsonl"
    write_jsonl(samples, [_sample_row("sample_0")])
    write_jsonl(
        responses,
        [
            _response_row(
                sample_id="sample_0",
                run_id="one_tenth_mathvista_qwen_direct_v1",
                model="qwen/qwen3-vl-8b-instruct",
                model_type="open",
                provider="openrouter_qwen3_vl_instruct",
                max_tokens=1024,
            )
        ],
    )
    group = DetectorGroup(
        run_id="judge_run",
        samples_path=str(samples),
        responses_path=str(responses),
        output_path=str(output),
        provider="gpt54_local",
        detector="zero_shot",
        dataset="mathvista",
        model="qwen",
        prompt="direct",
        experiment="tmp",
        limit=1,
    )
    monkeypatch.setattr(
        resume_groups,
        "INFERENCE_GROUPS",
        (
            InferenceGroup(
                run_id="one_tenth_mathvista_qwen_direct_v1",
                dataset_path=str(samples),
                prompt_path="prompts/answer/direct_mathvista.txt",
                output_path=str(responses),
                provider="qwen",
                limit=1,
                max_tokens=256,
                dataset="mathvista",
                model="qwen",
                prompt="direct",
                experiment="tmp",
            ),
        ),
    )
    calls = []

    def fake_detect_file(**kwargs: Any) -> None:
        calls.append(kwargs)
        row = _detector_row(
            run_id="judge_run",
            sample_id="sample_0",
            model_response_id="one_tenth_mathvista_qwen_direct_v1:sample_0",
        )
        row["model"] = "qwen/qwen3-vl-8b-instruct"
        write_jsonl(output, [row])

    monkeypatch.setattr(resume_groups, "detect_file", fake_detect_file)

    resume_detector_groups([group], chunk_size=1)

    assert [call["run_id"] for call in calls] == ["judge_run"]


def test_inference_resume_refuses_locked_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    samples = tmp_path / "samples.jsonl"
    responses = tmp_path / "responses.jsonl"
    lock = responses.with_suffix(responses.suffix + ".lock")
    write_jsonl(samples, [_sample_row("sample_0")])
    lock.write_text("other-process", encoding="utf-8")
    group = InferenceGroup(
        run_id="model_run",
        dataset_path=str(samples),
        prompt_path="prompts/answer/direct_mathvista.txt",
        output_path=str(responses),
        provider="gemini_local",
        limit=1,
        max_tokens=256,
        dataset="mathvista",
        model="gemini",
        prompt="direct",
        experiment="tmp",
    )
    calls = []

    def fake_run_inference(**kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(resume_groups, "run_inference", fake_run_inference)

    with pytest.raises(RuntimeError, match="Response output is already locked"):
        resume_inference_groups([group], chunk_size=1)

    assert calls == []


def test_inference_resume_cleans_duplicate_output_without_new_valid_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    samples = tmp_path / "samples.jsonl"
    responses = tmp_path / "responses.jsonl"
    write_jsonl(samples, [_sample_row("sample_0")])
    write_jsonl(
        responses,
        [_response_row(sample_id="sample_0"), _response_row(sample_id="sample_0")],
    )
    group = InferenceGroup(
        run_id="model_run",
        dataset_path=str(samples),
        prompt_path="prompts/answer/direct_mathvista.txt",
        output_path=str(responses),
        provider="gemini_local",
        limit=1,
        max_tokens=256,
        dataset="mathvista",
        model="gemini",
        prompt="direct",
        experiment="tmp",
    )
    calls = []

    def fake_run_inference(**kwargs: Any) -> None:
        calls.append(kwargs)
        write_jsonl(kwargs["output_path"], [_response_row(sample_id="sample_0")])

    monkeypatch.setattr(resume_groups, "run_inference", fake_run_inference)

    resume_inference_groups([group], chunk_size=1, max_chunk_attempts=1)

    assert len(calls) == 1
    assert calls[0]["resume"] is True
    assert inspect_inference_group(group).status.complete is True


def test_detector_resume_refuses_locked_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    samples = tmp_path / "samples.jsonl"
    responses = tmp_path / "responses.jsonl"
    output = tmp_path / "judge.jsonl"
    lock = output.with_suffix(output.suffix + ".lock")
    write_jsonl(samples, [_sample_row("sample_0")])
    write_jsonl(responses, [_response_row(sample_id="sample_0")])
    lock.write_text("other-process", encoding="utf-8")
    group = DetectorGroup(
        run_id="judge_run",
        samples_path=str(samples),
        responses_path=str(responses),
        output_path=str(output),
        provider="gpt54_local",
        detector="zero_shot",
        dataset="mathvista",
        model="gemini",
        prompt="direct",
        experiment="tmp",
        limit=1,
    )
    calls = []

    def fake_detect_file(**kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(resume_groups, "detect_file", fake_detect_file)
    _register_response_source(monkeypatch, responses=responses, samples=samples)

    with pytest.raises(RuntimeError, match="already locked"):
        resume_detector_groups([group], chunk_size=1)

    assert calls == []


def test_inference_resume_removes_lock_after_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    samples = tmp_path / "samples.jsonl"
    responses = tmp_path / "responses.jsonl"
    lock = responses.with_suffix(responses.suffix + ".lock")
    write_jsonl(samples, [_sample_row("sample_0")])
    group = InferenceGroup(
        run_id="model_run",
        dataset_path=str(samples),
        prompt_path="prompts/answer/direct_mathvista.txt",
        output_path=str(responses),
        provider="gemini_local",
        limit=1,
        max_tokens=256,
        dataset="mathvista",
        model="gemini",
        prompt="direct",
        experiment="tmp",
    )

    def fake_run_inference(**kwargs: Any) -> None:
        write_jsonl(kwargs["output_path"], [_response_row(sample_id="sample_0")])

    monkeypatch.setattr(resume_groups, "run_inference", fake_run_inference)

    resume_inference_groups([group], chunk_size=1)

    assert not lock.exists()


def test_inference_resume_removes_lock_after_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    samples = tmp_path / "samples.jsonl"
    responses = tmp_path / "responses.jsonl"
    lock = responses.with_suffix(responses.suffix + ".lock")
    write_jsonl(samples, [_sample_row("sample_0")])
    group = InferenceGroup(
        run_id="model_run",
        dataset_path=str(samples),
        prompt_path="prompts/answer/direct_mathvista.txt",
        output_path=str(responses),
        provider="gemini_local",
        limit=1,
        max_tokens=256,
        dataset="mathvista",
        model="gemini",
        prompt="direct",
        experiment="tmp",
    )

    def fake_run_inference(**kwargs: Any) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(resume_groups, "run_inference", fake_run_inference)

    with pytest.raises(RuntimeError, match="Resume stalled"):
        resume_inference_groups([group], chunk_size=1, max_chunk_attempts=1)

    assert not lock.exists()


def test_detector_resume_removes_lock_after_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    samples = tmp_path / "samples.jsonl"
    responses = tmp_path / "responses.jsonl"
    output = tmp_path / "judge.jsonl"
    lock = output.with_suffix(output.suffix + ".lock")
    write_jsonl(samples, [_sample_row("sample_0")])
    write_jsonl(responses, [_response_row(sample_id="sample_0")])
    group = DetectorGroup(
        run_id="judge_run",
        samples_path=str(samples),
        responses_path=str(responses),
        output_path=str(output),
        provider="gpt54_local",
        detector="zero_shot",
        dataset="mathvista",
        model="gemini",
        prompt="direct",
        experiment="tmp",
        limit=1,
    )

    def fake_detect_file(**kwargs: Any) -> None:
        write_jsonl(
            kwargs["output_path"],
            [
                _detector_row(
                    run_id=kwargs["run_id"],
                    sample_id="sample_0",
                    model_response_id="model_run:sample_0",
                )
            ],
        )

    monkeypatch.setattr(resume_groups, "detect_file", fake_detect_file)
    _register_response_source(monkeypatch, responses=responses, samples=samples)

    resume_detector_groups([group], chunk_size=1)

    assert not lock.exists()


def test_detector_resume_removes_lock_after_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    samples = tmp_path / "samples.jsonl"
    responses = tmp_path / "responses.jsonl"
    output = tmp_path / "judge_failure.jsonl"
    lock = output.with_suffix(output.suffix + ".lock")
    write_jsonl(samples, [_sample_row("sample_0")])
    write_jsonl(responses, [_response_row(sample_id="sample_0")])
    group = DetectorGroup(
        run_id="judge_run",
        samples_path=str(samples),
        responses_path=str(responses),
        output_path=str(output),
        provider="gpt54_local",
        detector="zero_shot",
        dataset="mathvista",
        model="gemini",
        prompt="direct",
        experiment="tmp",
        limit=1,
    )

    def fake_detect_file(**kwargs: Any) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(resume_groups, "detect_file", fake_detect_file)
    _register_response_source(monkeypatch, responses=responses, samples=samples)

    with pytest.raises(RuntimeError, match="Resume stalled"):
        resume_detector_groups([group], chunk_size=1, max_chunk_attempts=1)

    assert not lock.exists()


def test_detector_overwrite_applies_to_each_group_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    samples = tmp_path / "samples.jsonl"
    responses = tmp_path / "responses.jsonl"
    output_a = tmp_path / "judge_a.jsonl"
    output_b = tmp_path / "judge_b.jsonl"
    write_jsonl(samples, [_sample_row("sample_0")])
    write_jsonl(responses, [_response_row(sample_id="sample_0")])
    output_a.write_text("stale\n", encoding="utf-8")
    output_b.write_text("stale\n", encoding="utf-8")
    groups = [
        DetectorGroup(
            run_id="judge_a",
            samples_path=str(samples),
            responses_path=str(responses),
            output_path=str(output_a),
            provider="gpt54_local",
            detector="zero_shot",
            dataset="mathvista",
            model="gemini",
            prompt="direct",
            experiment="tmp",
            limit=1,
        ),
        DetectorGroup(
            run_id="judge_b",
            samples_path=str(samples),
            responses_path=str(responses),
            output_path=str(output_b),
            provider="gpt54_local",
            detector="zero_shot",
            dataset="mathvista",
            model="gemini",
            prompt="direct",
            experiment="tmp",
            limit=1,
        ),
    ]
    calls = []

    def fake_detect_file(**kwargs: Any) -> None:
        calls.append(kwargs)
        write_jsonl(
            kwargs["output_path"],
            [
                _detector_row(
                    run_id=kwargs["run_id"],
                    sample_id="sample_0",
                    model_response_id="model_run:sample_0",
                )
            ],
        )

    monkeypatch.setattr(resume_groups, "detect_file", fake_detect_file)
    _register_response_source(monkeypatch, responses=responses, samples=samples)

    resume_detector_groups(groups, chunk_size=1, overwrite=True)

    assert [(call["run_id"], call["overwrite"], call["resume"]) for call in calls] == [
        ("judge_a", True, False),
        ("judge_b", True, False),
    ]


def _register_response_source(
    monkeypatch: pytest.MonkeyPatch,
    *,
    responses: Path,
    samples: Path,
) -> None:
    monkeypatch.setattr(
        resume_groups,
        "INFERENCE_GROUPS",
        (
            InferenceGroup(
                run_id="model_run",
                dataset_path=str(samples),
                prompt_path="prompts/answer/direct_mathvista.txt",
                output_path=str(responses),
                provider="gemini_local",
                limit=1,
                max_tokens=256,
                dataset="mathvista",
                model="gemini",
                prompt="direct",
                experiment="tmp",
            ),
        ),
    )


def _sample_row(sample_id: str) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "dataset": "mathvista",
        "task_type": "visual_math_reasoning",
        "image_path": "data/raw/sample.jpg",
        "question": "Question?",
        "reference_answer": "UNAVAILABLE",
    }


def _response_row(
    *,
    sample_id: str,
    run_id: str = "model_run",
    dataset: str = "mathvista",
    model: str = "gemini-2.5-flash",
    model_type: str = "closed",
    prompt_type: str = "direct",
    provider: str = "gemini_local",
    max_tokens: int = 256,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "sample_id": sample_id,
        "dataset": dataset,
        "model": model,
        "model_type": model_type,
        "prompt_type": prompt_type,
        "prompt_version": "v1",
        "raw_response": "No",
        "parsed": {"parse_status": "ok", "final_answer": "No"},
        "inference_metadata": {
            "provider": provider,
            "max_tokens": max_tokens,
            "temperature": 0,
        },
    }


def _detector_row(
    *,
    run_id: str,
    sample_id: str,
    model_response_id: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "sample_id": sample_id,
        "model_response_id": model_response_id,
        "detector": "response_claim_zero_shot_judge",
        "answer_correct": None,
        "is_hallucination": False,
        "taxonomy": {"coarse": "None", "fine": "None"},
        "unsupported_visual_claim": False,
        "confidence": "high",
        "explanation": "ok",
        "raw_judge_response": "{}",
        "dataset": "mathvista",
        "model": "gemini-2.5-flash",
        "prompt_type": "direct",
        "details": {"judge_provider": "gpt54_local"},
    }

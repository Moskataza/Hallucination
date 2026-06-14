from pathlib import Path

import pytest

from src.datasets.convert_pope import convert_pope_record
from src.datasets.prepare_xlrs_sr import prepare_xlrs_sr_record
from src.datasets.schema import ModelResponse, ParsedResponse
from src.detectors.pope_rule_based import detect_pope_hallucination
from src.evaluation.agreement import cohens_kappa, matthews_corrcoef
from src.evaluation.hallucination_metrics import compute_hallucination_metrics
from src.evaluation.human_alignment import compute_binary_alignment
from src.models.response_parser import normalize_yes_no, parse_cot_response


def test_parse_cot_response_extracts_sections():
    parsed = parse_cot_response(
        "Visual Evidence: A dog is visible.\n"
        "Reasoning: The dog supports answering yes.\n"
        "Final Answer: Yes"
    )
    assert parsed.parse_status == "ok"
    assert parsed.visual_evidence == "A dog is visible."
    assert parsed.reasoning == "The dog supports answering yes."
    assert parsed.final_answer == "Yes"


def test_parse_cot_response_extracts_markdown_numbered_sections():
    parsed = parse_cot_response(
        "1.  **Visual Evidence**: A dog is visible.\n"
        "2.  **Reasoning:** The dog supports answering yes.\n"
        "3.  **Final Answer**: Yes"
    )

    assert parsed.parse_status == "ok"
    assert parsed.visual_evidence == "A dog is visible."
    assert parsed.reasoning == "The dog supports answering yes."
    assert parsed.final_answer == "Yes"


def test_parse_cot_response_extracts_bold_wrapped_numbered_sections():
    parsed = parse_cot_response(
        "**1. Visual Evidence:** A dog is visible.\n"
        "**2. Reasoning:** The dog supports answering yes.\n"
        "**3. Final Answer:** Yes"
    )

    assert parsed.parse_status == "ok"
    assert parsed.visual_evidence == "A dog is visible."
    assert parsed.reasoning == "The dog supports answering yes."
    assert parsed.final_answer == "Yes"


def test_parse_cot_response_extracts_markdown_heading_numbered_sections():
    parsed = parse_cot_response(
        "### 1. Visual Evidence:\n"
        "- A dog is visible.\n"
        "### 2. Reasoning:\n"
        "- The dog supports answering yes.\n"
        "### 3. Final Answer:\n"
        "Yes"
    )

    assert parsed.parse_status == "ok"
    assert parsed.visual_evidence == "- A dog is visible."
    assert parsed.reasoning == "- The dog supports answering yes."
    assert parsed.final_answer == "Yes"


def test_parse_cot_response_extracts_markdown_heading_bold_numbered_sections():
    parsed = parse_cot_response(
        "### **1. Visual Evidence:**\n"
        "- A dog is visible.\n"
        "### 2. **Reasoning:**\n"
        "- The dog supports answering yes.\n"
        "### **3. Final Answer:**\n"
        "Yes"
    )

    assert parsed.parse_status == "ok"
    assert parsed.visual_evidence == "- A dog is visible."
    assert parsed.reasoning == "- The dog supports answering yes."
    assert parsed.final_answer == "Yes"


def test_parse_xlrs_sr_cot_response_extracts_final_answer():
    parsed = parse_cot_response(
        "1. Visual Evidence:\n"
        "- [SR tile r1c2] A narrow road is visible.\n"
        "2. SR Consistency Check:\n"
        "- The road is also weakly visible in the original image.\n"
        "3. Reasoning:\n"
        "The paired evidence supports the answer.\n"
        "4. Final Answer:\n"
        "Yes"
    )

    assert parsed.parse_status == "ok"
    assert "[SR tile r1c2]" in parsed.visual_evidence
    assert parsed.reasoning == "The paired evidence supports the answer."
    assert parsed.final_answer == "Yes"


def test_prepare_xlrs_sr_record_selects_variant_image_and_metadata():
    record = {
        "id": "001",
        "question": "Is there a small vehicle near the road?",
        "answer": "yes",
        "image": "default/001.png",
        "original_image_path": "original/001.png",
        "sr_image_path": "sr/001.png",
        "paired_image_path": "paired/001.png",
        "sr_scale": 4,
        "sr_method": "example-sr",
        "tile_manifest_path": "tiles/001.json",
    }

    sample = prepare_xlrs_sr_record(record, variant="paired", image_root="data/xlrs")

    assert sample.sample_id == "xlrs_001_paired"
    assert Path(sample.image_path) == Path("data/xlrs/paired/001.png")
    assert Path(sample.metadata["original_image_path"]) == Path(
        "data/xlrs/original/001.png"
    )
    assert Path(sample.metadata["sr_image_path"]) == Path("data/xlrs/sr/001.png")
    assert Path(sample.metadata["paired_image_path"]) == Path(
        "data/xlrs/paired/001.png"
    )
    assert sample.metadata["sr_scale"] == 4
    assert sample.metadata["sr_method"] == "example-sr"
    assert sample.metadata["tile_manifest_path"] == "tiles/001.json"
    assert sample.metadata["xlrs_sr_variant"] == "paired"
    assert sample.metadata["evidence_protocol"] == "sr_aware_multiscale"


def test_prepare_xlrs_sr_record_requires_sr_specific_image():
    record = {
        "id": "001",
        "question": "Is there a small vehicle near the road?",
        "answer": "yes",
        "image": "default/001.png",
        "original_image_path": "original/001.png",
    }

    with pytest.raises(ValueError, match="SR variant requires"):
        prepare_xlrs_sr_record(record, variant="sr", image_root="data/xlrs")


def test_normalize_yes_no():
    assert normalize_yes_no("Final Answer: Yes") == "yes"
    assert normalize_yes_no("No, there is not.") == "no"
    assert normalize_yes_no("Cannot be determined from the image") == "unclear"


def test_pope_rule_detects_negative_yes_hallucination():
    sample = convert_pope_record(
        {
            "question_id": 1,
            "image": "x.jpg",
            "text": "Is there a dog in the image?",
            "label": "no",
        }
    )
    response = ModelResponse(
        run_id="response_1",
        sample_id=sample.sample_id,
        dataset="pope",
        model="gemini",
        model_type="closed",
        prompt_type="direct",
        prompt_version="v1",
        raw_response="Yes",
        parsed=ParsedResponse(final_answer="Yes", parse_status="ok"),
    )
    result = detect_pope_hallucination(sample, response)
    assert result.is_hallucination is True
    assert result.taxonomy.coarse == "Factual"
    assert result.taxonomy.fine == "OBJ"


def test_hallucination_metrics_and_grounded_accuracy():
    metrics = compute_hallucination_metrics(
        [
            {
                "answer_correct": True,
                "is_hallucination": False,
                "unsupported_visual_claim": False,
                "taxonomy": {"coarse": "None", "fine": "None"},
            },
            {
                "answer_correct": False,
                "is_hallucination": True,
                "unsupported_visual_claim": True,
                "taxonomy": {"coarse": "Factual", "fine": "OBJ"},
            },
        ]
    )
    assert metrics["count"] == 2
    assert metrics["hallucination_rate"] == 0.5
    assert metrics["grounded_accuracy"] == 0.5
    assert metrics["fine_type_counts"] == {"OBJ": 1}


def test_human_alignment_metrics():
    metrics = compute_binary_alignment(
        [
            {"predicted_is_hallucination": True, "human_is_hallucination": True},
            {"predicted_is_hallucination": True, "human_is_hallucination": False},
            {"predicted_is_hallucination": False, "human_is_hallucination": False},
            {"predicted_is_hallucination": False, "human_is_hallucination": True},
        ]
    )
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fn"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5


def test_agreement_helpers_return_values():
    assert cohens_kappa(tp=1, fp=0, tn=1, fn=0) == 1.0
    assert matthews_corrcoef(tp=1, fp=0, tn=1, fn=0) == 1.0

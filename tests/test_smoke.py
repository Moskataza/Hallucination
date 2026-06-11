from src.datasets.convert_pope import convert_pope_record
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

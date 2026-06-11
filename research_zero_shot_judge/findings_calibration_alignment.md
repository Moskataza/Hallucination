# Calibration and human-alignment methods for LLM-as-judge hallucination detectors

## Executive takeaways

1. Use fixed, human-annotated validation sets as the comparison anchor. Public resources such as RAGTruth, FActScore, SAFE/LongFact-style fact-level annotations, and domain-specific human annotation sets show the importance of stable labels for comparing detector prompts/models over time.
2. Report precision, recall, F1, and confusion matrices at a chosen operating point, not only average accuracy. Hallucination detectors often have asymmetric costs: false positives can block valid outputs, while false negatives let hallucinations pass.
3. Calibrate thresholds on human labels. If the judge emits probabilities, scores, or normalized ratings, choose thresholds on a fixed human validation set according to the target policy.
4. Normalize judge outputs into machine-readable labels and scores. Use constrained JSON/schema outputs, binary labels plus confidence, evidence spans, and explicit unsupported claims.
5. Analyze disagreement rather than hiding it. Human annotators may legitimately disagree on borderline factuality and severity.
6. Do not treat inter-LLM agreement as human alignment. LLM judge panels can share correlated errors; human-anchored calibration remains required.

## Fixed human validation sets

RAGTruth is a directly relevant benchmark for retrieval-augmented hallucination detection. It is a human-annotated corpus with nearly 18,000 naturally generated RAG responses, annotated at both response/case and word/span levels. Reported detector results show the precision/recall tradeoff: prompted GPT-4-turbo response-level detection had very high recall but much lower precision, while fine-tuned detectors improved balance.

Implications for this project:

- Keep `outputs/human_annotation/one_tenth_annotation_blind_complete.csv` fixed as the human validation anchor.
- Use stable annotation IDs for comparing detector versions.
- Treat response-level labels as the main benchmark and use claim-level/span-like labels for diagnosis.
- High recall with low precision is useful for candidate surfacing but risky for final automatic judgment.

## Precision/recall operating points

Hallucination detectors should be evaluated at explicit operating points. A common failure mode is over-flagging: high recall but low precision. Recent work such as Reasoning's Razor and Noisy but Valid emphasizes that judge accuracy alone is insufficient and that human-labeled calibration sets are needed to estimate and correct judge true-positive/false-positive behavior.

Recommendations:

- Report confusion matrix, precision, recall, F1, specificity, false-positive rate, and false-negative rate.
- Maintain separate modes: high-recall triage, balanced benchmark mode, and high-precision automatic blocking mode.
- Review false positives by category: valid paraphrase, judge missed visual evidence, ambiguous annotation guideline, over-strict prompt, and schema mapping errors.

## Output normalization

Free-form judge explanations are useful for debugging, but scoring should use constrained labels and numeric fields. G-EVAL is a useful reference because it separates criteria, evaluation steps, form-filling, and normalized scoring.

Suggested hallucination judge schema:

```json
{
  "claim_checks": [
    {
      "claim": "...",
      "claim_type": "object_claim | attribute_claim | spatial_claim | reasoning_claim | causal_claim | inconsistency_claim | answer_claim",
      "support_status": "supported | contradicted | unverifiable | not_applicable",
      "fine_label": "OBJ | ATT | SPA | IR | CI | INC | SO | None",
      "evidence": "...",
      "rationale": "..."
    }
  ],
  "overall": {
    "is_hallucination": true,
    "confidence": "low | medium | high",
    "primary_label": "...",
    "explanation": "..."
  }
}
```

Normalization guidance:

- Store rich claim-level output first; convert to binary labels only for evaluation.
- Score from normalized claim checks, not from free-text rationale.
- Preserve raw judge output, normalized detector result, and mismatch flags for audit.
- If summary fields conflict with claim checks, either fail validation or mark `raw_normalized_mismatch=true`.

## Human-alignment validation and disagreement analysis

Human annotations are not always single-answer labels. Borderline hallucinations, partial support, and severity judgments can produce legitimate disagreement. Rating-indeterminacy and judge-distribution papers recommend preserving disagreement structure rather than collapsing everything into a single label too early.

Recommended workflow:

1. Keep the current fixed human labels unchanged for longitudinal comparison.
2. Compute judge-human agreement using the same label mapping for every detector version.
3. Separate clear cases from ambiguous/disagreement cases if multi-annotator labels become available.
4. Review false positives and false negatives separately with image/question/reference/model response/judge rationale.
5. Version every detector prompt/schema/normalization change.

## Sources

- https://aclanthology.org/2024.acl-long.585/
- https://github.com/ParticleMedia/RAGTruth
- https://www.datadoghq.com/blog/ai/llm-hallucination-detection/
- https://aclanthology.org/2026.eacl-long.190/
- https://openreview.net/forum?id=hEhxreaLdU
- https://aclanthology.org/2023.emnlp-main.153.pdf
- https://proceedings.neurips.cc/paper_files/paper/2023/file/91f18a1287b398d378ef22505bf41832-Paper-Datasets_and_Benchmarks.pdf
- https://arxiv.org/abs/2305.14251
- https://arxiv.org/abs/2403.18802
- https://huggingface.co/vectara/hallucination_evaluation_model

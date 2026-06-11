# Research Report: Improving the Zero-shot Judge for Multimodal Hallucination Detection

## Purpose

This research supports the task: improve the current zero-shot judge method while keeping existing model responses and the fixed Human-as-Judge validation dataset unchanged.

The fixed human validation file is:

```text
outputs/human_annotation/one_tenth_annotation_blind_complete.csv
```

This dataset should be used as the stable comparison anchor for future zero-shot judge versions.

## Main conclusion

The strongest direction is not to replace the zero-shot judge with another simple binary prompt. The better method is a stricter process-based judge:

```text
claim extraction
→ evidence grounding
→ support decision
→ taxonomy mapping
→ consistency validation
→ response-level aggregation
```

The current detector already has part of this design, but the `ann_0009` case shows the failure mode: raw judge summary fields can contradict claim-level normalized results. The next judge version should make claim-level checks the only source of truth and force summary fields to be derived consistently from them.

## 1. Judge design implications

Recent LLM-as-judge and process-based hallucination detection work supports claim-level evaluation instead of a single holistic verdict. Useful design patterns include:

- decompose model response into atomic claims;
- classify each claim type;
- verify each claim against image/question/reference/context;
- label support status per claim;
- map only unsupported or contradicted claims to taxonomy labels;
- aggregate from claim labels into final response-level judgment;
- validate that final JSON fields are mutually consistent.

For this project, the judge should be redesigned so every hallucination label must be backed by a concrete claim with:

```text
claim + support_status + evidence/rationale + fine_label
```

If no claim is `contradicted` or `unverifiable` with a non-None label, then the final result must be non-hallucination.

## 2. Multimodal-specific implications

POPE and MathVista should not be judged with exactly the same informal criteria.

### POPE

POPE primarily tests object presence/absence. The judge should:

- focus on `OBJ` errors;
- count unsupported extra object mentions;
- avoid penalizing concise yes/no answers for lacking detailed evidence;
- avoid converting ambiguity or sparse wording into hallucination.

### MathVista

MathVista requires visual understanding plus mathematical/logical reasoning. The judge should:

- check chart/table/diagram values and attributes;
- check whether reasoning follows from image and question context;
- treat visible CoT as part of the model response;
- distinguish invalid reasoning from legitimate inference;
- never treat `reference_answer=UNAVAILABLE` as contradiction evidence.

The `ann_0009` issue is a concrete example: an ecological inference from a food web may be legitimate if grounded in diagram semantics, even if it is not a directly visible object claim.

## 3. Calibration and human alignment

The existing Human-as-Judge result shows high recall but low precision:

```text
Accuracy = 62.5%
Precision = 40.9%
Recall = 81.8%
F1 = 54.5%
Cohen's kappa = 0.282
MCC = 0.332
```

This means the current detector is useful for surfacing candidate hallucinations but too overstrict for reliable final judgment. Future improvements should primarily reduce false positives while monitoring whether recall drops too much.

For every detector version, report:

- TP / FP / TN / FN;
- Accuracy;
- Precision;
- Recall;
- F1;
- Cohen's kappa;
- MCC;
- detector hallucination rate vs human hallucination rate;
- disagreement categories.

## 4. Recommended next zero-shot judge method

### 4.1 Prompt structure

Use a staged prompt:

1. Understand the task and available evidence.
2. Extract atomic claims from the model response.
3. Mark whether each claim is answer-only, visual, reasoning, causal, or meta/explanatory.
4. Verify each claim against image/question/choices/reference context.
5. Assign support status:
   - `supported`
   - `contradicted`
   - `unverifiable`
   - `not_applicable`
6. Assign fine taxonomy only if support status is `contradicted` or `unverifiable`.
7. Aggregate final response-level fields from claim checks.
8. Perform a consistency check before returning JSON.

### 4.2 Schema changes

Recommended schema fields:

```json
{
  "claim_checks": [
    {
      "claim": "string",
      "claim_type": "object_claim | attribute_claim | spatial_claim | reasoning_claim | causal_claim | inconsistency_claim | semantic_claim | answer_claim | non_claim",
      "support_status": "supported | contradicted | unverifiable | not_applicable",
      "fine_label": "OBJ | ATT | SPA | IR | CI | INC | SO | None",
      "evidence_basis": "image | question | choices | reference | response_internal | diagram_semantics | none",
      "rationale": "string"
    }
  ],
  "aggregation": {
    "is_hallucination": true,
    "hallucination_labels": ["OBJ"],
    "primary_label": "OBJ",
    "coarse_labels": ["Factual Hallucination"],
    "unsupported_visual_claim": true,
    "confidence": "low | medium | high",
    "summary_consistent_with_claims": true,
    "aggregation_rule": "any_contradicted_or_unverifiable_claim_with_non_none_label"
  },
  "explanation": "short explanation"
}
```

### 4.3 Normalization rules

The detector should enforce these rules after parsing:

- If `support_status=supported`, force `fine_label=None`.
- If `fine_label=None`, the claim does not contribute to hallucination labels.
- Final `is_hallucination = bool(hallucination_labels)`.
- `hallucination_labels` are derived only from claim checks, not from explanation text.
- If raw aggregation fields disagree with derived labels, set `raw_normalized_mismatch=true`.
- Optionally reject/retry outputs where `summary_consistent_with_claims=false`.

## 5. Recommended validation workflow

1. Keep the fixed human annotation CSV unchanged.
2. Run current detector as baseline.
3. Implement improved prompt/schema/normalization as a versioned detector mode.
4. Re-run only the judge on existing model responses.
5. Export detector outputs for the same 40 human annotation units.
6. Evaluate against `outputs/human_annotation/one_tenth_annotation_blind_complete.csv`.
7. Compare old vs new metrics and disagreements.

Main success target:

```text
Increase precision and F1 while preserving acceptable recall.
```

A realistic target for the next iteration is:

```text
Precision: 40.9% → higher
F1: 54.5% → higher
Recall: do not collapse far below current 81.8%
```

## 6. Implementation priorities

P0:

- Rewrite `prompts/judge/zero_shot_judge.txt` with stricter claim-grounding and aggregation rules.
- Update JSON schema in `src/detectors/zero_shot_judge.py` to include aggregation consistency fields if needed.
- Strengthen normalization and mismatch handling.
- Add tests for raw/normalized contradiction cases like `ann_0009`.

P1:

- Add detector version name to outputs.
- Add comparison script for old-vs-new judge on the fixed human set.
- Export false-positive reduction analysis.

P2:

- Add optional two-pass judge or self-verification retry for inconsistent outputs.
- Add calibration threshold if confidence becomes numeric.

## Sources

- [PROBE: PROcess-Based BEnchmark for Hallucination Detection](https://openreview.net/forum?id=NuSq4FmzFb)
- [MARCH: Multi-Agent Reinforced Self-Check for LLM Hallucination](https://arxiv.org/abs/2603.24579)
- [G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634)
- [Promptfoo LLM-as-a-judge guide](https://www.promptfoo.dev/docs/guides/llm-as-a-judge/)
- [Structured Generation for LLM-as-a-Judge Evaluations](https://www.comet.com/site/structured-generation-llm-as-a-judge/)
- [POPE: Evaluating Object Hallucination in Large Vision-Language Models](https://arxiv.org/abs/2305.10355)
- [POPE GitHub](https://github.com/RUCAIBox/POPE)
- [MathVista benchmark](https://arxiv.org/abs/2310.02255)
- [MathVista project page](https://mathvista.github.io/)
- [MIRAGE: Assessing Hallucination in Multimodal Reasoning Chains](https://arxiv.org/abs/2505.24238)
- [More Thinking, Less Seeing?](https://arxiv.org/abs/2505.21523)
- [Seeing Clearly without Training](https://arxiv.org/abs/2603.02754)
- [RAGTruth](https://aclanthology.org/2024.acl-long.585/)
- [FActScore](https://arxiv.org/abs/2305.14251)
- [SAFE / Long-form factuality](https://arxiv.org/abs/2403.18802)

# Zero-shot LLM-as-Judge design patterns for hallucination detection

## Core findings

Zero-shot hallucination judges are moving away from single binary verdict prompts and toward process-based, claim-level judging. The most relevant pattern is:

1. decompose the model response into atomic claims;
2. classify each claim type;
3. verify each claim against available evidence;
4. assign support status and hallucination label per claim;
5. aggregate claim-level labels into the final response-level verdict;
6. validate that summary fields are consistent with claim-level evidence.

This directly matches the current project's need because the existing judge already uses claim checks, but raw summary fields can contradict normalized claim-level results.

## Process-based judge design

PROBE is the closest recent reference for process-based hallucination detection. It frames hallucination detection as multiple subtasks: claim decomposition, evidence finding, evidence evaluation, and hallucination localization. This suggests the project judge should explicitly separate:

- claim extraction;
- evidence grounding;
- support decision;
- taxonomy mapping;
- final aggregation.

For the current detector, this means the prompt should not ask the judge to simply output `is_hallucination`; it should require the judge to justify every triggered hallucination label from a concrete unsupported or contradicted claim.

## Structured output and schema discipline

G-EVAL is a useful general LLM-as-judge pattern: specify task instructions, evaluation criteria, evaluation steps, and a form-filling output. Practical LLM-as-judge tools also emphasize structured generation so downstream scoring does not depend on free-form explanations.

Recommended design implications:

- Keep JSON schema output.
- Make claim checks the source of truth.
- Add explicit consistency fields:
  - `summary_consistent_with_claims: true | false`
  - `aggregation_rule_applied: "any_unsupported_claim"`
- Require `fine_label=None` whenever `support_status=supported`.
- Require `is_hallucination=false` whenever no claim has `support_status in {contradicted, unverifiable}` with a non-None label.
- Treat schema violations as judge failures or retries, not valid detector outputs.

## Self-verification and consistency checks

Recent multi-agent/self-check methods such as MARCH decompose responses into atomic propositions and check them against evidence. For this project, a lightweight zero-shot version can be implemented without multi-agent overhead:

1. First pass: extract claims.
2. Second pass inside the same prompt: verify claims.
3. Third pass inside the same prompt: aggregate and check consistency.

The judge prompt should explicitly ask: "Before finalizing, verify that `is_hallucination`, `hallucination_labels`, `primary_label`, and `claim_checks` are mutually consistent."

## Known reliability limits

LLM-as-judge systems may be overstrict, biased toward certain wording, and sensitive to ambiguous evidence. Existing research and tool documentation emphasize that judge outputs should be benchmarked against human labels and analyzed with false positives/false negatives, not trusted as ground truth.

For this project, the main known failure is already visible: high recall but low precision against Human-as-Judge labels. Therefore judge improvement should focus on reducing false positives while preserving recall.

## Practical recommendations for the project

- Add a stricter aggregation contract in the judge prompt.
- Make unsupported claims mandatory for every hallucination label.
- Split support statuses into `supported`, `contradicted`, `unverifiable`, and `not_applicable`.
- Do not label valid ecological/mathematical inference as hallucination merely because it is not visually explicit, if it is grounded in the question context and diagram semantics.
- Preserve raw judge output, normalized output, and mismatch flags.
- Compare every prompt/schema version on the fixed Human-as-Judge CSV.

## Sources

- https://openreview.net/forum?id=NuSq4FmzFb
- https://openreview.net/forum?id=CUQZyxrWfp
- https://arxiv.org/abs/2603.24579
- https://arxiv.org/abs/2604.18803
- https://arxiv.org/abs/2303.16634
- https://deepeval.com/docs/metrics-llm-evals
- https://www.promptfoo.dev/docs/guides/llm-as-a-judge/
- https://www.comet.com/site/structured-generation-llm-as-a-judge/

# Multimodal hallucination detection benchmarks and taxonomies

## Core findings

Multimodal hallucination detection for image-grounded QA needs to distinguish visual factual errors from reasoning errors. The current project taxonomy is well aligned with the literature:

- visual factual hallucination: object/category, attribute, spatial/location;
- logical/reasoning hallucination: invalid reasoning, context inconsistency, internal inconsistency, semantic over-attribution.

The key design challenge is that POPE and MathVista expose different failure modes. POPE is mainly object-presence grounding; MathVista requires visual evidence plus mathematical/logical reasoning.

## POPE and object hallucination

POPE is a polling-based benchmark for object hallucination in large vision-language models. It uses yes/no object-presence questions such as "Is there a [object] in the image?" and includes random, popular, and adversarial sampling strategies.

Implications for this project:

- For POPE, hallucination labels should be strict but narrow: object presence/absence claims should map mainly to `OBJ`.
- Extra unsupported object mentions in reasoning or explanations should count.
- But short yes/no answers should not be over-penalized for lacking detailed evidence.
- The judge should avoid converting ambiguity or sparse explanation into hallucination.

Follow-up benchmarks such as H-POPE extend object hallucination to hierarchical attributes, while RePOPE highlights that even benchmark labels may need correction. This supports keeping human validation as the final alignment anchor.

## MathVista and visual reasoning errors

MathVista evaluates mathematical reasoning in visual contexts. It combines visual understanding with numerical, diagram, chart, table, geometry, and logical reasoning.

Recent multimodal reasoning research reports that extended reasoning can increase answer accuracy while reducing visual grounding, or can expose unsupported intermediate claims. Relevant patterns include:

- reading wrong values or attributes from charts/diagrams;
- using unsupported assumptions;
- producing valid final answer with unsupported CoT details;
- reasoning from diagram semantics but being incorrectly marked as unsupported if the judge demands direct visual statement only.

Implications for this project:

- For MathVista, the judge must check both visual evidence and reasoning validity.
- It should distinguish unsupported reasoning from legitimate inference grounded in the diagram/question.
- It should not mark hallucination solely because the reference answer is `UNAVAILABLE`.
- It should treat visible CoT as part of the response, but avoid overstrictly penalizing reasonable intermediate reasoning.

## Reasoning-chain hallucination

MIRAGE and related works focus on hallucinations inside multimodal reasoning chains. Categories include factual, logical, spatial, contextual, and fabrication hallucinations. This supports the project's decision to include visible CoT in judgment.

However, research on amplified hallucination and evidence collapse suggests that more reasoning can make unsupported claims easier to detect but can also create more opportunities for false positives. This aligns with the current Human-as-Judge result: the detector is sensitive but overstrict.

## Relation to Seeing Clearly without Training

Seeing Clearly without Training introduces RSHBench for factual/logical hallucinations in remote-sensing VQA and RADAR, a training-free mitigation method using intrinsic attention for localization and fine-grained reasoning. Although it is not directly a zero-shot text judge, it is useful conceptually:

- separate factual and logical hallucinations;
- localize visual evidence before reasoning;
- use fine-grained reasoning rather than only final answer matching;
- prefer training-free improvements when no new model training is planned.

For this project, the analogous idea is to make the judge explicitly ground each claim in image/question evidence before assigning taxonomy.

## Practical recommendations for the project

- Keep dataset-specific instructions in the judge prompt:
  - POPE: object presence/absence, avoid over-penalizing concise yes/no.
  - MathVista: visual attribute/value reading plus reasoning validity.
- Add explicit rule: `reference_answer=UNAVAILABLE` must not be treated as contradiction.
- Require each hallucination label to cite a concrete claim and grounding problem.
- Add `legitimate_inference` or `reasoning_supported_by_question_context` as a non-hallucination rationale class to reduce false positives.
- Evaluate separately by dataset and prompt type because POPE and MathVista have different error surfaces.

## Sources

- https://arxiv.org/abs/2305.10355
- https://github.com/RUCAIBox/POPE
- https://evalscope.readthedocs.io/en/latest/benchmarks/pope.html
- https://arxiv.org/abs/2507.19024
- https://arxiv.org/abs/2411.04077
- https://huggingface.co/MM-Hallu/RePOPE
- https://arxiv.org/abs/2412.20622
- https://arxiv.org/abs/2310.02255
- https://mathvista.github.io/
- https://arxiv.org/abs/2604.04207
- https://arxiv.org/abs/2505.24238
- https://arxiv.org/abs/2505.21523
- https://arxiv.org/abs/2603.02754
- https://aclanthology.org/2026.eacl-long.287/

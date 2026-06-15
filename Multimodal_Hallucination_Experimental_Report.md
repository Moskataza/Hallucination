# Hallucination Detection and Evaluation for Multimodal Large Language Models

## Abstract

This report studies hallucination detection and evaluation for multimodal large language models (MLLMs) on visual question answering and visual mathematical reasoning tasks. The study focuses on three questions: how to define hallucinations in multimodal responses, how to detect them automatically without training an additional model, and whether automatic detection agrees with human judgments. We evaluate two representative tasks, POPE for object-presence visual question answering and MathVista for multimodal mathematical reasoning. We compare two model families, Gemini as a closed-source model and Qwen-VL as an open-source or open-interface model, under two prompting settings: direct answer and evidence-grounded chain-of-thought (CoT). We implement a prompt-based zero-shot judge and improve it into Zero-shot Judge v2, a taxonomy-constrained claim-level judge that checks atomic claims against explicit evidence sources and then aggregates response-level hallucination labels deterministically.

The main experiment shows that hallucination patterns depend strongly on the task. POPE direct-answer responses are dominated by object hallucinations, while MathVista responses are dominated by invalid reasoning and visual attribute errors. CoT increases response-level hallucination rates across all settings, but claim-level analysis shows that this increase is partly caused by higher claim exposure: CoT produces more checkable claims per response, so more potential errors become visible to the detector. Human-as-Judge validation on 40 manually annotated samples shows that Zero-shot Judge v2 improves over the baseline zero-shot judge, increasing accuracy from 0.6750 to 0.8000 and F1 from 0.6286 to 0.6923, while reducing false positives from 11 to 4. However, v2 still misses implicit claims in short answers, visual counting or attribute errors, and reasoning-final answer inconsistencies. The report also discusses a bonus cross-task and domain-extension analysis, including the structural transfer of the claim-level judge from POPE to MathVista and a pilot-ready XLRS-bench remote-sensing extension.

## 1. Introduction

Multimodal large language models can answer questions about images, charts, diagrams, and visual scenes. However, they may generate responses that are fluent but not grounded in the visual input or the task constraints. These errors are often described as hallucinations. In multimodal settings, hallucination is not limited to fabricated objects. It may also include incorrect visual attributes, unsupported spatial relations, invalid mathematical reasoning, contradictions between intermediate reasoning and final answers, or semantic overreach beyond what the image can support.

This study evaluates hallucinations in MLLM outputs under the assessment topic: hallucination detection and evaluation for multimodal large language models. The work has four goals:

1. Define multiple types of hallucinations with clear examples.
2. Build an automatic hallucination detection method without training a new model.
3. Evaluate hallucination patterns across tasks, models, and prompting strategies.
4. Validate the automatic detector against human judgments and analyze disagreement cases.

The experimental design follows a controlled structure: two tasks, two models, two prompting styles, and two judge versions. POPE is used as the required general VQA task, and MathVista is used as the second task because it stresses visual mathematical reasoning. Gemini represents the closed-source model setting, while Qwen-VL represents the open-source or open-interface model setting. Direct prompting is compared with evidence-grounded CoT prompting to study how reasoning traces affect hallucination detection.

The central methodological contribution is Zero-shot Judge v2. It remains a prompt-based GPT-style judge and does not train any model, but it introduces a stricter claim-level protocol. The judge extracts atomic checkable claims, assigns an evidence source, classifies support status, maps unsupported or contradicted claims to a fine-grained hallucination taxonomy, and lets the program derive final response-level labels. This design reduces reliance on the judge's raw top-level summary and improves alignment with human annotation.

## 2. Hallucination Definition and Taxonomy

In this study, a multimodal hallucination is defined as a checkable assertion in a model response that is inconsistent with, unsupported by, or not entailed by the image, question, choices, reference answer, mathematical or diagram rules, or the response's own internal logic. This definition includes both visually grounded errors and reasoning errors.

The study uses four coarse hallucination categories and seven fine-grained labels.

| Coarse category | Fine labels | Main criterion |
|---|---|---|
| Visual factual hallucination | `OBJ`, `ATT`, `SPA` | Whether objects, attributes, quantities, or spatial relations are supported by the image |
| Reasoning hallucination | `IR`, `CI` | Whether the reasoning step, mathematical rule, or causal relation is valid |
| Internal consistency hallucination | `INC` | Whether the reasoning, visual evidence, and final answer are mutually consistent |
| Semantic overreach | `SO` | Whether the response extrapolates identity, intention, age, function, or scene semantics beyond available evidence |

### 2.1 Visual Factual Hallucination

Visual factual hallucination refers to incorrect or unsupported statements about directly observable visual content. It includes object presence, attributes, quantities, and spatial relations.

- `OBJ` example 1: In POPE, the question asks whether there is a handbag in the image. If the image does not contain a handbag but the model answers yes, the response contains an object-presence hallucination.
- `OBJ` example 2: If the image only shows a person and a dog, but the model states that there is a cat, the model fabricates an object.
- `ATT` example 1: In MathVista, the model states that there are 16 animals on land and 5 animals in water, but manual inspection shows that the counts are wrong. This is an attribute or quantity-reading error.
- `ATT` example 2: In a chart question, the model says that the Firebrick bar is clearly higher than the Midnight Blue bar, while the chart shows that the values are equal or nearly equal.
- `SPA` example 1: The model states that object A is to the left of object B, but the image shows object A on the right.
- `SPA` example 2: The model says that a vehicle is in the middle of the road, but the image shows it parked at the side.

### 2.2 Reasoning Hallucination

Reasoning hallucination refers to an invalid inference from visual evidence, textual constraints, options, or mathematical rules to an answer.

- `IR` example 1: In a geometry problem, the model concludes that two line segments are equal without sufficient corresponding sides or angles.
- `IR` example 2: In MathVista, the model uses an area, slope, or ratio formula that does not match the visual values in the image.
- `CI` example 1: The image shows a person holding a phone, and the model concludes that the person is calling a friend. The image supports holding a phone, but not the specific intention.
- `CI` example 2: In a remote-sensing image, the model sees a dense group of buildings and concludes that the region is an industrial pollution source without supporting evidence.

### 2.3 Internal Consistency Hallucination

Internal consistency hallucination occurs when different parts of the same response conflict with each other.

- `INC` example 1: The model reasons that the slope at `x=2` is greater than the slope at `x=-2`, but gives a final answer that semantically means `f'(-2) > f'(2)`.
- `INC` example 2: The model first states that the target object is absent, but later uses that object as evidence for a yes answer.
- `INC` example 3: The visual evidence says that the blue bar is the highest, but the reasoning uses the red bar as the highest value.

### 2.4 Semantic Overreach

Semantic overreach occurs when the model infers high-level meaning that the image does not support.

- `SO` example 1: The question asks how many people in the image were born after World War II. The model answers `0`, but the image alone cannot usually prove the birth years of the people.
- `SO` example 2: The model sees a person wearing white clothing and concludes that the person is a doctor, without textual or contextual evidence.
- `SO` example 3: In a remote-sensing image, the model claims that a building complex is a military base or an industrial park based only on appearance.

This taxonomy satisfies the requirement to define at least two hallucination types and supports later automatic detection, human annotation, and fine-grained error analysis.

## 3. Tasks, Models, and Experimental Variables

### 3.1 Tasks

The experiment uses two main tasks.

| Task | Role in assessment | Description |
|---|---|---|
| POPE | Required VQA task | General visual question answering focused on object presence and absence |
| MathVista | Second selected task | Visual mathematical reasoning involving charts, diagrams, quantities, geometry, and reasoning |

POPE mainly tests object grounding. The model must answer whether a queried object appears in the image. This task is expected to produce many `OBJ` errors when hallucinations occur.

MathVista is more complex. It requires the model to read visual attributes, interpret diagrams or charts, apply mathematical rules, and map reasoning results back to the final answer. This task is expected to produce more `ATT`, `IR`, and `INC` errors.

### 3.2 Models

The experiment evaluates two model families.

| Model | Type | Role |
|---|---|---|
| Gemini | Closed-source model | Representative closed-source multimodal model |
| Qwen-VL | Open/open-interface model | Representative open or open-interface multimodal model |

The goal is not to produce a universal benchmark ranking, but to compare hallucination profiles under the same tasks, prompts, and detector settings.

### 3.3 Prompting Settings

Each model is evaluated under two prompting styles.

| Prompt type | Description | Purpose |
|---|---|---|
| Direct answer | The model gives a concise final answer | Tests compact answer behavior and object/answer-level hallucinations |
| Evidence-grounded CoT | The model gives visual evidence, reasoning, and final answer | Tests whether explicit reasoning changes hallucination exposure and type distribution |

The CoT setting is used for analysis rather than for hiding or revealing private reasoning. The prompt asks for structured visual evidence and reasoning so that the detector can inspect intermediate claims.

### 3.4 Experimental Matrix

The full experimental matrix is:

```text
2 tasks × 2 models × 2 prompt types × 2 judge versions
```

The two judge versions are:

1. Baseline zero-shot judge, also treated as zero-shot judge v1.
2. Zero-shot Judge v2, the improved claim-level judge.

The main dependent variables include response-level hallucination label, fine-grained hallucination labels, hallucination rate, automatic-human alignment metrics, and claim-level exposure metrics.

## 4. Automatic Hallucination Detection Method

### 4.1 Baseline Zero-shot Judge

The baseline detector is a prompt-based MLLM-as-a-judge method. It does not train a model. For each model response, the judge receives the image, question, choices when available, reference answer, parsed visual evidence, parsed reasoning, parsed final answer, and taxonomy definitions.

The baseline judge asks the model to:

1. understand the question and task context;
2. extract explicit verifiable claims from the model response;
3. classify each claim into object, attribute, spatial, reasoning, causal, consistency, semantic, or answer-related types;
4. judge whether each claim is supported, contradicted, or unverifiable;
5. map problematic claims to `OBJ`, `ATT`, `SPA`, `IR`, `CI`, `INC`, or `SO`;
6. produce a JSON detector output.

The baseline output includes fields such as answer correctness, hallucination decision, hallucination vector, primary label, coarse labels, unsupported visual claim, confidence, claim checks, and explanation.

The baseline is interpretable and simple, but it has several weaknesses:

- It lacks an explicit `evidence_source` field, so the judge may not clearly separate image evidence, question semantics, reference answer, mathematical rules, and internal consistency.
- It does not distinguish non-claims or not-applicable text from checkable claims.
- It tends to treat unverifiable answers strictly, especially when the reference answer is unavailable.
- It may produce inconsistency between top-level hallucination summaries and claim-level checks.
- It is too strict for short yes/no answers and some legitimate MathVista reasoning.

As a result, the baseline has high recall but low precision on the human validation set.

### 4.2 Zero-shot Judge v2

Zero-shot Judge v2 keeps the zero-shot, prompt-based nature of the baseline but strengthens the evidence-checking protocol. Its core workflow is:

```text
Fixed hallucination taxonomy
→ atomic checkable claim extraction
→ evidence source assignment
→ support-status judgment
→ fine-label mapping
→ separate final-answer correctness judgment
→ deterministic response-level aggregation
```

The key additions are:

| Component | Role |
|---|---|
| `claim_checks` | Stores claim-level evidence checks |
| `evidence_source` | Records whether a claim should be checked against image, question, choices, reference answer, math rule, diagram rule, internal consistency, or none |
| `support_status` | Uses `supported`, `contradicted`, `unverifiable`, or `not_applicable` |
| `non_claim` handling | Prevents formatting text and vague statements from triggering hallucination |
| deterministic aggregation | Derives response-level hallucination labels from normalized claim checks instead of trusting raw top-level judge output |
| consistency audit | Records whether raw and normalized summaries disagree |

The v2 design addresses the baseline's main failure mode: excessive false positives caused by treating every uncertainty or missing reasoning step as hallucination. In v2, an incorrect final answer alone does not automatically imply a hallucination. The detector asks whether the response contains a claim that is contradicted, unsupported, semantically overreaching, internally inconsistent, or based on invalid reasoning.

### 4.3 Difference Between Baseline and v2

| Dimension | Baseline zero-shot judge | Zero-shot Judge v2 |
|---|---|---|
| Method type | Prompt-based zero-shot judge | Prompt-based zero-shot judge |
| Training | None | None |
| Main granularity | Response-level with claim checks | Taxonomy-constrained claim-level evidence checking |
| Claim extraction | Explicit verifiable claims | Atomic checkable claims, including answer, factual, reasoning, semantic, and non-claim cases |
| Evidence source | Mostly implicit | Explicit `evidence_source` field |
| Support status | Supported / contradicted / unverifiable | Supported / contradicted / unverifiable / not applicable |
| Non-claim handling | Not explicit | Explicit `non_claim` and `not_applicable` handling |
| Final answer handling | Final answer may trigger hallucination too easily | Final-answer correctness is judged separately |
| Reference unavailable | May still over-penalize uncertainty | `UNAVAILABLE` reference answer cannot be used as contradiction evidence |
| Aggregation | Partly trusts top-level judge summary | Uses normalized claim checks for deterministic aggregation |
| Main tendency | High recall, low precision | More balanced precision and recall |

## 5. Evaluation Metrics

The experiment uses four groups of metrics.

| Metric group | Metrics | Purpose |
|---|---|---|
| Response-level | Hallucination rate | Measures the proportion of responses detected as hallucinated |
| Binary alignment | Accuracy, precision, recall, F1 | Measures agreement between automatic detector and human labels |
| Agreement metrics | Cohen's Kappa, MCC | Measures agreement under possible class imbalance |
| Claim-level | Claims per response, unsupported claim ratio, label-triggering unsupported ratio | Separates true error increase from increased claim exposure |

Accuracy and F1 satisfy the requirement to design at least two evaluation metrics. Hallucination rate supports model, prompt, and task comparisons. Cohen's Kappa and MCC provide additional robustness for Human-as-Judge analysis. Claim-level metrics are used to interpret why CoT changes response-level hallucination rates.

## 6. Human-as-Judge Validation Design

The Human-as-Judge validation set contains 40 manually annotated samples. It covers POPE and MathVista, Gemini and Qwen-VL, and direct and CoT prompting. Human annotation includes:

- binary hallucination label;
- coarse hallucination category;
- fine-grained labels from `OBJ`, `ATT`, `SPA`, `IR`, `CI`, `INC`, and `SO`;
- human rationale;
- answer correctness.

The human annotation file is treated as the gold label for detector validation. The v2 alignment uses a versioned key to avoid overwriting baseline alignment results. The evaluation computes TP, FP, TN, FN, accuracy, precision, recall, F1, Cohen's Kappa, and MCC. Disagreement cases are exported and analyzed manually to understand where the detector differs from human judgment.

The sample size is intentionally small but satisfies the assessment requirement of 20–50 human-labeled samples. The analysis is therefore diagnostic rather than a statistically definitive benchmark.

## 7. Detector Output Completeness

Zero-shot Judge v2 was run on all one-tenth model responses without overwriting the baseline detector files.

| Dataset | Model / Prompt | Rows | Validity |
|---|---:|---:|---|
| POPE | Gemini Direct | 900 / 900 | valid JSON, no duplicates, no missing rows |
| POPE | Gemini CoT | 900 / 900 | valid JSON, no duplicates, no missing rows |
| POPE | Qwen Direct | 900 / 900 | valid JSON, no duplicates, no missing rows |
| POPE | Qwen CoT | 900 / 900 | valid JSON, no duplicates, no missing rows |
| MathVista | Gemini Direct | 514 / 514 | valid JSON, no duplicates, no missing rows |
| MathVista | Gemini CoT | 514 / 514 | valid JSON, no duplicates, no missing rows |
| MathVista | Qwen Direct | 514 / 514 | valid JSON, no duplicates, no missing rows |
| MathVista | Qwen CoT | 514 / 514 | valid JSON, no duplicates, no missing rows |

This indicates that the v2 JSON schema is stable across both object-presence VQA and visual mathematical reasoning tasks.

## 8. Main Experimental Results

### 8.1 Hallucination Rates and Fine-Label Distribution

| Dataset | Model | Prompt | Samples | Hallucination count | Hallucination rate | Main fine labels |
|---|---|---|---:|---:|---:|---|
| POPE | Gemini | Direct | 900 | 61 | 6.78% | `OBJ=61` |
| POPE | Gemini | CoT | 900 | 189 | 21.00% | `OBJ=95`, `IR=63`, `SO=17` |
| POPE | Qwen | Direct | 900 | 69 | 7.67% | `OBJ=69` |
| POPE | Qwen | CoT | 900 | 180 | 20.00% | `OBJ=95`, `SO=42`, `IR=35` |
| MathVista | Gemini | Direct | 514 | 97 | 18.87% | `IR=81`, `ATT=10` |
| MathVista | Gemini | CoT | 514 | 180 | 35.02% | `IR=103`, `ATT=46`, `OBJ=13` |
| MathVista | Qwen | Direct | 514 | 119 | 23.15% | `IR=95`, `ATT=12` |
| MathVista | Qwen | CoT | 514 | 234 | 45.53% | `IR=144`, `ATT=55`, `OBJ=20` |

Several patterns are clear.

First, POPE direct-answer hallucinations are almost entirely `OBJ` errors. This matches the task design because POPE asks whether a target object is present in the image.

Second, MathVista has higher hallucination rates than POPE in both direct and CoT settings. Its main labels are `IR` and `ATT`, which reflects the difficulty of combining visual reading with mathematical reasoning.

Third, CoT increases response-level hallucination rates in every dataset-model setting. However, this result must be interpreted with claim-level exposure metrics rather than as direct proof that CoT always makes models worse.

### 8.2 Human-as-Judge Alignment: Baseline vs v2

| Detector | TP | FP | TN | FN | Precision | Recall | F1 | Accuracy | Cohen's Kappa | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline zero-shot judge | 11 | 11 | 16 | 2 | 0.5000 | 0.8462 | 0.6286 | 0.6750 | 0.3720 | 0.4131 |
| Zero-shot Judge v2 | 9 | 4 | 23 | 4 | 0.6923 | 0.6923 | 0.6923 | 0.8000 | 0.5442 | 0.5442 |

Zero-shot Judge v2 improves overall agreement with human labels. False positives decrease from 11 to 4, precision increases from 0.5000 to 0.6923, F1 increases from 0.6286 to 0.6923, and accuracy increases from 0.6750 to 0.8000. Cohen's Kappa and MCC also improve to 0.5442.

The tradeoff is lower recall. False negatives increase from 2 to 4, and recall decreases from 0.8462 to 0.6923. This is expected because v2 is more conservative: it no longer treats every uncertain or unsupported-looking answer as a hallucination. The result is a more balanced detector that better matches human boundaries, but it still misses some implicit or visually subtle errors.

## 9. CoT Impact Analysis

### 9.1 Response-Level Effect

| Dataset | Model | Direct hallucination rate | CoT hallucination rate | Difference | Direct main labels | CoT main labels |
|---|---|---:|---:|---:|---|---|
| POPE | Gemini | 6.78% | 21.00% | +14.22% | `OBJ=61` | `OBJ=95`, `IR=63`, `SO=17` |
| POPE | Qwen | 7.67% | 20.00% | +12.33% | `OBJ=69` | `OBJ=95`, `SO=42`, `IR=35` |
| MathVista | Gemini | 18.87% | 35.02% | +16.15% | `IR=81`, `ATT=10` | `IR=103`, `ATT=46`, `OBJ=13` |
| MathVista | Qwen | 23.15% | 45.53% | +22.37% | `IR=95`, `ATT=12` | `IR=144`, `ATT=55`, `OBJ=20` |

At the response level, CoT increases hallucination rates for both models and both datasets. This does not mean that CoT is simply harmful. CoT changes both model behavior and observability.

### 9.2 Claim Exposure Effect

CoT substantially increases the number of checkable claims per response.

| Dataset | Model | Direct claims / response | CoT claims / response | Change | Direct label-triggering unsupported ratio | CoT label-triggering unsupported ratio | Change in response hallucination rate |
|---|---|---:|---:|---:|---:|---:|---:|
| POPE | Gemini | 1.12 | 5.57 | +4.45 | 6.06% | 6.70% | +14.22% |
| POPE | Qwen | 1.13 | 6.60 | +5.47 | 6.93% | 5.94% | +12.33% |
| MathVista | Gemini | 2.50 | 8.15 | +5.66 | 11.96% | 7.89% | +16.15% |
| MathVista | Qwen | 1.85 | 6.86 | +5.01 | 14.89% | 13.30% | +22.37% |

This table shows that CoT greatly increases claim exposure. Even when the unsupported-claim ratio does not increase, a response with more claims has more opportunities to contain at least one label-triggering unsupported claim. Therefore, response-level hallucination rate can rise even if the per-claim error rate is stable or lower.

The CoT effect has two components:

1. Exposure effect: CoT reveals more visual observations, numerical readings, and reasoning steps. The detector has more claims to check.
2. Generation effect: CoT may introduce new errors, including over-explanations, invalid intermediate inferences, and semantic overreach.

The study therefore interprets CoT as a diagnostic tool. It improves error localization but also increases the number of borderline claims and the difficulty of automatic judgment.

### 9.3 CoT Effects by Hallucination Type

The effect of CoT depends on the task.

In POPE, direct responses mainly contain a yes/no answer and object-presence claim, so hallucinations are mostly `OBJ`. CoT adds object descriptions, scene explanations, and semantic interpretations. This introduces additional `IR` and `SO` errors.

In MathVista, direct responses already involve hidden visual-mathematical reasoning. CoT makes these hidden steps explicit. As a result, `ATT` and `IR` errors become more visible. The model may misread quantities, chart values, slopes, lengths, colors, or diagram structures, and then build reasoning on those incorrect readings.

Thus, CoT does not have a single universal effect. In object-presence VQA, it mainly increases semantic and explanatory claims. In visual mathematical reasoning, it exposes intermediate visual attributes and mathematical reasoning steps.

## 10. Model Comparison

### 10.1 Hallucination Rate Differences

| Dataset | Prompt | Gemini hallucination rate | Qwen hallucination rate | Difference |
|---|---:|---:|---:|---:|
| POPE | Direct | 6.78% | 7.67% | Qwen +0.89% |
| POPE | CoT | 21.00% | 20.00% | Gemini +1.00% |
| MathVista | Direct | 18.87% | 23.15% | Qwen +4.28% |
| MathVista | CoT | 35.02% | 45.53% | Qwen +10.51% |

Model differences are small on POPE but larger on MathVista. In POPE direct-answer settings, both models mainly make object-presence errors. In POPE CoT, the overall rates remain close, but the label distribution differs: Gemini CoT has more `IR` labels, while Qwen CoT has more `SO` labels. This suggests that Gemini's CoT responses more often trigger reasoning-related concerns, while Qwen's CoT responses more often include broader semantic or scene-level claims.

In MathVista, Qwen has higher hallucination rates than Gemini in both direct and CoT settings. The gap is especially large for CoT, where Qwen reaches 45.53% compared with Gemini's 35.02%. This suggests that complex visual mathematical reasoning amplifies model differences more than object-presence VQA does.

### 10.2 Interpretation

POPE mainly tests whether the model can ground an object label in the image. When the task is this constrained, the two model families show similar hallucination rates.

MathVista combines several abilities: visual attribute reading, chart or diagram understanding, mathematical rule application, natural-language frame binding, and final-answer mapping. Errors in any part of this chain may become `ATT`, `IR`, or `INC` hallucinations. This makes MathVista a stronger stress test for model-level hallucination profiles.

The model comparison therefore supports a task-dependent conclusion: similar performance on object grounding does not imply similar reliability on complex visual reasoning.

## 11. Detailed Failure Case Analysis

The most useful errors are not merely wrong answers. They reveal where the evidence chain breaks. This section analyzes three representative failure cases from the Human-as-Judge disagreement set.

### 11.1 Case 1: Visual Counting Error as `ATT` Hallucination

| Field | Description |
|---|---|
| ID | `ann_0010` |
| Dataset / model / prompt | MathVista / Gemini / CoT |
| Question | Are most of the animals in the water? |
| Model behavior | The model states that there are 16 animals on land and 5 in water, then concludes that most animals are not in the water. |
| Human label | Hallucination, `ATT` |
| Judge result | Non-hallucination, false negative |

This case is an attribute-level visual grounding error. The model gives a structured explanation, but the key visual counts are wrong. If the counts are wrong, the later reasoning may be formally valid but still grounded in false evidence.

The error chain has three steps: identifying animal instances, determining whether each animal is in water or on land, and comparing the counts. A failure in any step can corrupt the final answer. The v2 judge misses the error because it accepts the response's stated visual evidence instead of independently verifying the counts. This shows that CoT can make an answer look more reliable even when the visual evidence is fabricated or misread.

This case reveals a limitation of MLLM-as-a-Judge methods: the judge may be influenced by the evaluated response's wording and may not fully re-check the image. MathVista attribute claims require stricter independent verification.

### 11.2 Case 2: Reasoning-Final Answer Inconsistency as `INC` Hallucination

| Field | Description |
|---|---|
| ID | `ann_0020` |
| Dataset / model / prompt | MathVista / Qwen / CoT |
| Question | The derivative of `f(x)` at `x=-2` is ____ that at `x=2`. |
| Model behavior | The reasoning says that the curve at `x=2` is steeper, so `f'(2)` is greater than `f'(-2)`, but the final answer is `greater`. |
| Human label | Hallucination, `INC` |
| Judge result | Non-hallucination, false negative |

This case is not simply a visual error. The model's local reasoning may appear plausible, but the final answer does not match the question frame. The blank asks about `f'(-2)` relative to `f'(2)`. If the reasoning concludes that `f'(2)` is greater, the final answer should indicate that `f'(-2)` is smaller. Answering `greater` reverses the comparison.

The failure source is language-frame binding. The model must connect a visual or mathematical comparison to the exact subject and direction of the natural-language sentence. The judge misses this because it checks local reasoning claims but does not fully verify whether the final answer is entailed by the reasoning under the question's semantic frame.

This case motivates a specific improvement: the detector should include reasoning-to-answer entailment checks for direction-sensitive questions such as greater/less than, yes/no, multiple-choice, and fill-in-the-blank comparisons.

### 11.3 Case 3: Short Answer with Implicit `SO` Hallucination

| Field | Description |
|---|---|
| ID | `ann_0002` |
| Dataset / model / prompt | MathVista / Gemini / Direct |
| Question | How many people in the image were born after the end of World War II? |
| Model behavior | The model answers `0`. |
| Human label | Hallucination, `SO` |
| Judge result | Non-hallucination, false negative |

This case shows the weakness of direct answers. The model gives only a number, so the judge treats the response as having no explicit checkable claim. However, the answer `0` implies that no person in the image was born after World War II. Unless the image contains strong identity or historical evidence, birth year cannot be inferred from ordinary visual appearance.

The error is semantic overreach. The model converts visual appearance into a historical biographical claim that the image does not support. The v2 detector misses it because its conservative final-answer treatment avoids penalizing short answers unless an explicit claim is present.

The solution is not to return to the baseline rule that any answer mismatch is hallucination. Instead, the detector should reconstruct implicit answer claims for short numbers, yes/no answers, and choices. In this case, `0` should be expanded into the implicit claim that no visible person was born after World War II, and then the judge should evaluate whether the image can support that claim.

### 11.4 Overall Disagreement Patterns

Zero-shot Judge v2 has eight disagreements on the 40-sample Human-as-Judge set: four false negatives and four false positives.

| Error pattern | Cases | Count | Main direction |
|---|---|---:|---|
| Bare-answer implicit claim missed | `ann_0002`, `ann_0014` | 2 | FN |
| Visual attribute or counting error missed | `ann_0010`, `ann_0014` | 2 | FN |
| Reasoning-final answer inconsistency missed | `ann_0020` | 1 | FN |
| Over-strict unsupported reasoning judgment | `ann_0013`, `ann_0016` | 2 | FP |
| Ambiguous estimate judged as hallucination | `ann_0015` | 1 | FP |
| Reasonable visual inference judged as over-assertion | `ann_0037` | 1 | FP |

The remaining false negatives are concentrated in MathVista. They involve implicit answer claims, visual attributes, counting, and internal consistency. The false positives mainly occur when the judge is too strict about missing reasoning, approximate estimates, or cautious visual inference.

The disagreement analysis shows that v2's conservative design improves precision, but also creates blind spots for short direct answers and visually subtle MathVista claims.

## 12. Cross-Task Claim-Level Analysis as a Bonus Item

The first bonus analysis studies whether a unified claim-level judge can apply across tasks. This is not a training-transfer setting. The detector is not trained on POPE and transferred to MathVista. Instead, the same abstract protocol is applied to both tasks:

```text
claim extraction → evidence source assignment → support-status judgment → taxonomy mapping → deterministic aggregation
```

### 12.1 Structural Coverage Across Tasks

| Dataset | Model | Prompt | Responses | Claims / response | Checkable claims / response | Unsupported claims / response | Unsupported claim ratio | Label-triggering unsupported ratio | Response hallucination rate |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| POPE | Gemini | Direct | 900 | 1.12 | 1.12 | 0.07 | 6.36% | 6.06% | 6.78% |
| POPE | Gemini | CoT | 900 | 5.57 | 5.38 | 0.37 | 6.97% | 6.70% | 21.00% |
| POPE | Qwen | Direct | 900 | 1.13 | 1.12 | 0.08 | 7.43% | 6.93% | 7.67% |
| POPE | Qwen | CoT | 900 | 6.60 | 6.49 | 0.41 | 6.27% | 5.94% | 20.00% |
| MathVista | Gemini | Direct | 514 | 2.50 | 2.34 | 0.36 | 15.28% | 11.96% | 18.87% |
| MathVista | Gemini | CoT | 514 | 8.15 | 8.07 | 0.68 | 8.49% | 7.89% | 35.02% |
| MathVista | Qwen | Direct | 514 | 1.85 | 1.66 | 0.35 | 21.34% | 14.89% | 23.15% |
| MathVista | Qwen | CoT | 514 | 6.86 | 6.83 | 0.97 | 14.15% | 13.30% | 45.53% |

The unified protocol covers both tasks structurally. No v2 detector file has missing `claim_checks`, illegal claim types, illegal support statuses, illegal evidence sources, or illegal fine labels.

However, the claim structure differs strongly by task. POPE direct responses mostly contain answer claims and object claims. POPE CoT adds object descriptions, reasoning claims, and semantic claims. MathVista direct responses already contain more reasoning and attribute claims, while MathVista CoT exposes many visual, numerical, and mathematical reasoning claims.

### 12.2 Evidence-Source Differences

POPE primarily depends on image evidence. The judge mainly checks whether the queried object is present. MathVista depends on multiple evidence sources: image, question, math rule, diagram rule, reference answer when available, and internal consistency.

This difference explains why the same judge is more stable on POPE than on MathVista. POPE tests object-level grounding with a short evidence chain. MathVista tests a longer chain from visual reading to mathematical inference to answer-frame binding.

### 12.3 Human Alignment by Dataset and Prompt

| Dataset | Prompt | Count | TP | FP | TN | FN | Precision | Recall | F1 | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MathVista | Direct | 10 | 4 | 1 | 5 | 0 | 0.8000 | 1.0000 | 0.8889 | 0.9000 |
| MathVista | CoT | 10 | 2 | 1 | 5 | 2 | 0.6667 | 0.5000 | 0.5714 | 0.7000 |
| POPE | Direct | 10 | 2 | 0 | 8 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| POPE | CoT | 10 | 2 | 1 | 7 | 0 | 0.6667 | 1.0000 | 0.8000 | 0.9000 |

The diagnostic human-alignment split confirms the task difference. POPE Direct is the most stable setting in the 40-sample validation set. MathVista CoT is the hardest setting because it combines visual attributes, diagram reasoning, mathematical rules, and long intermediate reasoning chains.

### 12.4 Fine-Label Alignment

| Fine label | Count | TP | FP | TN | FN | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `OBJ` | 40 | 2 | 0 | 38 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `ATT` | 40 | 0 | 1 | 36 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.9000 |
| `SPA` | 40 | 0 | 0 | 40 | 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| `IR` | 40 | 3 | 6 | 30 | 1 | 0.3333 | 0.7500 | 0.4615 | 0.8250 |
| `CI` | 40 | 0 | 0 | 40 | 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| `INC` | 40 | 1 | 0 | 38 | 1 | 1.0000 | 0.5000 | 0.6667 | 0.9750 |
| `SO` | 40 | 0 | 1 | 37 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.9250 |

Fine-label alignment is stricter than binary hallucination alignment. The current validation set shows strong `OBJ` alignment but weak `ATT` alignment. This is consistent with the case analysis: visual counting, chart values, and fine-grained attributes are difficult for the judge to verify reliably. `IR` has several false positives, indicating that the judge sometimes over-penalizes missing or incomplete reasoning.

### 12.5 Cross-Task Conclusion

Zero-shot Judge v2 has structural cross-task generality. The same claim-level schema runs on POPE and MathVista without schema failure. However, performance transfer is uneven. POPE success mostly demonstrates object-grounding capability. MathVista exposes harder evidence-boundary problems: visual attributes, mathematical rules, internal consistency, and implicit answer claims.

The value of the cross-task analysis is therefore not merely showing that one judge can run on two datasets. It identifies where the method needs improvement when moving from object grounding to visual reasoning.

## 13. Remote-Sensing Extension: XLRS-Bench Bonus Item

The second bonus item extends the framework toward remote sensing through XLRS-bench. This domain is relevant because remote-sensing images often contain small objects, dense spatial layouts, land-cover categories, roads, water bodies, vegetation, and high-level scene semantics. These properties create hallucination risks that differ from ordinary VQA and MathVista.

The repository is pilot-ready for XLRS-bench, but it does not claim quantitative XLRS results because the processed XLRS data and raw images are not present. This avoids fabricating results.

The existing XLRS support includes:

| Component | Purpose |
|---|---|
| `configs/datasets.yaml` | Contains `xlrs_bench -> data/processed/xlrs_eval.jsonl` |
| `src/datasets/convert_xlrs.py` | Converts XLRS-like JSONL into the unified `EvalSample` schema |
| `prompts/answer/direct_xlrs.txt` | Direct-answer prompt for XLRS |
| `prompts/answer/evidence_grounded_cot_xlrs.txt` | Evidence-grounded CoT prompt for XLRS |
| `src/pipelines/experiment_groups.py` | Registers XLRS pilot groups for two models and two prompts |
| `src/pipelines/run_stable_pipeline.py` | Allows `--dataset xlrs_bench` and reports missing data files clearly |
| `prompts/judge/zero_shot_judge.txt` | Includes XLRS-specific judge rules |

The registered XLRS pilot groups are:

```text
xlrs_pilot_xlrs_bench_gemini_direct_v1
xlrs_pilot_xlrs_bench_gemini_cot_v1
xlrs_pilot_xlrs_bench_qwen_direct_v1
xlrs_pilot_xlrs_bench_qwen_cot_v1
```

If XLRS data becomes available, it should be converted to:

```text
data/processed/xlrs_eval.jsonl
```

Then the pipeline can be run with:

```bash
python -m src.pipelines.run_stable_pipeline --experiment xlrs_pilot --dataset xlrs_bench --stage validate
python -m src.pipelines.run_stable_pipeline --experiment xlrs_pilot --dataset xlrs_bench --stage responses --chunk-size 10 --concurrency 2
python -m src.pipelines.run_stable_pipeline --experiment xlrs_pilot --dataset xlrs_bench --stage detectors --chunk-size 10 --concurrency 2 --detector-provider gpt54_local
```

The expected XLRS hallucination analysis would focus on:

- `OBJ`: buildings, roads, vehicles, ships, water bodies, vegetation, and land-cover categories;
- `ATT`: object count, size, texture, land-cover attribute, or visual property errors;
- `SPA`: spatial layout, adjacency, direction, and relative-location errors;
- `SO`: unsupported land-use, function, intention, military, or industrial interpretation;
- `IR`: invalid inference from remote-sensing evidence to high-level scene conclusions.

XLRS-bench is therefore a natural domain extension for the unified claim-level judge. It keeps the same detection protocol while testing a new evidence boundary: fine-grained remote-sensing grounding and high-level semantic overreach.

## 14. Method Limitations

### 14.1 The Judge Still Depends on Multimodal Perception

Zero-shot Judge v2 uses an MLLM as the judge. The claim-level schema makes the judgment more transparent, but it does not guarantee that the judge correctly perceives the image. When a response states a wrong visual count or attribute, the judge may accept it if the wording sounds plausible. This is especially risky for MathVista attributes, chart values, quantities, geometry, OCR-like text, colors, slopes, and small objects.

### 14.2 `Unverifiable` Does Not Always Mean Hallucinated

Some claims lack explicit evidence but should not automatically be treated as hallucinations. A correct direct answer may not include reasoning. An approximate estimate may be acceptable if the task itself is visually ambiguous. Conversely, a confident unsupported statement may be a hallucination even if it is not directly contradicted. The detector still needs a finer distinction among contradicted claims, risky unverifiable claims, acceptable estimates, and correct but unexplained answers.

### 14.3 Short Answers Hide Implicit Claims

Direct answers such as `Yes`, `No`, `0`, or a multiple-choice option can hide a complete visual or semantic claim. V2 reduces false positives by avoiding overly aggressive final-answer checks, but this creates a blind spot. Future versions should reconstruct implicit answer claims before judging short answers.

### 14.4 CoT Changes Both Generation and Observability

CoT affects both what the model generates and what the detector can observe. It may introduce new reasoning errors, but it also exposes errors that were hidden in direct answers. Therefore, response-level hallucination-rate comparisons between direct and CoT prompts should not be interpreted as a pure prompt ablation. Claim-level exposure metrics are necessary but still cannot fully separate generation effect from observability effect.

### 14.5 Human Validation Is Diagnostic, Not Definitive

The 40-sample Human-as-Judge set satisfies the assessment requirement and is useful for disagreement analysis. However, it is too small for strong statistical claims, especially after splitting by dataset, model, prompt, and fine label. A stronger benchmark would require more samples, multiple human annotators, and inter-annotator agreement analysis.

### 14.6 Reference Answer Availability Affects Judgment

Some MathVista examples have unavailable reference answers. When the reference answer is missing, the judge cannot use it as contradiction evidence and must rely on image, question, choices, math rules, diagram rules, and internal consistency. This increases uncertainty, especially for short answers and visually ambiguous questions.

### 14.7 Taxonomy Labels Are Useful but Not Fully Mutually Exclusive

The fine labels identify the primary error source, but real cases often overlap. A chart-reading error may be both `ATT` and a cause of later `IR`. A short-answer semantic overreach may also involve attribute misreading. A final-answer inconsistency may involve language-frame binding. Therefore, label statistics should be combined with qualitative case analysis.

### 14.8 MLLM-as-a-Judge May Have Systematic Bias

The judge may show modality neglect, verbosity bias, CoT bias, or model-preference bias. It may rely too heavily on fluent text instead of re-checking the image. It may penalize short answers for missing reasoning or penalize long CoT answers for containing more borderline claims. Human-as-Judge validation is therefore not optional; it is necessary for calibrating automatic detector behavior.

## 15. Conclusion

This study builds and evaluates an automatic hallucination detector for multimodal large language models under controlled task, model, prompt, and judge settings. It defines a multimodal hallucination taxonomy covering visual factual errors, reasoning errors, internal inconsistencies, and semantic overreach. It evaluates POPE and MathVista using Gemini and Qwen-VL under direct and evidence-grounded CoT prompts. It implements a baseline zero-shot judge and an improved Zero-shot Judge v2 based on taxonomy-constrained claim-level evidence checking.

The main results show that hallucination patterns are task-dependent. POPE direct-answer errors are mainly object hallucinations, while MathVista errors are dominated by invalid reasoning and visual attribute mistakes. CoT increases response-level hallucination rates, but claim-level metrics show that much of this effect comes from increased claim exposure. Human-as-Judge validation shows that v2 is more aligned with human labels than the baseline, improving accuracy, F1, Cohen's Kappa, and MCC while reducing false positives.

The strongest remaining limitations are implicit claims in short answers, visual attribute verification, reasoning-to-answer consistency, and the subjectivity of unverifiable claims. The cross-task and XLRS-bench bonus analyses show that the unified claim-level protocol has structural generality, but performance depends on evidence-boundary complexity. Overall, Zero-shot Judge v2 should be understood as a diagnostic hallucination analysis tool: it is useful for trend analysis, model comparison, and failure-mode discovery, but it does not replace larger-scale human annotation or task-specific verification for high-stakes evaluation.

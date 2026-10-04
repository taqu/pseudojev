# Phase 4 — Multilingual Validation

## Objective

Implement Phase 4 of **pjev**.

The goal of this phase is to validate whether pjev can function as a:

> small CPU multilingual decision engine

The primary languages for this phase are:

```text
English
Japanese
```

Additional languages may be added later, but English and Japanese are the formal Phase 4 targets.

The purpose is not merely to run translated benchmark examples.

The purpose is to measure:

> language degradation

That means we need to determine how decision quality, calibration, stability, and confidence change when the same underlying decision task is presented in another language.

---

# 1. Project Context

The command name is:

```bash
pjev
```

Use `pjev` consistently in:

- documentation
- experiment commands
- dataset generation
- multilingual evaluation
- reports
- configuration artifacts

The existing architecture should remain unchanged:

```text
HTTP API
   ↓
Jev compatibility layer
   ↓
DecisionEngine
   ↓
PromptStrategy
   ↓
Calibration
   ↓
LlamaBackend
   ↓
llama.cpp
```

Do not create a separate multilingual inference path.

Multilingual evaluation must exercise the same pjev runtime and decision logic used for English.

---

# 2. Phase 4 Depends on Earlier Phases

Assume the project already has:

```text
Phase 1:
native C++ server and DecisionEngine

Phase 2:
prompt/formulation experiment infrastructure

Phase 3:
calibration pipeline and JevBench-derived tuning datasets
```

Reuse these systems.

Do not duplicate benchmark parsing, inference, calibration, or metrics code.

The multilingual work should extend the existing experiment framework.

---

# 3. Main Research Question

The main question is:

> How much does pjev performance degrade when the same semantic decision is expressed in a different language?

The primary comparison is:

```text
English
vs
Japanese
```

Do not report only absolute Japanese performance.

Always compare Japanese behavior against the corresponding English baseline.

---

# 4. Required Evaluation Dimensions

At minimum, evaluate multilingual performance across:

```text
decision accuracy
score MAE
calibration quality
candidate binding stability
prompt sensitivity
confidence shift
language degradation
```

Where existing metrics are available, reuse them.

For example:

```text
choice accuracy
noul accuracy
score MAE
NLL
Brier score
ECE
QWK
```

Do not introduce a separate metric implementation if the Phase 2/3 infrastructure already provides one.

---

# 5. Preserve Semantic Equivalence

For multilingual comparisons, the semantic task must remain the same.

Conceptually:

```text
English source example
        ↓
Japanese version
        ↓
same target semantics
```

The correct semantic answer must remain identical across languages.

For `choice`, preserve:

```text
same option semantics
same correct semantic option
```

even if wording or option length changes.

For `score`, preserve:

```text
same intended ordinal target
```

Do not allow translation or localization to change the benchmark label.

---

# 6. Source Dataset Strategy

Use the existing JevBench-derived dataset infrastructure.

Do not create an unrelated multilingual benchmark format.

Extend the derived dataset schema so that records can identify:

```text
language
source example ID
translation pair ID
source language
target language
translation provenance
```

A multilingual record should remain traceable back to the original JevBench example.

---

# 7. Paired Example Identity

English and Japanese versions of the same semantic example must have an explicit shared identity.

For example:

```json
{
  "source_id": "example-123",
  "pair_id": "example-123",
  "language": "ja"
}
```

or equivalent.

Do not rely on record ordering to identify pairs.

This is required for paired degradation analysis.

---

# 8. Translation Provenance

Record how each non-English example was produced.

For example:

```text
human translation
existing benchmark translation
machine translation
manual revision
```

Do not treat these as equivalent.

The dataset manifest should identify translation provenance.

If the repository already contains Japanese benchmark data, preserve its source rather than regenerating it unnecessarily.

---

# 9. Do Not Silently Repair Benchmark Content

If a translated example appears awkward, ambiguous, or semantically inconsistent:

- do not silently rewrite it;
- record the issue;
- preserve the original source;
- exclude it only through an explicit filtering rule.

Any manual correction must be versioned and documented.

---

# 10. Language Field in Dataset Schema

Extend derived records with explicit language metadata.

Conceptually:

```json
{
  "language": "ja",
  "source_language": "en",
  "pair_id": "..."
}
```

Use standard language identifiers consistently.

Prefer:

```text
en
ja
zh
es
de
fr
```

rather than inconsistent custom labels.

---

# 11. Primary Languages

Phase 4 must support:

```text
en
ja
```

as first-class experiment dimensions.

Optional later extensions may include:

```text
zh
es
de
fr
```

but do not delay Phase 4 completion in order to support all of them.

English and Japanese are sufficient for the initial multilingual validation gate.

---

# 12. Baseline Language

Treat English as the reference baseline.

For every important metric, report:

```text
English value
Japanese value
absolute delta
relative degradation where meaningful
```

Example:

```text
choice accuracy

English:   0.72
Japanese:  0.63
delta:    -0.09
```

Do not hide the baseline in a separate report.

---

# 13. Language Degradation Metric

Introduce explicit paired degradation reporting.

For a metric `M`:

```text
degradation = M_ja - M_en
```

or the sign-correct equivalent for lower-is-better metrics such as MAE/NLL.

Be explicit about direction.

For higher-is-better metrics:

```text
degradation =
    Japanese - English
```

For lower-is-better metrics:

```text
degradation =
    Japanese - English
```

but document that positive values mean worse performance.

Do not use an ambiguous generic "degradation score" without defining it.

---

# 14. Paired Accuracy Analysis

For paired English/Japanese examples, classify outcomes into:

```text
correct in both
correct only in English
correct only in Japanese
wrong in both
```

This is more useful than aggregate accuracy alone.

Report counts and percentages.

This helps distinguish:

```text
general model weakness
```

from:

```text
language-specific failure
```

---

# 15. Semantic Consistency Metric

Measure whether pjev produces the same semantic answer across languages.

For paired examples:

```text
English semantic prediction
vs
Japanese semantic prediction
```

Compute:

```text
cross-language semantic consistency
```

This metric should ignore candidate labels and compare semantic answers.

Example:

```text
English predicts semantic option 2
Japanese predicts semantic option 2
→ consistent
```

even if candidate labels differ.

---

# 16. Confidence Shift

Measure confidence differences between languages.

For paired examples record:

```text
English target probability
Japanese target probability

English predicted confidence
Japanese predicted confidence
```

Report distributions or summary statistics of:

```text
confidence shift
```

This is especially important because Phase 0 suggested that Japanese mechanics worked but confidence/quality signals were weaker.

---

# 17. Calibration by Language

Do not assume that the English calibration parameters automatically transfer to Japanese.

Evaluate at least these two cases:

```text
English-fitted calibration applied to English
English-fitted calibration applied to Japanese
```

Then determine whether language-specific calibration appears necessary.

If justified, allow comparison with:

```text
Japanese-fitted calibration
```

but do not immediately make language-specific calibration the default.

First measure transfer quality.

---

# 18. Calibration Transfer Experiment

The recommended experiment is:

```text
fit calibration on English tuning data
        ↓
evaluate on English validation
        ↓
evaluate same calibration on Japanese validation
```

Report:

```text
NLL
Brier
ECE
accuracy
```

for both.

This reveals whether confidence calibration transfers across languages.

---

# 19. Optional Japanese-Specific Calibration

If English calibration transfers poorly, run a controlled experiment with Japanese-specific calibration.

Conceptually:

```text
T_choice_en
T_choice_ja
```

and equivalent values for other primitives.

Do not hard-code language-specific temperatures into C++.

Store them in configuration artifacts.

---

# 20. Prompt Language Strategy

Evaluate whether prompt instructions should remain:

```text
English
```

or be localized to:

```text
Japanese
```

when user content is Japanese.

This is a controlled experiment.

At minimum consider:

```text
English instructions + Japanese content

Japanese instructions + Japanese content
```

Do not assume localized instructions are automatically better.

Measure both.

---

# 21. Separate Instruction Language From Content Language

The experiment configuration must distinguish:

```text
content_language
instruction_language
```

For example:

```yaml
content_language: ja
instruction_language: en
```

versus:

```yaml
content_language: ja
instruction_language: ja
```

Do not encode both into a single generic "language" field.

This distinction will matter for future multilingual experiments.

---

# 22. Candidate Scheme Across Languages

Evaluate whether the selected candidate scheme remains stable in Japanese.

Candidate labels such as:

```text
A/B/C/D
0/1/2/3
True/False
Yes/No
```

may behave differently depending on surrounding language.

Reuse the Phase 2 candidate-binding framework.

Do not invent a separate multilingual candidate system.

---

# 23. Natural-Language Candidates

If `natural` candidate schemes use words such as:

```text
Yes
No
True
False
```

evaluate whether:

```text
English natural labels
```

versus:

```text
Japanese natural labels
```

change behavior.

For example, if relevant:

```text
Yes / No
```

versus localized semantic equivalents.

Only use localized candidates if they remain valid single-token candidates where the constrained-logit design requires it.

Always validate tokenization.

---

# 24. Tokenization Analysis by Language

Record tokenization statistics for English and Japanese prompts.

At minimum consider:

```text
prompt token count
state token count where measurable
question token count
option token count
```

This helps determine whether degradation correlates with much longer tokenized inputs.

Do not interpret token count correlation as causation without evidence.

---

# 25. Special Token Safety

Retain all Phase 1 user-content tokenization protections.

Japanese text must pass through the same trusted/untrusted prompt construction boundary.

Do not introduce a special Japanese prompt builder that bypasses these protections.

---

# 26. Prompt Layout Experiment

Reuse the existing PromptStrategy framework.

For Japanese, compare the current selected default against at least the primary alternative if prior evidence suggests layout sensitivity.

For example:

```text
state-first
state-last
```

Do not rerun a large prompt search.

The purpose is to determine whether the English-selected prompt layout transfers to Japanese.

---

# 27. Formulation Transfer

For every important Phase 2-selected formulation choice, ask:

> Does this choice transfer from English to Japanese?

Relevant choices include:

```text
layout
candidate scheme
prompt wording
score anchors
prior correction
calibration
```

Do not independently optimize every dimension in Japanese immediately.

First measure transfer from the English configuration.

---

# 28. Baseline Transfer Configuration

Define one baseline transfer configuration:

```text
English-selected prompt
English-selected candidate scheme
English-selected prior correction
English-selected calibration
```

Apply it unchanged to Japanese.

This is the first multilingual baseline.

Only after measuring this baseline should Japanese-specific variants be tested.

---

# 29. Language-Specific Configuration

The architecture may support language-specific overrides, but do not make them mandatory.

Conceptually:

```json
{
  "default": {
    "layout": "...",
    "candidate_scheme": "...",
    "temperature": 1.2
  },
  "languages": {
    "ja": {
      "temperature": 1.3
    }
  }
}
```

Only introduce overrides where evidence supports them.

Avoid unnecessary per-language configuration complexity.

---

# 30. Per-Primitive Analysis

Always report multilingual results separately for:

```text
noul
choice
score
```

Do not collapse everything into one multilingual score.

Language degradation may differ significantly by primitive.

---

# 31. Score-Specific Analysis

Score deserves special attention.

For English and Japanese compare:

```text
MAE
QWK
expected-value distribution
confidence / entropy
ordering consistency
```

Also inspect whether the score distribution becomes:

```text
flatter
more biased
less monotonic
```

in Japanese.

Do not report only expected-value MAE.

---

# 32. Noul-Specific Analysis

For `noul`, measure:

```text
accuracy
positive prediction rate
target probability
calibration
```

Compare English and Japanese.

This is important because language-specific prior behavior may alter positive prediction rate.

---

# 33. Choice-Specific Analysis

For `choice`, measure:

```text
accuracy
semantic consistency
option-order sensitivity
candidate-position bias
confidence
```

Compare English and Japanese.

Use semantic option identity rather than candidate labels for cross-language comparison.

---

# 34. Difficulty Breakdown

Where JevBench provides or derives:

```text
easy
original
hard
```

report each language separately.

For example:

```text
English easy
Japanese easy

English original
Japanese original

English hard
Japanese hard
```

This is required because aggregate multilingual degradation may be driven mainly by harder examples.

---

# 35. Pairwise Difficulty Analysis

Where English and Japanese examples correspond to the same source item, preserve the same difficulty category.

Do not reclassify difficulty independently by language unless there is a clearly documented reason.

---

# 36. Error Transition Analysis

For each paired example, record transitions such as:

```text
EN correct → JA correct
EN correct → JA wrong
EN wrong   → JA correct
EN wrong   → JA wrong
```

Pay particular attention to:

```text
EN correct → JA wrong
```

because these are direct language-degradation cases.

---

# 37. Error Dataset

Generate a machine-readable error-analysis artifact containing paired multilingual failures.

For example:

```json
{
  "pair_id": "...",
  "primitive": "choice",
  "english": {
    "prediction": 2,
    "target": 2,
    "confidence": 0.78
  },
  "japanese": {
    "prediction": 1,
    "target": 2,
    "confidence": 0.64
  }
}
```

This should support later manual inspection.

---

# 38. Do Not Build Automatic Translation Infrastructure Unless Needed

If the Japanese benchmark data already exists, use it.

Do not build an automatic translation subsystem just because multilingual evaluation needs Japanese examples.

If translation generation is required, isolate it as dataset preparation tooling.

It must not become part of pjev runtime.

---

# 39. Machine Translation Caution

If machine translation is used to generate Japanese examples:

- record the translation method;
- freeze the generated dataset;
- do not regenerate translations on every evaluation;
- preserve source/target pairing;
- validate semantic labels.

Do not allow changing translations to become an uncontrolled experimental variable.

---

# 40. Multilingual Dataset Manifest

Extend dataset manifests with multilingual metadata.

Conceptually:

```json
{
  "languages": ["en", "ja"],
  "reference_language": "en",
  "pair_count": 500,
  "translation_provenance": "...",
  "prompt_configuration": "...",
  "calibration": "..."
}
```

Use actual values from generated artifacts.

---

# 41. Dataset Validation

Add multilingual validation rules.

At minimum verify:

```text
pair IDs are valid
language IDs are supported
targets match across paired semantic examples
option counts match where required
semantic option mapping is preserved
difficulty labels remain consistent
candidate/logit dimensions are valid
```

Do not silently accept mismatched pair semantics.

---

# 42. pjev Commands

Extend existing pjev experiment commands rather than creating standalone tools.

A workflow may conceptually look like:

```bash
pjev dataset build \
  --source jevbench \
  --language en \
  ...

pjev dataset build \
  --source jevbench \
  --language ja \
  ...

pjev experiment multilingual \
  --english ... \
  --japanese ... \
  --output ...
```

The exact CLI should follow existing project conventions.

Use the actual implemented command syntax in final documentation.

---

# 43. Paired Evaluation Command

Provide a command or evaluation mode that accepts paired datasets and generates cross-language metrics.

Conceptually:

```bash
pjev experiment multilingual \
  --reference artifacts/en \
  --target artifacts/ja
```

The evaluator should verify that pair identities match before comparison.

---

# 44. Machine-Readable Results

Store multilingual results in a machine-readable format.

At minimum include:

```text
language
primitive
difficulty
sample count
accuracy
MAE where applicable
NLL
Brier
ECE
semantic consistency
confidence statistics
degradation vs English
```

Preserve enough detail to compare future models or configurations.

---

# 45. Result Tables

Generate concise summary tables.

Example:

```text
Primitive: choice

                EN       JA       Delta
Accuracy        ...      ...      ...
NLL             ...      ...      ...
Brier           ...      ...      ...
ECE             ...      ...      ...
Consistency              ...      
```

For score:

```text
                EN       JA       Delta
MAE             ...      ...      ...
QWK             ...      ...      ...
NLL             ...      ...      ...
```

Do not merge incompatible metrics into one score.

---

# 46. Confidence Distribution

Where practical, report summary statistics such as:

```text
mean predicted confidence
mean target probability
entropy
```

for each language.

This can reveal whether Japanese degradation is primarily:

```text
wrong but confident
```

or:

```text
less certain overall
```

---

# 47. Calibration Reliability Data

Produce reliability-bin data separately by language.

For example:

```text
English reliability bins
Japanese reliability bins
```

Do not combine them before computing ECE.

---

# 48. Performance Measurements

Record inference performance by language where practical.

At minimum:

```text
prompt token count
prompt evaluation latency
```

Optionally record:

```text
total request latency
```

The purpose is to identify whether Japanese tokenization materially changes CPU cost.

Do not turn this phase into Phase 7 performance optimization.

Measurement only.

---

# 49. Do Not Optimize Runtime Yet

Do not implement:

```text
KV cache reuse
batching
thread tuning
SIMD work
GPU acceleration
```

as part of Phase 4.

If multilingual prompts are slower, record the result.

Optimization belongs to a later phase.

---

# 50. No Model Comparison Yet

Do not replace Bonsai or compare alternative models in this phase.

The purpose is to understand the multilingual behavior of the current pjev model/configuration.

Model comparison belongs to Phase 5.

---

# 51. Avoid Overfitting Japanese

Do not immediately create a large Japanese-specific prompt search.

The sequence should be:

```text
1. transfer English configuration unchanged
2. measure degradation
3. identify specific failure dimensions
4. test a small number of targeted Japanese-specific changes
5. compare against baseline
```

Do not optimize dozens of prompt variants until one happens to score well.

---

# 52. Configuration Freeze

After completing multilingual experiments, create a candidate configuration artifact that explicitly records whether settings are:

```text
shared across languages
```

or:

```text
language-specific
```

Example:

```json
{
  "default": {
    "layout": "state-last",
    "candidate_scheme": "letters"
  },
  "languages": {
    "en": {
      "temperature": 1.18
    },
    "ja": {
      "temperature": 1.31
    }
  }
}
```

Only include language-specific values if validated.

---

# 53. Recalibration Rule

If prompt formulation changes for Japanese:

```text
prompt wording
instruction language
candidate scheme
layout
```

do not reuse old calibration parameters automatically.

Generate fresh inference data and refit/revalidate calibration.

Prompt changes invalidate assumptions behind previously fitted calibration.

---

# 54. Reproducibility

Every multilingual experiment must record:

```text
model
model hash
pjev revision
llama.cpp revision
dataset version
language
translation provenance
prompt config
instruction language
candidate scheme
prior correction
calibration artifact
```

The same experiment should be reproducible later.

---

# 55. Tests

Add tests for:

```text
language metadata parsing
paired example matching
semantic target preservation
cross-language semantic consistency
language degradation calculations
confidence-delta calculations
multilingual result serialization
language-specific calibration lookup
shared/default calibration fallback
multilingual manifest validation
```

---

# 56. Pair Mapping Tests

For choice examples, verify:

```text
English semantic option N
```

and:

```text
Japanese semantic option N
```

refer to the same answer semantics.

Candidate labels may differ.

Semantic identity must not.

---

# 57. Calibration Transfer Tests

Add synthetic or fixture tests verifying:

```text
English calibration artifact
        ↓
Japanese evaluation
```

is supported without accidentally selecting Japanese-specific values.

Also test explicit language-specific overrides.

---

# 58. Unsupported Language Behavior

If a language is not configured, behavior must be deterministic.

Choose and document one clear policy, such as:

```text
use default/shared configuration
```

or:

```text
fail if explicit language config is required
```

Do not silently select arbitrary language-specific settings.

---

# 59. Documentation

Document:

```text
how multilingual datasets are represented
how EN/JA pairs are identified
how to build English derived data
how to build Japanese derived data
how to run paired evaluation
how calibration transfer is evaluated
how language-specific configuration works
```

Use `pjev` in all commands.

---

# 60. Required Experimental Sequence

Run experiments in this order unless the actual repository structure makes a minor adjustment necessary.

## Experiment 1 — Baseline English

Run the frozen Phase 3 configuration on English.

Establish:

```text
choice accuracy
noul accuracy
score MAE
NLL
Brier
ECE
```

---

## Experiment 2 — Direct Japanese Transfer

Apply exactly the same:

```text
prompt strategy
candidate scheme
prior correction
calibration
```

to Japanese.

Measure degradation.

Do not modify the configuration first.

---

## Experiment 3 — Paired Error Analysis

Identify:

```text
EN correct → JA wrong
```

cases.

Measure semantic consistency and confidence shift.

---

## Experiment 4 — Instruction Language

If useful, compare:

```text
English instruction + Japanese content
```

against:

```text
Japanese instruction + Japanese content
```

Keep other variables fixed.

---

## Experiment 5 — Candidate/Formulation Transfer

Test only the highest-value formulation alternatives identified by the earlier analysis.

Avoid broad searches.

---

## Experiment 6 — Calibration Transfer

Evaluate:

```text
English-fitted calibration
```

on Japanese.

If it transfers poorly, fit a Japanese-specific calibration and compare.

---

# 61. Phase 4 Exit Criteria

Phase 4 is complete when:

```text
[ ] English and Japanese are supported as explicit experiment languages.

[ ] English/Japanese semantic pairs can be identified reliably.

[ ] The same DecisionEngine path is used for both languages.

[ ] The Phase 3 English configuration can be transferred unchanged to
    Japanese for baseline evaluation.

[ ] Choice accuracy is reported by language.

[ ] Noul accuracy is reported by language.

[ ] Score MAE is reported by language.

[ ] Calibration metrics are reported by language.

[ ] Cross-language semantic consistency is measured.

[ ] Confidence shift is measured.

[ ] EN-correct → JA-wrong transitions are measurable.

[ ] Difficulty-specific degradation is reported where available.

[ ] Prompt token counts and basic latency can be compared by language.

[ ] English calibration transfer to Japanese has been evaluated.

[ ] Japanese-specific calibration is evaluated only if justified.

[ ] Prompt/instruction-language transfer has been evaluated where useful.

[ ] All multilingual experiment artifacts include provenance metadata.

[ ] Multilingual dataset validation passes.

[ ] Documentation contains reproducible pjev commands.

[ ] No model replacement work has been introduced.
```

---

# 62. Phase 4 Decision Output

At the end of this phase, we should be able to answer:

```text
How large is the English → Japanese degradation?

Which primitive degrades most?

Is the degradation mostly accuracy, calibration, or confidence?

Does the English prompt formulation transfer?

Does the English candidate scheme transfer?

Does English-fitted calibration transfer?

Are Japanese-specific settings actually necessary?

Does Japanese materially increase prompt evaluation cost?
```

Do not reduce the answer to a single multilingual score.

---

# 63. Final Implementation Report

When implementation is complete, provide the following.

## Dataset

Document:

```text
English example count
Japanese example count
paired example count
translation provenance
excluded/unpaired examples
```

## Baseline Configuration

Show the exact frozen English configuration used for direct transfer.

## Results by Primitive

Report separately for:

```text
noul
choice
score
```

## Results by Language

For every relevant metric show:

```text
English
Japanese
delta
```

## Difficulty Breakdown

Where available:

```text
easy
original
hard
```

## Cross-Language Consistency

Report:

```text
same semantic prediction rate
EN correct → JA wrong
EN wrong → JA correct
```

## Calibration

Report:

```text
English calibration on English
English calibration on Japanese
Japanese-specific calibration if tested
```

Include:

```text
NLL
Brier
ECE
```

## Prompt Transfer

Report each tested:

```text
instruction language
layout
candidate scheme
```

and its result.

Do not report only the selected variant.

## Performance

Report:

```text
token counts
prompt evaluation latency
```

by language where collected.

## Selected Configuration

Show whether the final candidate configuration is:

```text
language-independent
```

or requires:

```text
Japanese-specific overrides
```

Do not hard-code the result into source.

## Commands

Provide exact `pjev` commands used to reproduce all important experiments.

## Tests

Provide exact test commands and test results.

## Known Limitations

Document:

```text
translation quality limitations
dataset-size limitations
benchmark-specific effects
remaining Japanese weaknesses
calibration-transfer limitations
```

---

# Final Instruction

Phase 4 is not a translation feature project.

It is a controlled experiment answering:

> How much of pjev's decision quality survives a language change?

Keep the English configuration as the reference.

Measure transfer before introducing language-specific changes.

Use paired semantic examples wherever possible.

Separate:

```text
accuracy degradation
calibration degradation
confidence degradation
runtime/tokenization differences
```

and report them independently.

The deliverable is a reproducible multilingual validation pipeline and clear evidence about whether pjev can credibly be described as a small CPU multilingual decision engine.
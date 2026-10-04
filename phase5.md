# Phase 5 — Model Value Gate

## Objective

Implement Phase 5 of **pjev**.

This phase answers one concrete question:

> Is Bonsai 1.7B good enough, efficient enough, and robust enough to remain the default model for pjev?

This is the first phase where model comparison is allowed.

Do not begin by replacing Bonsai.

First establish a complete, reproducible value profile for the current Bonsai-based pjev configuration.

Only compare alternative models if the evidence shows that comparison is necessary.

---

# 1. Project Context

The executable and command namespace is:

```bash
pjev
```

Use `pjev` consistently in:

- commands
- reports
- benchmark artifacts
- model evaluation tooling
- configuration
- documentation

The current architecture remains:

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
   ↓
GGUF model
```

Do not create a separate inference implementation for model evaluation.

Every model must be evaluated through the same pjev runtime path.

---

# 2. Phase 5 Inputs

Assume previous phases have already produced:

```text
Phase 2:
selected decision formulation

Phase 3:
calibration pipeline and fitted calibration artifacts

Phase 4:
English/Japanese multilingual validation
```

Use the frozen or selected configuration from those phases as the baseline.

Do not retune prompt formulation independently for every candidate model unless explicitly required by the experiment.

The first comparison should measure:

> model effect under the same pjev decision formulation.

---

# 3. Primary Evaluation Dimensions

Evaluate each model across:

```text
quality
calibration
latency
RSS
model size
multilingual quality
```

These dimensions must remain separate.

Do not collapse them into a single opaque score.

The purpose of this phase is to expose tradeoffs.

---

# 4. Baseline Model

The baseline model is the current Bonsai 1.7B GGUF used by pjev.

Before testing alternatives, generate a complete baseline report.

Record:

```text
exact model name
model source
GGUF filename
GGUF file size
model hash
quantization
architecture metadata
context configuration
thread configuration
llama.cpp revision
pjev revision
```

Do not refer to the model only as "Bonsai 1.7B" if multiple files or quantizations exist.

---

# 5. Preserve the Existing pjev Configuration

The baseline model evaluation should use the selected:

```text
prompt layout
candidate scheme
candidate prior correction
calibration parameters
language configuration
```

from previous phases.

Freeze these inputs before running the gate.

This prevents Phase 5 from becoming another prompt-search phase.

---

# 6. Model Comparison Policy

Do not compare models until the Bonsai baseline is complete.

If alternatives are evaluated, constrain the initial candidate set to approximately:

```text
0.5B–2B parameter models
```

or the practical size class that matches pjev's product goal.

Do not compare against very large models merely to maximize benchmark quality.

The relevant question is:

> Can another small CPU-friendly model provide meaningfully better value than Bonsai?

---

# 7. Same Runtime Path

All candidate models must use:

```text
pjev
    ↓
LlamaBackend
    ↓
llama.cpp
```

Do not benchmark one model through another runtime.

Do not use Python Transformers results as substitutes for pjev measurements.

The purpose is to measure actual product behavior.

---

# 8. Model Compatibility Check

Before full evaluation, run a compatibility probe.

Validate:

```text
GGUF loads successfully
tokenizer works
chat template behavior is understood
required candidate labels can be represented
single-token candidate constraints remain valid
logit extraction works
context creation works
```

A model that cannot support the constrained-logit design cleanly should be marked incompatible before expensive evaluation.

---

# 9. Candidate Token Compatibility

Candidate tokenization may differ by model.

Do not assume that:

```text
A
B
C
D
Yes
No
digits
```

have the same tokenization across models.

For every candidate model, record:

```text
candidate text
token ID
token count
valid single-token status
```

If the selected candidate scheme is invalid for a model, report this explicitly.

Do not silently switch candidate schemes during the primary comparison.

---

# 10. Two-Stage Model Evaluation

Use two stages.

## Stage A — Compatibility / Screening

Cheap checks:

```text
model load
tokenization
candidate compatibility
small smoke test
basic latency
basic RSS
```

Models that clearly fail product constraints can stop here.

## Stage B — Full Value Evaluation

Run the complete benchmark and multilingual suite only for viable candidates.

This avoids wasting CPU time on obviously unsuitable models.

---

# 11. Quality Metrics

Reuse the established pjev/JevBench metrics.

At minimum report:

```text
noul accuracy
choice accuracy
score accuracy
score MAE
score QWK
```

where applicable.

Also preserve:

```text
easy
original
hard
```

breakdowns where available.

Do not report only aggregate accuracy.

---

# 12. Calibration Metrics

For each model report:

```text
NLL
Brier score
ECE
```

separately from accuracy.

Use the calibration methodology established in Phase 3.

Do not assume calibration parameters fitted for Bonsai transfer to another model.

---

# 13. Calibration Policy for Candidate Models

For each alternative model, evaluate in two useful modes where practical:

```text
uncalibrated
model-specific fitted calibration
```

Do not apply Bonsai temperatures as though they were universal.

If time or dataset constraints require one mode only, prefer fitting calibration using the same tuning procedure used for Bonsai.

Document the choice.

---

# 14. Prompt Policy Across Models

Start each model with the same semantic formulation selected for Bonsai.

Do not immediately optimize prompts per model.

The initial comparison should answer:

> How does this model perform under the same decision formulation?

If a model appears promising but clearly suffers from a tokenizer or chat-template mismatch, a small model-specific formulation adjustment may be evaluated later.

Document such adjustments separately.

---

# 15. Avoid Unfair Per-Model Tuning

Do not heavily tune one model while leaving others at defaults.

If prompt adjustments are allowed, use the same tuning budget.

For example:

```text
same number of prompt variants
same calibration dataset
same validation rules
same stopping criteria
```

The comparison must remain interpretable.

---

# 16. English Evaluation

Run the established English benchmark suite.

Report at least:

```text
noul
choice
score
calibration
difficulty breakdown
```

Use the same dataset partitions and evaluation procedure established earlier.

---

# 17. Japanese Evaluation

Run the established Japanese multilingual validation suite.

Report:

```text
noul
choice
score
calibration
cross-language degradation
```

Do not compare English quality for one model against Japanese quality for another.

Each model needs its own EN/JA paired evaluation.

---

# 18. Language Degradation

For each model, calculate:

```text
English → Japanese degradation
```

using the Phase 4 definitions.

This is a key product dimension.

A model with slightly higher English accuracy but substantially worse Japanese transfer may not be a better pjev default.

Do not reduce multilingual evaluation to Japanese accuracy alone.

---

# 19. Latency Measurement

Measure real pjev latency through the native llama.cpp path.

At minimum capture:

```text
model load latency
prompt evaluation latency
end-to-end decision latency
```

Report distributions where practical:

```text
median
p90
p95
```

If the experiment count is too small for meaningful percentiles, report that limitation.

---

# 20. Separate Cold and Warm Latency

Do not mix:

```text
model load
```

with:

```text
warm request inference
```

Measure them separately.

At minimum report:

```text
cold startup / model load
warm typical request
```

This distinction is important because future background runtime support can hide load latency but not per-request prompt evaluation cost.

---

# 21. Representative Prompt Sizes

Measure latency using representative pjev workloads.

Do not benchmark only trivial prompts.

Include examples approximating:

```text
short
typical
long
```

prompt sizes if the benchmark dataset naturally provides them.

Record token counts.

---

# 22. RSS Measurement

Measure resident memory usage.

At minimum report:

```text
post-model-load RSS
peak RSS during representative inference
```

Use a consistent methodology across models.

Document the platform and measurement mechanism.

Do not compare RSS numbers collected using incompatible methods.

---

# 23. Model File Size

Record:

```text
GGUF file size
quantization
parameter count where known
```

File size is a direct product/distribution cost and must be part of the gate.

Do not infer parameter count from filename if metadata is available.

---

# 24. CPU Configuration

Record hardware/runtime configuration for every performance run.

At minimum:

```text
OS
CPU model
architecture
physical/logical core count where available
pjev thread setting
llama.cpp backend
```

Do not compare latency numbers from different machines without labeling them clearly.

---

# 25. Deterministic Benchmark Configuration

Keep inference settings fixed across models where technically valid.

Record:

```text
context size
thread count
batch settings
candidate formulation
prompt layout
```

Do not let llama.cpp auto-settings create hidden model-specific differences unless unavoidable.

---

# 26. Quantization

Treat quantization as part of model identity.

For example:

```text
model X Q4_K_M
```

and:

```text
model X Q8_0
```

are distinct evaluation candidates.

Do not merge results from different quantizations.

---

# 27. Quantization Search Scope

Do not perform a huge quantization sweep.

Start with the quantization class that best matches the current Bonsai deployment target.

Only test additional quantizations when there is a clear reason, such as:

```text
memory reduction
quality recovery
latency tradeoff
```

Keep the search small.

---

# 28. Value Profile Artifact

Generate a machine-readable artifact for every evaluated model.

Conceptually:

```json
{
  "model": {
    "name": "...",
    "hash": "...",
    "size_bytes": 0,
    "quantization": "..."
  },
  "quality": {
    "choice_accuracy": 0.0,
    "noul_accuracy": 0.0,
    "score_mae": 0.0
  },
  "calibration": {
    "nll": 0.0,
    "brier": 0.0,
    "ece": 0.0
  },
  "multilingual": {
    "ja_choice_accuracy": 0.0,
    "choice_degradation": 0.0
  },
  "performance": {
    "model_load_ms": 0.0,
    "warm_p50_ms": 0.0,
    "warm_p95_ms": 0.0,
    "rss_bytes": 0
  }
}
```

Exact schema may differ.

The important requirement is comparability.

---

# 29. Model Manifest

Maintain a model manifest or equivalent metadata registry for evaluated models.

Record:

```text
model ID
source
license information if already available in project metadata
GGUF path
hash
quantization
evaluation status
```

Do not scatter model paths throughout scripts.

---

# 30. pjev Model Evaluation Command

Add model evaluation under the `pjev` namespace.

Conceptually:

```bash
pjev model evaluate \
  --model /path/to/model.gguf \
  --config config.json \
  --output artifacts/model-eval/
```

or:

```bash
pjev benchmark model ...
```

Use whichever command structure matches the existing CLI.

Do not create unrelated standalone binaries.

---

# 31. Screening Command

If useful, support a cheaper command such as:

```bash
pjev model inspect ...
```

or:

```bash
pjev model probe ...
```

to report:

```text
model metadata
candidate tokenization
model load success
basic memory
basic smoke test
```

This should not replace full evaluation.

---

# 32. Reuse Cached Derived Data Carefully

Post-logit calibration experiments may reuse stored logits.

But model comparison requires fresh inference for each model.

Never reuse Bonsai logits for another model.

Each model needs its own derived evidence dataset.

---

# 33. Derived Dataset Identity

Include model identity in every derived dataset manifest.

At minimum:

```text
model hash
quantization
pjev commit
prompt config
calibration config
```

A dataset generated by one model must never be mistaken for another model's data.

---

# 34. Benchmark Dataset Consistency

All compared models must use the same benchmark examples.

If an example fails for one model because of candidate-token incompatibility, report that failure.

Do not silently remove it from only that model's denominator.

Produce both:

```text
full-set compatibility statistics
```

and, if needed:

```text
common-valid-subset comparison
```

with clear labeling.

---

# 35. Common Valid Subset

If different tokenizers make some candidate schemes invalid, create a common-valid-subset analysis where useful.

But do not replace the full compatibility result with it.

Report:

```text
full dataset support rate
common-valid-subset quality
```

separately.

---

# 36. Correctness Before Performance

A faster model is not useful if it breaks the constrained-logit mechanism.

Evaluation priority should be:

```text
compatibility
quality
calibration
multilingual behavior
performance
distribution cost
```

Do not select models based only on latency.

---

# 37. No Composite Score by Default

Do not invent a weighted score such as:

```text
0.4 * accuracy
+ 0.3 * latency
+ 0.3 * memory
```

unless explicitly requested later.

Different product tradeoffs should remain visible.

Produce a comparison table instead.

---

# 38. Comparison Table

Generate a concise table like:

```text
Model       Choice  Noul   Score MAE  JA Δ   NLL   Warm ms  RSS   Size
Bonsai      ...     ...    ...        ...    ...   ...      ...   ...
Model B     ...     ...    ...        ...    ...   ...      ...   ...
```

Do not label one row as "winner."

The evidence should make tradeoffs visible without hiding dimensions.

---

# 39. Baseline Historical Measurements

Previous Phase 0 measurements suggested approximately:

```text
model load      ~440 ms
RSS             ~940 MiB
typical prompt  140–420 ms
decision math   ~0.01 ms
```

Do not hard-code these as current truth.

Re-measure using the current Phase 5 implementation and hardware.

Historical values are useful only as a sanity check.

---

# 40. Prompt Evaluation Cost

Explicitly record the proportion of runtime attributable to:

```text
prompt evaluation
```

versus:

```text
decision math / calibration
```

where practical.

Do not over-engineer profiling.

Simple measured timing boundaries are sufficient.

---

# 41. Alternative Model Search

If Bonsai clearly fails the value gate, evaluate a small set of alternatives.

Prefer candidates that satisfy:

```text
CPU-friendly
GGUF-supported
approximately 0.5B–2B
multilingual-capable or plausibly multilingual
usable with constrained candidate logits
```

Do not search the entire model ecosystem.

A small, evidence-driven shortlist is enough.

---

# 42. Model Acquisition

Keep downloaded model artifacts outside source control unless the repository already has a different policy.

Record:

```text
source
version/revision
hash
```

Do not commit large GGUF files accidentally.

---

# 43. License Awareness

Record known model license metadata when available from existing model artifacts or project configuration.

Do not perform a legal interpretation.

Simply preserve provenance and license identifiers needed for later product review.

---

# 44. Model-Specific Chat Templates

Different models may use different chat templates.

Do not force the Bonsai chat template onto another model if llama.cpp/model metadata provides a different required template.

However, separate:

```text
chat-template compatibility
```

from:

```text
semantic pjev prompt formulation
```

The same semantic PromptStrategy should be mapped into the model's appropriate chat representation.

Document deviations.

---

# 45. Tokenization Safety

Retain Phase 1 special-token protections for every model.

User-controlled content must remain ordinary user content.

Do not disable safe tokenization merely because another model uses different special tokens.

Add model-specific fixtures if necessary.

---

# 46. Calibration Refit Requirement

For every candidate model promoted to full evaluation:

```text
generate model-specific logits
fit model-specific calibration
evaluate on validation
```

Do not reuse Bonsai calibration parameters.

The fitting procedure itself should remain identical.

---

# 47. Prompt Retuning Gate

Only consider model-specific prompt retuning if:

```text
the model is otherwise promising
AND
the shared formulation causes an obvious compatibility or representation problem
```

If retuning occurs, clearly distinguish:

```text
shared-config result
```

from:

```text
model-tuned result
```

Do not mix them.

---

# 48. Multilingual Retuning Gate

Likewise, do not create model-specific Japanese prompt variants before measuring direct transfer.

Use the Phase 4 sequence:

```text
same English-selected formulation
        ↓
Japanese transfer
        ↓
measure degradation
```

before introducing overrides.

---

# 49. Repeated Measurements

Performance measurements should include multiple runs.

Discard or separately report obvious startup/warmup effects.

Use enough samples to estimate stable central tendency.

Do not report a single request latency as representative performance.

---

# 50. Output Reproducibility

Every model evaluation artifact must contain enough information to reproduce the run.

Record:

```text
pjev commit
llama.cpp revision
model hash
model config
prompt config
calibration config
dataset version
hardware info
thread settings
```

---

# 51. Regression Tests

Add tests for:

```text
model metadata loading
model identity hashing
candidate token compatibility checks
evaluation artifact serialization
performance result serialization
model-specific calibration selection
common-valid-subset handling
```

Do not make unit tests depend on several gigabytes of model files.

Use fixtures/test doubles where possible.

---

# 52. Real-Model Smoke Tests

For each model promoted to full evaluation, verify at least one real inference for:

```text
noul
choice
score
```

before launching the complete benchmark.

Fail early on incompatible models.

---

# 53. Failure Reporting

Do not silently exclude model failures.

Report:

```text
load failures
tokenization incompatibility
invalid candidates
inference failures
unsupported chat template
out-of-memory failures
```

A model that cannot complete the pjev workload reliably has failed an important part of the value gate.

---

# 54. Bonsai Value Gate

The final Bonsai assessment must summarize evidence across:

```text
quality
calibration
English/Japanese behavior
latency
memory
model size
compatibility
```

Do not make the decision based solely on benchmark accuracy.

The report should make clear where Bonsai is strong and where it is weak.

---

# 55. Decision Artifact

Produce a versioned Phase 5 decision artifact.

Conceptually:

```json
{
  "phase": 5,
  "baseline_model": "...",
  "evaluated_models": [
    "..."
  ],
  "selected_default_model": "...",
  "evidence_artifacts": [
    "..."
  ],
  "notes": [
    "..."
  ]
}
```

If the project chooses to retain Bonsai, record that.

If another model is selected, record that.

Do not encode the decision only in documentation prose.

---

# 56. Model Selection Criteria

Do not hard-code arbitrary thresholds unless the project already defines them.

Instead, compare the candidate's practical tradeoffs against the product goal:

> small CPU multilingual decision engine

Relevant considerations include:

```text
meaningful quality gain
calibration quality
Japanese degradation
latency cost
RSS cost
distribution size
candidate compatibility
implementation complexity
```

Keep the evidence explicit.

---

# 57. Do Not Start Production Optimization

Phase 5 is not Phase 7.

Do not implement:

```text
KV reuse optimization
batch scheduling
aggressive thread tuning
custom kernels
platform-specific SIMD optimization
```

Measure current performance consistently.

Performance optimization comes after the model value decision.

---

# 58. Do Not Start Distribution Engineering

Do not implement:

```text
model embedding
single-binary packaging
automatic extraction
download manager
cross-platform installer
```

in this phase.

Model size should be measured, not solved.

Distribution work comes later.

---

# 59. Required Evaluation Sequence

Use this sequence.

## Step 1 — Freeze pjev Formulation

Record the selected:

```text
PromptStrategy
candidate scheme
prior correction
calibration procedure
language settings
```

---

## Step 2 — Re-evaluate Bonsai

Generate a fresh full baseline using the current code.

Do not rely only on old Phase 0 numbers.

---

## Step 3 — Produce Bonsai Value Profile

Measure:

```text
quality
calibration
EN/JA multilingual behavior
latency
RSS
model size
```

---

## Step 4 — Evaluate Whether Comparison Is Necessary

If Bonsai clearly satisfies the intended value profile, do not create a large model-search project.

If significant weaknesses remain, shortlist a small number of alternatives.

---

## Step 5 — Screen Alternatives

Check:

```text
loadability
candidate compatibility
basic quality
basic latency
basic RSS
```

---

## Step 6 — Full Evaluation

For viable candidates, run the same complete evaluation pipeline used for Bonsai.

---

## Step 7 — Fit Model-Specific Calibration

Use the same calibration methodology and tuning split.

---

## Step 8 — Compare Multilingual Transfer

Run English/Japanese paired validation.

---

## Step 9 — Produce Final Evidence Table

Present all dimensions without collapsing them into one score.

---

## Step 10 — Freeze Default Model Decision

Produce a versioned decision/config artifact.

---

# 60. Definition of Done

Phase 5 is complete when:

```text
[ ] Bonsai has a complete current value profile.

[ ] Exact Bonsai GGUF identity and hash are recorded.

[ ] Quality metrics are measured.

[ ] Calibration metrics are measured.

[ ] English/Japanese behavior is measured.

[ ] Language degradation is measured.

[ ] Model load latency is measured.

[ ] Warm inference latency is measured.

[ ] RSS is measured.

[ ] GGUF size is recorded.

[ ] Candidate token compatibility is documented.

[ ] Measurements use the native pjev/llama.cpp path.

[ ] Benchmark configuration is reproducible.

[ ] Performance environment is recorded.

[ ] Alternative models are evaluated only if justified.

[ ] Any candidate model uses fresh model-specific inference data.

[ ] Any candidate model uses model-specific calibration.

[ ] Candidate models are compared on the same datasets.

[ ] Full-set incompatibilities are reported.

[ ] A common-valid-subset analysis exists if needed.

[ ] A machine-readable value profile exists for every fully evaluated model.

[ ] A comparison report exists.

[ ] A versioned default-model decision artifact exists.

[ ] No Phase 7 performance optimization has been mixed into the evaluation.
```

---

# 61. Final Implementation Report

When implementation is complete, provide the following sections.

## Baseline Model

Report:

```text
model name
GGUF file
hash
quantization
size
```

## Runtime Environment

Report:

```text
OS
CPU
architecture
thread settings
llama.cpp revision
pjev revision
```

## Frozen pjev Configuration

Report:

```text
prompt strategy
candidate scheme
prior correction
calibration methodology
multilingual configuration
```

## Bonsai Quality

Report:

```text
noul accuracy
choice accuracy
score accuracy
score MAE
QWK
easy/original/hard breakdown
```

## Bonsai Calibration

Report:

```text
NLL
Brier
ECE
fitted temperatures
```

## Bonsai Multilingual Results

Report:

```text
English metrics
Japanese metrics
language degradation
semantic consistency
```

## Bonsai Performance

Report:

```text
model load time
warm latency
prompt token counts
RSS
model file size
```

## Alternative Models

If evaluated, list every candidate and explain why it entered the shortlist.

For each candidate report the same metrics.

Do not omit failed candidates.

## Compatibility Issues

Report:

```text
tokenization problems
candidate incompatibilities
chat-template issues
load/inference failures
```

## Comparison

Provide a compact evidence table covering:

```text
quality
calibration
multilingual behavior
latency
RSS
size
```

Do not reduce the comparison to a single numerical score.

## Default Model Decision

State which model is frozen as the pjev default for the next phase.

Reference the evidence artifact used for that decision.

## Commands

Provide exact working `pjev` commands used for:

```text
model inspection
benchmark generation
calibration fitting
multilingual evaluation
performance measurement
final comparison
```

## Tests

Provide exact test commands and outcomes.

## Remaining Risks

Document:

```text
benchmark overfitting
hardware-specific performance
multilingual weaknesses
calibration limitations
model license/provenance considerations
```

---

# Final Instruction

Phase 5 is a value gate, not a model leaderboard.

Do not ask:

> Which model gets the highest benchmark score?

Ask:

> Which small model gives pjev the best evidence-backed balance of decision quality, calibration, multilingual behavior, CPU latency, memory usage, model size, and implementation compatibility?

Preserve the formulation improvements from earlier phases.

Evaluate the actual native pjev runtime.

Do not hide tradeoffs behind a composite score.

The output of Phase 5 should be a defensible, reproducible decision about which model pjev should carry forward into production-server and optimization work.
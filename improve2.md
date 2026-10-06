# Task: Implement a JevBench-Derived Calibration Pipeline for `pjev`

## Objective

Build a reproducible calibration and evaluation pipeline for `pjev` using JevBench-derived data.

The purpose of this work is to determine how much of the current decision-quality problem comes from:

- candidate-token prior bias,
- prompt formulation,
- probability miscalibration,
- versus the underlying model capability.

Do not treat JevBench as a single benchmark score only.

The system should support:

```text
JevBench dataset
    Å´
pjev inference
    Å´
raw candidate logits
    Å´
derived tuning dataset
    Å´
post-logit calibration
    Å´
validation-based parameter selection
    Å´
locked configuration
    Å´
held-out evaluation
```

The implementation must make it possible to rerun calibration without rerunning llama.cpp inference whenever the prompt and model configuration have not changed.

---

# Core design principle

Separate two classes of experiments.

## 1. Prompt / formulation experiments

These require inference to be rerun.

Examples:

```text
candidate representation:
  A/B/C/D
  Yes/No
  True/False

prompt layout:
  state Å® question Å® options
  question Å® options Å® state
  question Å® state Å® options

assistant suffix:
  chat template with think block
  chat template without think block
  explicit "Answer:"
  raw classification prompt

other wording changes
```

## 2. Post-logit calibration

These operate only on stored logits.

Examples:

```text
temperature scaling
candidate prior correction
prior correction strength
probability recalculation
threshold diagnostics
```

Do not mix these two layers.

A prompt change invalidates stored logits.

A calibration parameter change does not.

---

# Required output dataset

Create a reproducible derived dataset from JevBench inference results.

Use a line-oriented format such as JSONL unless an existing project format is already clearly preferable.

Each record must contain at least:

```text
source_id
split
primitive
difficulty

state
question
options

ground_truth

prompt_config
candidate_scheme
candidate_mapping
candidate_token_ids

raw_logits
prior_logits
corrected_logits

raw_probs
calibrated_probs

prediction
calibrated_prediction

model identifier
model hash if available

prompt token count
inference latency if already available
```

Where practical, also include:

```text
temperature
prior_alpha
prompt template identifier
layout identifier
assistant suffix identifier
```

The stored raw logits must be sufficient to recompute post-logit calibration without running llama.cpp again.

---

# Dataset split requirements

This is critical.

Do not tune and evaluate on the same examples.

Create deterministic splits:

```text
tuning
validation
held_out
```

The split must be reproducible from a fixed seed or stable source IDs.

The held-out split must not influence:

```text
temperature selection
prior correction strength
prompt selection
candidate-scheme selection
threshold selection
```

The final held-out metrics must only be computed after all parameters and prompt choices are frozen.

If JevBench already exposes a meaningful split structure, preserve it where appropriate rather than inventing a conflicting one.

Record the split assignment in the derived dataset.

---

# Phase A: Raw inference collection

Implement an inference runner that executes JevBench examples through the real `pjev` decision path.

Do not create a separate toy decision implementation.

The runner should use the same:

```text
PromptStrategy
DecisionEngine
LlamaBackend
candidate token validation
restricted candidate logits
```

as production inference.

For every sample, capture the raw candidate logits before:

```text
prior correction
temperature scaling
softmax calibration
```

The runner must preserve the semantic mapping between:

```text
ground-truth answer
candidate
internal label
token ID
raw logit
```

This is especially important for `noul`.

Never infer semantic `true` from candidate position.

---

# Phase B: Candidate prior measurement

Implement content-free or minimally informative candidate-prior measurement.

The goal is to estimate token / prompt-format preference independently of task content.

For each relevant configuration, measure candidate logits from a content-free prompt.

For example, conceptually:

```text
State:

Question:

Possible answers:
A: No
B: Yes
```

The exact blank prompt should be generated using the same PromptStrategy path as normal inference.

Do not manually construct a different prompt unless explicitly required.

Store the resulting candidate prior logits.

At minimum, measure priors separately for:

```text
primitive
candidate scheme
prompt configuration
number of candidates
```

If later evidence shows that additional conditioning is needed, keep the design extensible.

---

# Phase C: Prior correction

Implement configurable candidate prior correction.

Use a parameter:

```text
alpha
```

with the general form:

```text
corrected_logit_i =
    raw_logit_i - alpha * prior_logit_i
```

where:

```text
alpha = 0
```

means no correction.

Support at least:

```text
alpha Å∏ continuous or fine-grid range
```

For an initial implementation, a deterministic grid search is acceptable.

Example search range:

```text
0.0 to 2.0
```

with a reasonably fine step.

Do not hard-code this range deep inside the engine. Put it in the calibration / tuning layer.

For diagnostics, retain both:

```text
raw_logits
corrected_logits
```

---

# Phase D: Temperature scaling

Implement standard temperature scaling over candidate logits.

Use:

```text
P_i =
exp(corrected_logit_i / T)
/
sum_j exp(corrected_logit_j / T)
```

with:

```text
T > 0
```

Fit temperature by minimizing negative log-likelihood on the tuning split.

Support primitive-specific temperatures:

```text
T_noul
T_choice
T_score
```

Do not assume one global temperature is optimal.

Start with independent primitive-level temperatures.

If a primitive has too little data, fail safely or fall back to a documented default rather than silently overfitting.

---

# Special handling for `noul`

For binary `noul`, expose and store the semantic margin:

```text
margin =
    logit_true - logit_false
```

Also store:

```text
raw_margin
prior_corrected_margin
calibrated_p_true
```

For diagnostics, compute:

```text
P(true)
```

using the candidate that is semantically bound to true.

Do not assume:

```text
index 0 = true
```

or:

```text
index 1 = true
```

The semantic mapping must come from candidate metadata.

---

# Optimization procedure

Use the tuning split to fit calibration parameters.

A good initial search procedure is:

```text
for each primitive:
    search alpha
        fit temperature T
        compute tuning NLL

select promising parameter sets

evaluate them on validation

freeze the best validation configuration

evaluate exactly once on held_out
```

Do not select final parameters from held-out metrics.

Do not optimize directly for accuracy only.

Primary calibration objective:

```text
NLL
```

Also report:

```text
Brier score
ECE
accuracy
```

For score-type tasks, additionally report:

```text
MAE
QWK if already supported or practical
```

---

# Metrics

Implement reproducible metric computation.

At minimum:

## Classification

```text
accuracy
negative log-likelihood
Brier score
ECE
```

## `noul`

Also report:

```text
mean P(true)
mean semantic margin
margin distribution
false-positive / false-negative counts
```

## `score`

Report:

```text
accuracy
MAE
NLL where applicable
Brier where applicable
QWK if already supported
```

Do not combine calibration quality and classification accuracy into one opaque score.

Report them separately.

---

# ECE implementation

Implement Expected Calibration Error with clearly documented binning.

Use a fixed default such as:

```text
15 equal-width confidence bins
```

unless the project already defines something else.

Record the bin count in output metadata.

Avoid adaptive behavior that makes runs difficult to compare.

---

# Prompt configuration experiments

The calibration pipeline must support comparing multiple prompt configurations.

At minimum make it possible to identify runs such as:

```text
letters + think
letters + no-think
letters + no-think + Answer:
natural Yes/No
raw classification prompt
```

Do not necessarily implement all prompt variants in this task if they do not exist yet.

However, the dataset and runner must record prompt configuration explicitly so results from different prompt forms cannot be mixed accidentally.

A calibration artifact must be tied to the exact prompt configuration that produced its logits.

---

# Important consistency rule

Calibration parameters must be invalidated when any of the following changes:

```text
model
model hash
candidate scheme
prompt layout
assistant suffix
candidate token mapping
primitive formulation
```

Do not apply a calibration artifact to incompatible logits silently.

Add compatibility metadata and validation.

---

# Calibration artifact

Produce a versioned calibration artifact that can be loaded by `pjev`.

A possible format:

```json
{
  "version": 1,
  "model": "...",
  "model_hash": "...",
  "prompt_config": "...",
  "candidate_scheme": "...",
  "parameters": {
    "noul": {
      "temperature": 1.12,
      "prior_alpha": 0.74
    },
    "choice": {
      "temperature": 0.93,
      "prior_alpha": 0.31
    },
    "score": {
      "temperature": 1.28,
      "prior_alpha": 0.0
    }
  }
}
```

This is illustrative.

Use the project's existing configuration conventions where possible.

The runtime should reject incompatible calibration artifacts with a clear error.

---

# Runtime integration

Integrate calibration into the existing decision path without changing its basic architecture.

The intended order is:

```text
raw candidate logits
    Å´
prior correction
    Å´
temperature scaling
    Å´
softmax
    Å´
probabilities
    Å´
argmax / expected score
```

Keep:

```text
raw_probs
calibrated_probs
```

distinguishable where the existing API supports it.

Do not overwrite raw diagnostic data.

---

# Command-line tooling

Add a reproducible CLI workflow.

Names can follow existing project conventions, but conceptually support commands equivalent to:

```text
pjev-calibrate collect ...
pjev-calibrate fit ...
pjev-calibrate evaluate ...
```

or a single tool with subcommands.

Required capabilities:

```text
collect inference results
fit calibration parameters
evaluate a saved calibration
compare calibrated vs uncalibrated metrics
write JSON / JSONL reports
```

Prefer machine-readable output.

A human-readable summary is useful but secondary.

---

# Example workflow

The finished tooling should support a workflow equivalent to:

```text
1. Collect logits

jevbench
    Å´
pjev inference
    Å´
results/raw.jsonl

2. Fit

raw.jsonl
    Å´
tuning split
    Å´
fit prior alpha + temperature
    Å´
calibration.json

3. Validate

raw.jsonl + calibration.json
    Å´
validation metrics

4. Freeze

select configuration

5. Held-out evaluation

raw.jsonl + frozen calibration
    Å´
held-out report
```

---

# Required comparison report

Generate a report showing, per primitive:

```text
uncalibrated
prior-corrected only
temperature-only
prior-corrected + temperature
```

Include:

```text
accuracy
NLL
Brier
ECE
```

For example:

```text
noul

                         Acc     NLL     Brier    ECE
raw                      ...
temperature              ...
prior correction         ...
prior + temperature      ...
```

The goal is to make the contribution of each calibration step visible.

---

# Diagnostic report for candidate bias

Add a report for candidate-token bias.

For each candidate scheme, show something equivalent to:

```text
candidate
mean raw logit
content-free prior logit
selection frequency
ground-truth frequency
```

For binary tasks also include:

```text
mean margin
median margin
margin stddev
```

This is important for identifying cases where the model prefers a token such as `A`, `B`, `Yes`, or `No` independent of semantic content.

---

# Reproducibility

All tuning must be reproducible.

Record:

```text
dataset version / identifier
split seed
model identifier
model hash
prompt configuration
candidate scheme
calibration search range
metric configuration
ECE bin count
tool version
```

Avoid undocumented random search.

If randomness is used, expose and record the seed.

---

# Tests

Add unit tests and integration tests.

At minimum:

## 1. Temperature scaling

Verify that:

```text
T = 1
```

reproduces unscaled softmax.

Verify that lower / higher temperatures behave as expected.

## 2. Prior correction

Given known:

```text
raw logits
prior logits
alpha
```

verify exact corrected logits.

## 3. Binary `noul` semantic mapping

Verify that reversed candidate order does not change which probability is reported as `p_true`.

## 4. Split determinism

The same dataset and seed must always produce identical split assignments.

## 5. No held-out leakage

Add tests or structural safeguards so held-out labels cannot be used during parameter fitting.

## 6. Artifact compatibility

Calibration created for one:

```text
model / prompt configuration / candidate scheme
```

must not silently apply to an incompatible configuration.

## 7. Metrics

Use small hand-computable examples to verify:

```text
accuracy
NLL
Brier
ECE
```

---

# Performance constraints

Do not rerun model inference during pure calibration parameter search.

Once raw candidate logits have been collected, searches over:

```text
temperature
prior alpha
metrics
threshold diagnostics
```

must operate only on stored data.

This is one of the primary reasons for building the derived dataset.

---

# Do not overfit the benchmark

The calibration tool must not become an automatic JevBench prompt optimizer that repeatedly inspects held-out performance.

Do not:

```text
search prompts against held_out
search alpha against held_out
search temperature against held_out
select model against held_out
```

The held-out set is for final evaluation only.

---

# Initial recommended experiment

Once the pipeline works, run an initial controlled experiment for `noul`.

Use the same Bonsai model and compare:

```text
Configuration A:
letters
current chat template with think block

Configuration B:
letters
chat template without think block
```

If already implemented, optionally add:

```text
Configuration C:
letters
chat template without think block
explicit Answer:

Configuration D:
natural Yes/No
```

For each configuration:

```text
collect fresh raw logits
measure content-free priors
fit alpha and T on tuning
select on validation
report held-out metrics
```

Do not reuse logits across prompt configurations.

---

# Questions the final report should answer

The implementation should make it possible to answer:

1. How much does candidate prior correction improve NLL?
2. How much does temperature scaling improve calibration?
3. Does calibration improve accuracy, or only probability quality?
4. Does removing the `<think>...</think>` suffix improve held-out performance?
5. Are `A/B` candidates more stable than `Yes/No` candidates?
6. How strong is candidate-position bias?
7. How strong is candidate-token bias?
8. Does `noul` semantic margin react appropriately to input-state changes?
9. Are improvements stable on validation and held-out data?
10. Is Bonsai 1.7B still viable after formulation and calibration are properly controlled?

---

# Non-goals

Do not in this task:

- replace the Bonsai model;
- redesign the HTTP server;
- optimize KV-cache reuse;
- redesign the CLI architecture;
- implement background workers;
- alter unrelated production behavior;
- introduce a large ML framework unless clearly necessary.

Prefer a small, auditable calibration implementation.

Simple C++ or Python-based offline calibration tooling is acceptable if it integrates cleanly with the native C++ runtime and produces a stable versioned calibration artifact.

The runtime inference path itself should remain native C++.

---

# Deliverables

Provide:

1. JevBench inference collection tooling.
2. Derived JSONL dataset generation.
3. Deterministic tuning / validation / held-out splitting.
4. Candidate-prior measurement.
5. Prior correction with tunable `alpha`.
6. Primitive-specific temperature scaling.
7. NLL / Brier / ECE / accuracy evaluation.
8. Versioned calibration artifact output.
9. Runtime loading and compatibility validation.
10. Tests.
11. A comparison report for calibrated vs uncalibrated results.
12. Documentation describing the complete reproduction procedure.

At completion, report:

```text
files changed
commands used
tests run
dataset split sizes
selected alpha values
selected temperatures
validation metrics
held-out metrics
```

Do not claim improvement based on tuning-set metrics alone.

The main success criterion is not merely a higher JevBench score.

The success criterion is a reproducible system that can distinguish:

```text
model capability
prompt/formulation effects
candidate prior bias
probability calibration
```

without contaminating the held-out evaluation.
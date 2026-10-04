# Phase 3 — Calibration

## Objective

Implement Phase 3 of **pjev**.

The goal of this phase is to improve the meaning of the probabilities returned by pjev.

Example raw output:

```text
0.85
0.12
0.03
```

should become closer to a calibrated probability distribution, where confidence better matches empirical correctness.

This phase should begin only after Phase 2 has established a sufficiently stable decision formulation.

Do not use calibration to hide poor argmax quality.

Calibration and accuracy must be evaluated separately.

---

# 1. Project Context

The command name is:

```bash
pjev
```

Use `pjev` consistently in:

- command examples
- documentation
- benchmark tooling
- calibration tooling
- configuration
- output examples

The native architecture remains:

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

The exact internal placement of calibration may differ, but calibration must remain logically separate from raw inference and prompt construction.

---

# 2. Core Principle

Calibration must not change the underlying semantic decision mechanism unnecessarily.

The base pipeline should remain:

```text
prompt
   ↓
llama.cpp logits
   ↓
candidate logits
   ↓
optional prior correction
   ↓
calibration transform
   ↓
probabilities
   ↓
argmax / expected value
```

Preserve access to the uncalibrated values.

Do not overwrite raw logits or raw probabilities in a way that makes analysis impossible.

---

# 3. Phase 2 Dependency

Before implementing calibration, inspect the Phase 2 results and current default formulation.

Identify:

```text
default prompt layout
default candidate scheme
prior correction behavior
candidate mapping
score formulation
```

Calibration must be fitted against one explicit formulation configuration.

Do not mix calibration data from incompatible formulations unless deliberately testing that behavior.

Store enough metadata to know which formulation a calibration artifact belongs to.

---

# 4. Prior Correction Comes First

If Phase 2 introduced candidate-token prior correction, apply that before temperature scaling.

Conceptually:

```text
raw logits
    ↓
prior correction
    ↓
temperature scaling
    ↓
softmax
```

For example:

```text
corrected_logit_i =
    raw_logit_i - prior_logit_i
```

followed by:

```text
scaled_logit_i =
    corrected_logit_i / T
```

Do not calibrate raw token bias and semantic evidence together if Phase 2 already established a prior correction mechanism.

---

# 5. Start With Temperature Scaling

Implement temperature scaling as the first calibration method.

Do not begin with more complex calibration methods.

Use:

```text
scaled_logit_i = logit_i / T
```

where:

```text
T > 0
```

and probabilities are computed from the scaled candidate logits.

The fitting objective should minimize validation-set negative log-likelihood unless the existing benchmark infrastructure strongly suggests another standard objective.

Keep the implementation small and explicit.

---

# 6. Primitive-Specific Temperatures

Support independent calibration parameters for:

```text
noul
choice
score
```

Conceptually:

```text
T_noul
T_choice
T_score
```

Do not force them to share a single temperature.

Different primitives may have different confidence behavior.

However, also support a global/default mode if useful for comparison.

---

# 7. Calibration Configuration

Introduce an explicit calibration configuration.

Conceptually:

```yaml
calibration:
  enabled: true
  method: temperature
  noul_temperature: 1.0
  choice_temperature: 1.0
  score_temperature: 1.0
```

The exact format may differ.

The important requirements are:

- calibration can be enabled or disabled;
- parameters are explicit;
- raw behavior remains reproducible;
- calibration settings are visible in experiment output;
- defaults do not silently change during fitting.

---

# 8. Calibration Artifact

Create a machine-readable calibration artifact.

It should contain enough information to reproduce the transform.

For example:

```json
{
  "version": 1,
  "method": "temperature",
  "model": "bonsai-1.7b",
  "formulation": {
    "layout": "state-last",
    "candidate_scheme": "letters",
    "prior_correction": true
  },
  "parameters": {
    "noul_temperature": 1.08,
    "choice_temperature": 1.31,
    "score_temperature": 0.92
  }
}
```

The exact schema may differ.

Include metadata that prevents accidental use with the wrong experimental setup.

---

# 9. Fit / Evaluate Separation

Separate calibration fitting from evaluation.

Do not fit and report metrics on the same examples as though they were unbiased evaluation.

Use explicit dataset roles such as:

```text
train / calibration split
validation
test
```

or the closest structure supported by the current JevBench data.

If only a limited split structure exists, document exactly what is being used.

Do not silently tune on the final evaluation set.

---

# 10. Reproducibility

Calibration fitting must be deterministic where practical.

Record:

```text
dataset/split
sample count
model identifier
prompt formulation
candidate scheme
prior correction state
fitted temperature
objective value
```

If optimization uses an iterative method, make it deterministic unless randomness is necessary.

---

# 11. Temperature Fitting

Implement a numerically stable fitting procedure.

Possible approaches include:

```text
1D optimization over log(T)
bounded scalar optimization
simple Newton/gradient method
```

Avoid heavyweight ML dependencies just to fit one scalar.

Prefer a small native C++ implementation or existing lightweight project dependency.

Parameterize using:

```text
log_T
```

if useful to guarantee:

```text
T > 0
```

Do not allow zero or negative temperatures.

---

# 12. Numerical Stability

Use numerically stable softmax / log-softmax operations.

Do not compute:

```text
exp(logit)
```

naively when logits may be large.

Use standard max-subtraction or log-sum-exp techniques.

Calibration code must not introduce NaNs or infinities for normal candidate logits.

Add tests for:

```text
large positive logits
large negative logits
nearly equal logits
extreme temperature values
```

---

# 13. Required Metrics

Evaluate at least:

```text
NLL
Brier score
ECE
accuracy
```

Report accuracy separately from calibration metrics.

Do not present calibration as successful merely because accuracy changes.

The main interpretation should be:

```text
argmax quality
vs
probability quality
```

as two distinct dimensions.

---

# 14. Negative Log-Likelihood

For classification-like primitives, compute:

```text
NLL = -mean(log(p_true))
```

Use epsilon-safe or numerically stable log probabilities.

Document whether NLL is averaged:

```text
per sample
per primitive
overall
```

Prefer per-primitive reporting.

---

# 15. Brier Score

Implement Brier score over the candidate probability distribution.

For a one-hot target:

```text
Brier =
    mean(sum_i (p_i - y_i)^2)
```

Report per primitive where possible.

Do not reduce everything to top-1 probability only.

---

# 16. Expected Calibration Error

Implement ECE with an explicit binning policy.

For example:

```text
10 or 15 fixed-width confidence bins
```

Record the bin count in result metadata.

Do not hide the ECE definition.

Compute confidence using the predicted class probability for classification-style primitives.

If score requires different handling, document it explicitly rather than forcing an inappropriate definition.

---

# 17. Score Calibration

Treat score carefully.

The existing score mechanism uses a candidate distribution and expected value.

Preserve that behavior unless Phase 2 established a different default.

Calibration should operate on the score candidate distribution before expected value computation.

Conceptually:

```text
score candidate logits
        ↓
prior correction
        ↓
temperature scaling
        ↓
probabilities
        ↓
expected value
```

Measure both:

```text
distribution calibration
score MAE
```

where applicable.

Do not assume that improving NLL automatically improves MAE.

---

# 18. Raw vs Calibrated Output

The experiment and diagnostics paths should be able to expose both:

```text
raw probabilities
calibrated probabilities
```

Do not remove raw values.

This is important for later analysis.

For example, result output may include:

```json
{
  "raw": {
    "probabilities": [...]
  },
  "calibrated": {
    "probabilities": [...]
  }
}
```

Exact schema is flexible.

---

# 19. Runtime Behavior

At runtime, pjev should be able to:

```text
run without calibration
run with a calibration artifact
```

Calibration should be optional.

Do not hard-code fitted temperatures into source code.

Load them from explicit configuration or artifact.

If the artifact is invalid or incompatible, fail clearly.

Do not silently fall back to another calibration.

---

# 20. Calibration Compatibility Checks

When loading a calibration artifact, verify relevant metadata where possible.

At minimum consider:

```text
primitive support
method
artifact version
candidate scheme
prompt layout
prior correction mode
model identifier or model hash
```

The strictness level may depend on what metadata exists in the current project.

Prefer explicit warnings/errors over silently applying a calibration fitted for another formulation.

---

# 21. pjev Commands

Add calibration operations under the `pjev` command namespace.

Possible structure:

```bash
pjev calibration fit ...
pjev calibration evaluate ...
```

or:

```bash
pjev calibrate ...
```

Use the command structure that best fits the existing CLI conventions.

Do not create unrelated standalone executables.

Examples should use the actual command implemented.

---

# 22. Example Workflow

The intended user/developer workflow should resemble:

```bash
# collect/fetch calibration data using the current default formulation
pjev calibration fit \
  --input ... \
  --output calibration.json

# evaluate raw vs calibrated metrics
pjev calibration evaluate \
  --calibration calibration.json \
  --input ...

# run server using fitted calibration
pjev serve \
  --model ... \
  --calibration calibration.json
```

Adapt to the repository's actual CLI design.

---

# 23. Experiment Result Format

Extend the Phase 2 experiment result schema.

Include calibration metadata.

Conceptually:

```json
{
  "primitive": "choice",
  "calibration": {
    "enabled": true,
    "method": "temperature",
    "temperature": 1.27
  },
  "metrics": {
    "accuracy": 0.61,
    "nll": 0.92,
    "brier": 0.31,
    "ece": 0.08
  }
}
```

Also record the corresponding uncalibrated metrics.

---

# 24. Comparative Reporting

Every calibration evaluation should show:

```text
before calibration
after calibration
delta
```

for:

```text
NLL
Brier
ECE
accuracy
```

For score also include:

```text
MAE
```

Do not report only the calibrated result.

---

# 25. Reliability Diagram Data

Generate machine-readable reliability-bin data.

For example:

```json
[
  {
    "confidence_min": 0.0,
    "confidence_max": 0.1,
    "count": 20,
    "mean_confidence": 0.07,
    "accuracy": 0.10
  }
]
```

Producing graphical plots is optional.

The important requirement is that the data needed to draw reliability diagrams is available.

Do not introduce a plotting dependency into the C++ runtime unless already justified.

---

# 26. Calibration Dataset Leakage

Be careful about leakage.

Do not:

```text
fit T on test
then report test calibration as independent evaluation
```

If JevBench does not provide a dedicated calibration split, create or document a deterministic split from an allowed development/public split.

Record the split procedure.

Use a fixed seed if random partitioning is required.

---

# 27. Keep Calibration Separate From Phase 2 Search

Do not retune:

```text
layout
candidate scheme
candidate binding
prompt wording
```

while fitting calibration.

Freeze the formulation first.

Otherwise the source of improvement becomes ambiguous.

If formulation must change because of a correctness issue, refit calibration afterward.

---

# 28. Accuracy Must Remain Visible

Temperature scaling should generally preserve candidate ordering when applied uniformly within a primitive.

Therefore, unexpected top-1 accuracy changes may indicate:

```text
implementation bug
different prior correction
different candidate set
score-specific expected-value behavior
```

Investigate such changes.

Do not assume they are expected.

For `noul` and `choice`, verify explicitly that temperature scaling alone does not change argmax.

---

# 29. Testing Requirements

Add unit tests for:

```text
temperature transform
T = 1 identity behavior
positive temperature validation
stable softmax
stable log-softmax
NLL
Brier score
ECE binning
artifact serialization
artifact deserialization
artifact compatibility checks
```

Add primitive-specific tests for:

```text
noul
choice
score
```

---

# 30. Invariants

Test these invariants.

## Identity

```text
T = 1
```

must reproduce uncalibrated probabilities within floating-point tolerance.

## Argmax Preservation

For noul/choice:

```text
temperature scaling
```

must preserve candidate ordering.

## Probability Normalization

Calibrated probabilities must sum to approximately:

```text
1.0
```

## Monotonic Confidence Effect

For:

```text
T > 1
```

distributions should become softer.

For:

```text
0 < T < 1
```

distributions should become sharper.

---

# 31. Real-Model Integration Test

Include at least one integration path using the real Bonsai/llama.cpp backend.

Verify:

```text
model logits
    ↓
prior correction
    ↓
temperature scaling
    ↓
probabilities
```

and compare with calibration disabled.

Do not rely only on synthetic logits.

---

# 32. JevBench Integration

Use the existing JevBench-based evaluation workflow.

The intended cycle is:

```text
fixed Phase 2 formulation
        ↓
collect calibration data
        ↓
fit temperature
        ↓
run JevBench evaluation
        ↓
compare calibration metrics
```

Do not replace JevBench with a new private evaluation suite.

---

# 33. Performance

Calibration math should be negligible compared with prompt evaluation.

Do not spend significant effort optimizing temperature scaling.

Do not introduce SIMD or special acceleration for calibration.

Keep the implementation obvious and correct.

---

# 34. No New Model Work

Do not evaluate alternative models in Phase 3.

Do not introduce model-specific calibration hacks.

The model value decision belongs to Phase 5.

Phase 3 should calibrate the currently selected Bonsai-based pjev formulation.

---

# 35. No Complex Calibration Methods Yet

Do not implement unless evidence clearly requires them:

```text
isotonic regression
Platt scaling variants
Dirichlet calibration
vector scaling
matrix scaling
neural calibration
per-class temperature scaling
```

Temperature scaling is the intended first method.

If it fails to improve calibration metrics, document that result rather than immediately escalating complexity.

---

# 36. Definition of Done

Phase 3 is complete when:

```text
[ ] Calibration is implemented as a separate reusable component.

[ ] Temperature scaling is supported.

[ ] Calibration can be disabled.

[ ] Separate temperatures can be fitted for noul, choice, and score.

[ ] Phase 2 prior correction is applied before calibration where enabled.

[ ] A machine-readable calibration artifact can be written and loaded.

[ ] Calibration artifacts contain formulation metadata.

[ ] NLL is implemented and reported.

[ ] Brier score is implemented and reported.

[ ] ECE is implemented and reported.

[ ] Accuracy remains separately reported.

[ ] Score MAE remains separately reported.

[ ] Raw and calibrated metrics are both available.

[ ] Reliability-bin data is available.

[ ] Calibration fitting and evaluation use separate data.

[ ] T=1 identity tests pass.

[ ] Argmax-preservation tests for noul/choice pass.

[ ] Numerical stability tests pass.

[ ] Real-model integration tests pass.

[ ] JevBench can evaluate calibrated pjev behavior.

[ ] Documentation describes fitting, evaluation, and runtime use.
```

---

# 37. Phase 3 Exit Gate

The goal is not a predetermined temperature value.

The desired result is evidence that:

```text
calibrated probabilities
```

better reflect observed correctness than:

```text
raw probabilities
```

without confusing calibration improvement with accuracy improvement.

At minimum, expect to see meaningful improvement in one or more of:

```text
NLL
Brier score
ECE
```

while keeping argmax accuracy separately visible.

If calibration does not improve these metrics, report that clearly.

Do not manipulate the evaluation procedure to force a positive result.

---

# 38. Final Implementation Report

When implementation is complete, provide:

## Architecture

Explain where calibration sits relative to:

```text
DecisionEngine
prior correction
softmax
score expected value
```

## Files Changed

List added and significantly modified files.

## Calibration Method

Document:

```text
temperature scaling implementation
fitting objective
optimizer
parameter constraints
```

## Calibration Artifacts

Show the actual artifact schema.

## Commands

Provide exact working commands using:

```bash
pjev ...
```

for:

```text
fit
evaluate
serve with calibration
```

## Dataset / Split

Document:

```text
which data was used for fitting
which data was used for evaluation
sample counts
split method
```

## Metrics

Report before/after values for:

```text
NLL
Brier
ECE
accuracy
```

and for score:

```text
MAE
```

## Per-Primitive Results

Report separately for:

```text
noul
choice
score
```

## Temperatures

Report fitted:

```text
T_noul
T_choice
T_score
```

## Tests

Provide exact test commands and results.

## Remaining Issues

Document:

```text
poorly calibrated primitives
dataset limitations
artifact compatibility limitations
score-specific problems
```

Do not omit negative findings.

---

# Final Instruction

Phase 3 is about **probability quality**, not benchmark-score chasing.

Preserve the distinction:

```text
Can pjev choose the right answer?
```

versus:

```text
When pjev says 80%, is it correct roughly 80% of the time?
```

Phase 2 addresses the first question.

Phase 3 addresses the second.

Prioritize:

```text
correct calibration methodology
reproducibility
data separation
numerical stability
transparent reporting
```

over:

```text
complex calibration methods
feature expansion
performance optimization
new model experiments
```
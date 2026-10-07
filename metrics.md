# Task: Add Confidence, Calibration, and Margin Metrics to JevBench Evaluation

## Objective

Extend the current JevBench evaluation output so that model and formulation improvements can still be measured when classification accuracy is already saturated at `1.0`.

The current evaluation primarily reports:

```text
accuracy
eval_ms
labeled
n
```

This is insufficient for cases such as the `easy` split, where both baseline and E1 can achieve 100% accuracy.

We need metrics that distinguish:

```text
barely correct predictions
from
strong, well-separated, well-calibrated predictions
```

The implementation should focus first on `noul`, but shared metric code should be reusable for other classification primitives where practical.

---

# Required new metrics

Add at least the following metrics for `noul`:

```text
NLL
Brier score
ECE
mean signed semantic margin
median signed semantic margin
p10 signed semantic margin
minimum signed semantic margin
mean confidence on correct predictions
```

For E1 / order-ensemble experiments, additionally support:

```text
order disagreement rate
mean ensemble margin gain
median ensemble margin gain
improved margin count
degraded margin count
```

Do not remove existing metrics.

---

# 1. Negative Log-Likelihood

For each labeled `noul` sample, use the probability assigned to the ground-truth semantic answer.

For binary classification:

```text
NLL_i = -log(P(correct_i))
```

Dataset-level:

```text
NLL = mean(NLL_i)
```

Lower is better.

Clamp probabilities numerically before taking `log`, for example using a small epsilon.

Use a clearly defined constant such as:

```text
epsilon = 1e-12
```

or the project's existing numeric convention.

Do not silently discard extreme probabilities.

---

# 2. Brier Score

For `noul`, use binary semantic probability.

Let:

```text
y = 1 if ground truth is True
y = 0 if ground truth is False
```

and:

```text
p = P(True)
```

Then:

```text
Brier_i = (p - y)^2
```

Dataset-level:

```text
Brier = mean(Brier_i)
```

Lower is better.

Do not compute Brier using candidate index directly.

Use semantic `P(True)`.

---

# 3. Expected Calibration Error

Implement ECE using fixed equal-width confidence bins.

Default:

```text
15 bins
```

For each prediction define:

```text
confidence = probability of predicted semantic class
correct = 1 or 0
```

For each bin:

```text
ECE contribution =
    bin_fraction
    * abs(mean_confidence - mean_accuracy)
```

Then sum over bins.

Use deterministic fixed bin boundaries.

Record the bin count in evaluation output or metadata.

Do not use adaptive bins in the initial implementation.

---

# 4. Semantic Logit Margin

For every `noul` prediction, derive the semantic logit margin:

```text
semantic_margin =
    logit_true - logit_false
```

This must be computed after mapping candidate labels back to semantic True / False.

Do not assume candidate position.

For example:

```text
A = False
B = True
```

means:

```text
semantic_margin = logit_B - logit_A
```

while:

```text
A = True
B = False
```

means:

```text
semantic_margin = logit_A - logit_B
```

---

# 5. Signed Semantic Margin

Convert semantic margin into a ground-truth-oriented score.

Define:

```text
signed_margin =
    semantic_margin
    if ground_truth == True

signed_margin =
    -semantic_margin
    if ground_truth == False
```

Interpretation:

```text
signed_margin > 0
    correct side of the decision boundary

signed_margin < 0
    incorrect side

larger positive value
    stronger correct prediction

value near zero
    fragile / uncertain prediction
```

This metric is especially important when accuracy is already 100%.

---

# 6. Margin Summary Statistics

Report:

```text
mean_signed_margin
median_signed_margin
p10_signed_margin
min_signed_margin
```

Use the 10th percentile as a lower-tail robustness metric.

The percentile implementation must be deterministic and documented.

If the project already has a percentile convention, reuse it.

Otherwise choose a standard interpolation method and state it in comments or documentation.

---

# Why lower-tail margin matters

If two configurations both achieve:

```text
accuracy = 1.0
```

but produce:

```text
Config A:
mean margin = 2.1
min margin = 0.08

Config B:
mean margin = 2.0
min margin = 0.91
```

Config B may be more robust even though its mean margin is slightly lower.

Do not evaluate margin quality from the mean alone.

---

# 7. Mean Confidence on Correct Predictions

For each correctly classified sample:

```text
confidence =
    max(P(True), P(False))
```

Report:

```text
mean_confidence_correct
```

If there are no correct predictions, return a documented null / NaN representation consistent with the existing output format.

Do not divide by zero.

---

# Raw vs calibrated metrics

If both raw and calibrated probabilities are available, keep them distinguishable.

Preferred structure:

```text
raw:
  nll
  brier
  ece

calibrated:
  nll
  brier
  ece
```

If the current evaluator only has one probability set available, implement metrics for that set first, but structure the code so raw and calibrated values can be added without duplication.

Do not overwrite raw metrics with calibrated metrics.

---

# Margin source

Margins should preferably be computed from logits before temperature scaling.

For example:

```text
raw_semantic_margin
corrected_semantic_margin
ensemble_semantic_margin
```

where available.

Temperature changes probability sharpness but should not erase the underlying model-margin diagnostics.

If prior correction is enabled, preserve enough information to report both:

```text
raw margin
corrected margin
```

where practical.

---

# E1-specific metrics

When binary option-order ensemble E1 is enabled, add the following.

## Order disagreement rate

For each sample, compute the semantic prediction of both ensemble members independently.

Define:

```text
disagreement = 1
if order1_prediction != order2_prediction
```

Report:

```text
order_disagreement_count
order_disagreement_rate
```

This reveals option-order sensitivity directly.

---

# E1 margin gain

When baseline and E1 results can be matched by source/sample ID, compute:

```text
margin_gain =
    signed_margin_E1
    - signed_margin_baseline
```

Report:

```text
mean_margin_gain
median_margin_gain
improved_margin_count
degraded_margin_count
unchanged_margin_count
```

Use a small numeric tolerance when classifying unchanged floating-point margins.

For example:

```text
abs(gain) < 1e-9
```

or the project's existing tolerance.

---

# Important interpretation

A configuration may have identical accuracy but still be meaningfully different.

Example:

```text
baseline:
accuracy = 1.000
NLL = 0.31
Brier = 0.09
mean_signed_margin = 1.42
min_signed_margin = 0.18

E1:
accuracy = 1.000
NLL = 0.12
Brier = 0.03
mean_signed_margin = 2.37
min_signed_margin = 0.91
```

This should be reported as a meaningful improvement even though accuracy did not change.

Conversely:

```text
baseline:
accuracy = 1.000
mean_signed_margin = 2.1

E1:
accuracy = 1.000
mean_signed_margin = 0.8
```

should reveal that E1 made the predictions less robust.

---

# Output format

Extend the existing result JSON.

For `noul`, a target shape could be:

```json
{
  "noul": {
    "accuracy": 1.0,
    "nll": 0.12,
    "brier": 0.031,
    "ece": 0.08,

    "mean_signed_margin": 2.37,
    "median_signed_margin": 2.10,
    "p10_signed_margin": 1.02,
    "min_signed_margin": 0.91,

    "mean_confidence_correct": 0.89,

    "eval_ms": 12392,
    "labeled": 12,
    "n": 12
  }
}
```

For E1, optionally extend with:

```json
{
  "order_disagreement_count": 2,
  "order_disagreement_rate": 0.0833,
  "mean_margin_gain": 0.27,
  "median_margin_gain": 0.19,
  "improved_margin_count": 8,
  "degraded_margin_count": 4
}
```

Use the project's existing JSON style.

---

# Comparison report

Add or extend the human-readable benchmark summary to show baseline and E1 side by side.

Example:

```text
noul / easy

metric                  baseline        E1
------------------------------------------------
accuracy                1.0000          1.0000
NLL                     0.3100          0.1200
Brier                   0.0900          0.0310
ECE                     0.1100          0.0800
mean signed margin      1.4200          2.3700
median signed margin    1.3100          2.1000
p10 signed margin       0.3200          1.0200
min signed margin       0.1800          0.9100
eval_ms                 6146            12392
```

Do not hide latency when showing quality improvements.

---

# Per-example diagnostics

Where the evaluator already supports detailed output, include per-example fields such as:

```text
source_id
ground_truth
prediction
p_true
confidence
semantic_margin
signed_margin
correct
```

For E1:

```text
order1_semantic_margin
order2_semantic_margin
ensemble_semantic_margin
order1_prediction
order2_prediction
order_disagreement
```

This is useful for identifying examples where the aggregate accuracy is unchanged but confidence deteriorates.

---

# Metric behavior with unlabeled samples

Metrics requiring ground truth must use only labeled samples.

This includes:

```text
accuracy
NLL
Brier
ECE
signed margin
mean confidence correct
margin gain against ground truth
```

Do not mix unlabeled samples into denominators.

Continue to report:

```text
labeled
n
```

separately.

---

# Tests

Add unit tests with small hand-computable examples.

## NLL

Example:

```text
ground truth = True
P(True) = 0.8
```

Expected:

```text
NLL = -log(0.8)
```

---

## Brier

Example:

```text
ground truth = True
P(True) = 0.8
```

Expected:

```text
Brier = (0.8 - 1)^2 = 0.04
```

---

## Semantic margin

Example:

```text
A=False
B=True

logit(A)=1
logit(B)=3
```

Expected:

```text
semantic_margin = +2
```

---

## Reversed candidate mapping

Example:

```text
A=True
B=False

logit(A)=3
logit(B)=1
```

Expected:

```text
semantic_margin = +2
```

This test is mandatory.

---

## Signed margin

For:

```text
ground truth = False
semantic_margin = -2
```

Expected:

```text
signed_margin = +2
```

For:

```text
ground truth = False
semantic_margin = +2
```

Expected:

```text
signed_margin = -2
```

---

## Saturated accuracy comparison

Create two synthetic sets where both achieve:

```text
accuracy = 1.0
```

but one set has consistently larger correct-class margins.

Verify that:

```text
accuracy is identical
NLL differs
Brier differs
mean signed margin differs
minimum margin differs
```

This test demonstrates why the new metrics are needed.

---

## ECE

Use a small fixed set with known confidence and correctness values.

Verify exact bin assignment and expected ECE.

---

## E1 disagreement

Create:

```text
order1 prediction = True
order2 prediction = False
```

and verify:

```text
disagreement = true
```

---

# Numerical stability

Use stable floating-point implementations.

For NLL:

```text
clamp probabilities away from 0 and 1
```

For sigmoid / softmax:

reuse existing stable implementations where possible.

Do not create a separate numerically weaker softmax implementation inside the metrics code.

---

# Architecture

Keep metric calculation independent from model inference where possible.

Preferred flow:

```text
inference result
    «
semantic normalization
    «
metric accumulator
    «
aggregate result
```

This allows metrics to be recomputed later from stored JevBench results without rerunning llama.cpp.

Do not couple NLL / Brier / percentile calculation directly to `LlamaBackend`.

---

# Acceptance criteria

The task is complete when:

1. Existing accuracy output remains unchanged.
2. NLL is reported for labeled `noul`.
3. Brier score is reported.
4. ECE is reported.
5. Semantic margins are position-independent.
6. Signed margins are ground-truth aligned.
7. Mean / median / p10 / minimum margin are reported.
8. 100%-accuracy configurations can be differentiated by the new metrics.
9. E1 disagreement can be measured.
10. Baseline-vs-E1 margin gain can be measured where matching sample IDs are available.
11. Unit tests pass.
12. Existing benchmark behavior remains backward-compatible apart from additional output fields.

---

# Non-goals

Do not in this task:

- modify model prompts;
- change E1 aggregation;
- change calibration parameters;
- change the model;
- add few-shot examples;
- optimize inference performance;
- alter candidate mapping behavior.

This task is measurement only.

Do not improve benchmark results by changing inference behavior.

---

# Deliverables

Provide:

1. metric implementation;
2. JSON output changes;
3. human-readable comparison output where applicable;
4. unit tests;
5. per-example diagnostic fields if supported;
6. documentation defining each metric.

At completion, report:

```text
files changed
tests run
metric definitions
example benchmark output
```

Also run the existing baseline and E1 JevBench evaluations and show at least:

```text
accuracy
NLL
Brier
ECE
mean signed margin
median signed margin
p10 signed margin
min signed margin
eval_ms
```

for:

```text
easy
original
hard
```

The key success criterion is:

> Two configurations with identical 100% accuracy must still be meaningfully distinguishable by confidence, calibration, and decision-margin quality.
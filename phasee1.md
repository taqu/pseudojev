# Task: Implement E1 — Binary Option-Order Ensemble for `noul`

## Objective

Implement a binary option-order ensemble for the `noul` primitive.

The purpose is to reduce candidate-position and candidate-token bias by evaluating the same semantic decision under both possible candidate orderings:

```text id="36u9u2"
Run 1:
A: No
B: Yes

Run 2:
A: Yes
B: No
```

The two runs must represent the same semantic question.

After inference, convert each run back into semantic `True` / `False` space and combine the evidence there.

The ensemble must not average candidate positions directly.

---

# Motivation

Small language models may have systematic preferences for:

```text id="i1c8at"
A over B
first option over second option
particular candidate tokens
```

These biases can affect constrained-logit decisions even when the semantic content is unchanged.

For binary `noul`, there are only two possible candidate permutations, so we can evaluate both exactly.

The goal is to estimate the semantic decision while reducing sensitivity to option ordering.

---

# Scope

Implement this experiment only for:

```text id="5n5vc7"
primitive = noul
```

Do not generalize to arbitrary `choice` permutations in this task.

Do not redesign unrelated prompt or calibration code.

---

# Required semantic invariant

Candidate labels and semantic meaning must remain separate.

For example:

```text id="h8qgx5"
Run 1:
A = semantic False
B = semantic True

Run 2:
A = semantic True
B = semantic False
```

The model still emits candidate-token logits for:

```text id="6mdng1"
A
B
```

but all ensemble calculations must first remap those logits to:

```text id="z3k53c"
semantic True
semantic False
```

Never assume:

```text id="z5iw3p"
A = False
B = True
```

globally.

---

# Reference algorithm

For a single `noul` request, construct two semantically equivalent prompts.

## Ordering 1

```text id="p6dfza"
A: No
B: Yes
```

Let the raw candidate logits be:

```text id="a9srp4"
L1_A
L1_B
```

Semantic logits are therefore:

```text id="z39wpi"
L1_false = L1_A
L1_true  = L1_B
```

## Ordering 2

```text id="2tq5pe"
A: Yes
B: No
```

Let the raw candidate logits be:

```text id="hjmpww"
L2_A
L2_B
```

Semantic logits are:

```text id="mnjhs0"
L2_true  = L2_A
L2_false = L2_B
```

---

# Ensemble in semantic logit-margin space

For each ordering, compute the semantic margin:

```text id="pb2bwm"
m1 = L1_true - L1_false
m2 = L2_true - L2_false
```

Then combine:

```text id="a54gwb"
m_ensemble = (m1 + m2) / 2
```

For the uncalibrated binary probability:

```text id="ij7647"
P(true) = sigmoid(m_ensemble)
P(false) = 1 - P(true)
```

If temperature calibration is enabled:

```text id="76ferh"
P(true) = sigmoid(m_ensemble / T_noul)
```

Use the existing calibration conventions rather than creating a second incompatible implementation.

---

# Prior correction

If candidate prior correction is enabled, apply it independently to each ordering before computing the semantic margin.

For ordering `r`:

```text id="8nue78"
corrected_candidate_logit =
    raw_candidate_logit
    - alpha * prior_candidate_logit
```

Then remap corrected candidate logits into semantic True / False space.

For example:

```text id="gu31zo"
m1 =
    corrected_L1_true
    - corrected_L1_false

m2 =
    corrected_L2_true
    - corrected_L2_false
```

Then:

```text id="n37m41"
m_ensemble = (m1 + m2) / 2
```

Do not average raw `A` logits across runs.

Do not average raw `B` logits across runs.

The candidate identity changes semantic meaning between the two permutations.

---

# Why use logit-margin averaging

Do not make probability averaging the primary implementation.

The preferred implementation is:

```text id="u6ns5j"
candidate logits
    ↓
semantic remapping
    ↓
semantic margin per ordering
    ↓
average margins
    ↓
temperature
    ↓
sigmoid
```

This keeps the ensemble in model-score space and allows calibration to be applied cleanly after aggregation.

For diagnostics only, it is acceptable to also report:

```text id="wfz4cb"
mean of per-run P(true)
```

but the production E1 result should use the semantic logit-margin ensemble unless benchmark evidence clearly shows otherwise.

---

# API / configuration

Add an explicit configuration switch.

Use the project's existing configuration style.

Conceptually:

```text id="v5gcfi"
noul_order_ensemble = false | true
```

or an enum such as:

```text id="n5ktmr"
NoulEnsembleMode::NONE
NoulEnsembleMode::BINARY_ORDER
```

Default behavior should remain backward-compatible unless the existing project configuration already treats this branch as experimental.

Do not silently enable the ensemble everywhere before JevBench evaluation confirms its value.

---

# Decision output diagnostics

When the ensemble is enabled, expose enough diagnostic information for research.

Where practical, record:

```text id="sf1hd5"
ordering_1:
  candidate mapping
  raw candidate logits
  corrected candidate logits
  semantic margin

ordering_2:
  candidate mapping
  raw candidate logits
  corrected candidate logits
  semantic margin

ensemble:
  mean semantic margin
  calibrated P(true)
  final semantic prediction
```

This can be debug/test metadata rather than part of the stable public HTTP contract.

Avoid noisy production logging.

---

# Reference execution path

First implement a simple correctness-first reference path:

```text id="586xd8"
build ordering 1 prompt
evaluate ordering 1

build ordering 2 prompt
evaluate ordering 2

combine semantic margins
```

This sequential implementation is the reference behavior.

Keep it easy to audit.

---

# Parallel evaluation

Yes: add support for evaluating the two ensemble members in parallel where the backend architecture safely permits it.

However, correctness is more important than concurrency.

The implementation should distinguish:

```text id="90lxb4"
reference path:
    sequential, deterministic

optimized path:
    parallel and/or shared-prefix optimized
```

The optimized result must be numerically equivalent to the reference result within a documented floating-point tolerance.

Do not remove the sequential reference implementation.

---

# Important llama.cpp concurrency constraint

Before implementing true parallel calls, inspect the existing `LlamaBackend`.

Do not assume that two concurrent calls to:

```text id="rl4mnx"
eval_tokens(...)
```

on the same llama.cpp context are thread-safe.

If the backend owns one mutable llama context, do not invoke it concurrently from two threads unless the backend explicitly supports this.

Possible safe implementations include:

```text id="c50dlg"
1. sequential evaluation on one context

2. llama.cpp sequence/batch support within one decode call

3. separate inference contexts for each ensemble member

4. backend-supported parallel sequence evaluation
```

Choose the smallest correct implementation supported by the current architecture.

Do not introduce unsafe thread-level concurrency merely to satisfy the word "parallel".

---

# Preferred optimization direction

The two prompts share almost all semantic input.

Conceptually:

```text id="4gqvpb"
shared state
shared question
shared instruction
       ↓
candidate ordering differs near the end
```

Where possible, structure the implementation so future optimization can reuse the common prefix.

Preferred long-term shape:

```text id="kc57tp"
shared prompt prefix
        ↓
evaluate once
        ↓
fork / reuse KV state
      ↙       ↘
order 1      order 2
      ↓       ↓
logits       logits
       ↘     ↙
semantic ensemble
```

Do not make KV reuse mandatory for E1 correctness.

The first implementation may evaluate both full prompts separately.

But design the API so E1 does not prevent later prefix/KV reuse.

---

# Batch implementation option

If the existing backend supports evaluating multiple sequences in a single batch safely, this is preferable to naïve multithreading.

Conceptually:

```text id="x34p9i"
request
  ↓
construct prompt AB
construct prompt BA
  ↓
backend batch evaluation
  ↓
two logits vectors
  ↓
semantic ensemble
```

Do not duplicate DecisionEngine logic unnecessarily.

If batch support already exists, reuse it where correct.

---

# Tests

Add focused tests.

## 1. Semantic remapping

Synthetic logits:

```text id="eq2mmt"
Run 1:
A = 1.0
B = 3.0

mapping:
A=False
B=True
```

must produce:

```text id="ze1hmc"
m1 = +2.0
```

Run 2:

```text id="6ztz5x"
A = 4.0
B = 1.0

mapping:
A=True
B=False
```

must produce:

```text id="nvi92f"
m2 = +3.0
```

Ensemble:

```text id="l30onf"
m = 2.5
```

Verify the final `P(true)` against the expected sigmoid result.

---

## 2. Position-bias cancellation test

Use synthetic logits representing an `A` preference.

For example:

```text id="zcmynk"
Run 1:
A = 3.0
B = 2.0

Run 2:
A = 3.0
B = 2.0
```

The model prefers `A` equally in both prompts.

Because semantic bindings are reversed:

```text id="t1f7k3"
Run 1:
A=False
B=True
=> negative semantic margin

Run 2:
A=True
B=False
=> positive semantic margin
```

The ensemble semantic margin should approach zero.

This is a key E1 test.

---

## 3. Semantic evidence survives permutation

Construct synthetic logits where the model consistently prefers semantic True despite candidate reversal.

For example:

```text id="5kxnid"
Run 1:
A=False = 1.0
B=True  = 3.0

Run 2:
A=True  = 3.5
B=False = 1.5
```

The ensemble must strongly prefer semantic True.

---

## 4. Prior correction

Verify candidate priors are applied in candidate-token space before semantic remapping.

---

## 5. Temperature

Verify `T_noul = 1` matches the unscaled ensemble probability.

---

## 6. Sequential vs optimized equivalence

If a parallel/batched implementation is added:

```text id="fweoaz"
reference sequential result
optimized result
```

must match within tolerance for:

```text id="nb63e9"
raw logits
semantic margins
ensemble margin
P(true)
selected semantic result
```

---

# JevBench experiment

After implementation, run E1 as a controlled JevBench experiment.

Use the same model and prompt configuration for both conditions.

Compare:

```text id="i3brqh"
Baseline:
single ordering

E1:
binary order ensemble
```

Do not change few-shot examples, temperature, prompt wording, model, or other parameters simultaneously.

If prior correction and calibration are already available, report both:

```text id="7pv0yb"
raw baseline vs raw E1

calibrated baseline vs calibrated E1
```

---

# Metrics

Report at minimum for `noul`:

```text id="a0evqf"
accuracy
NLL
Brier
ECE
```

Also report:

```text id="f6cs7t"
mean P(true)
semantic margin mean
semantic margin stddev
option-order disagreement rate
```

Define disagreement rate as cases where:

```text id="u1s36h"
ordering 1 semantic prediction
!=
ordering 2 semantic prediction
```

This is an important diagnostic for option-order sensitivity.

---

# Latency measurement

E1 approximately doubles inference work in the naïve reference implementation.

Measure:

```text id="2xk0yf"
baseline latency
sequential E1 latency
optimized E1 latency, if implemented
```

Also record:

```text id="7g3q4n"
prompt tokens evaluated
RSS if practical
```

Do not claim the ensemble is free.

The quality gain must be evaluated against its latency cost.

---

# Acceptance criteria

E1 is correctly implemented when:

1. Both binary option orderings are evaluated.
2. Candidate logits are remapped to semantic True / False correctly.
3. Ensemble combination occurs in semantic logit-margin space.
4. `p_true` is independent of candidate position.
5. Synthetic position bias cancels as expected.
6. Semantic evidence survives candidate reversal.
7. Sequential reference tests pass.
8. Any optimized/parallel path matches the reference path numerically.
9. JevBench can compare baseline and E1 without changing unrelated parameters.

---

# Non-goals

Do not in this task:

```text id="fwm75f"
add few-shot examples
optimize few-shot selection
implement arbitrary choice permutations
replace the model
redesign calibration
change multilingual behavior
redesign the server
```

E1 should isolate one hypothesis:

> Does evaluating both binary candidate orders improve `noul` quality by reducing positional and candidate-token bias?

---

# Deliverables

Provide:

1. E1 implementation.
2. Configuration switch.
3. Sequential reference path.
4. Safe optimized/parallel path if supported by the current backend.
5. Unit tests with synthetic logits.
6. Sequential-vs-optimized equivalence tests where applicable.
7. JevBench comparison.
8. Latency comparison.
9. Short implementation note describing how semantic remapping works.

At completion, report:

```text id="lu1azs"
files changed
tests run

baseline:
  accuracy
  NLL
  Brier
  ECE
  latency

E1:
  accuracy
  NLL
  Brier
  ECE
  disagreement rate
  latency

implementation:
  sequential / batched / parallel
  whether KV reuse is used
```

Do not make E1 the production default solely because synthetic tests pass.

Only promote it after held-out JevBench results show that the quality gain justifies the additional inference cost.
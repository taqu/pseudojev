# Task: Implement E2 — Cyclic Option Rotation Ensemble for `choice`

## Objective

Implement a cyclic option-rotation ensemble for the `choice` primitive.

The purpose is to reduce:

- candidate-position bias,
- candidate-token bias,
- sensitivity to which semantic option is bound to `A`, `B`, `C`, etc.

For a choice task with `N` semantic options, evaluate `N` cyclic rotations so that every semantic option appears exactly once in every candidate-label position.

Example for four semantic options:

```text
Run 0:
A = option 0
B = option 1
C = option 2
D = option 3

Run 1:
A = option 1
B = option 2
C = option 3
D = option 0

Run 2:
A = option 2
B = option 3
C = option 0
D = option 1

Run 3:
A = option 3
B = option 0
C = option 1
D = option 2
```

After inference, remap all candidate scores back into semantic-option space and aggregate them there.

Do not average `A`, `B`, `C`, or `D` directly across runs.

---

# Scope

Implement E2 only for:

```text
primitive = choice
```

Do not modify `noul` E1 behavior in this task.

Do not implement all `N!` permutations.

Use exactly `N` cyclic rotations for `N` options.

The goal is to give every semantic option equal exposure to every candidate-label position while keeping inference cost linear in the number of options.

---

# Core invariant

Candidate labels and semantic options must remain separate.

A candidate label such as:

```text
A
```

is only an internal output token.

It must never be treated as the semantic option itself.

The ensemble pipeline must be:

```text
semantic options
    ↓
rotation-specific candidate binding
    ↓
candidate-token logits
    ↓
remap to semantic option IDs
    ↓
aggregate semantic evidence
    ↓
final probabilities
```

---

# Rotation generation

For `N` semantic options indexed:

```text
0, 1, 2, ..., N-1
```

define rotation `r` so candidate position `j` maps to semantic option:

```text
semantic_index = (j + r) mod N
```

Equivalent inverse mapping must also be available so candidate logits can be mapped back into semantic order.

For example, with four options:

```text
rotation 0:
candidate A -> semantic 0
candidate B -> semantic 1
candidate C -> semantic 2
candidate D -> semantic 3

rotation 1:
candidate A -> semantic 1
candidate B -> semantic 2
candidate C -> semantic 3
candidate D -> semantic 0
```

and so on.

Keep this mapping explicit in data structures.

Do not reconstruct it later from prompt text.

---

# Prompt behavior

Each rotation must preserve the same:

```text
state
question
semantic option descriptions
instruction wording
answer anchor
```

Only the option-to-candidate-label binding should change.

Example:

```text
Question:
Which action should be taken?

Possible answers:
A: Restart the server
B: Ignore the error
C: Delete the database
D: Shut down the network

Answer:
```

A later rotation might be:

```text
Possible answers:
A: Ignore the error
B: Delete the database
C: Shut down the network
D: Restart the server

Answer:
```

Do not change unrelated prompt wording between rotations.

---

# Reference aggregation method

For each rotation `r`, obtain candidate logits:

```text
L_r(A)
L_r(B)
...
```

Map them back to semantic-option logits:

```text
S_r(option_i)
```

Then aggregate in semantic score space.

The preferred initial implementation is:

```text
S_ensemble(option_i) =
    mean_r S_r(option_i)
```

Then compute:

```text
P(option_i) =
    softmax(S_ensemble / T_choice)
```

if temperature calibration is enabled.

Without temperature scaling:

```text
P(option_i) =
    softmax(S_ensemble)
```

---

# Important: aggregate logits, not labels

Do not:

```text
vote for whichever candidate letter wins
```

as the primary method.

Do not:

```text
average probability assigned to A
```

across rotations.

The semantic meaning of `A` changes between rotations.

Always remap candidate scores back to semantic option identity first.

---

# Alternative diagnostic aggregation

For research purposes only, optionally compute:

```text
mean semantic probability
majority semantic vote
```

as additional diagnostics.

Do not make these the production E2 result unless benchmark evidence later supports them.

The initial E2 definition is:

```text
mean semantic logits
→ softmax
```

---

# Prior correction

If candidate prior correction is enabled, apply it before semantic remapping.

For every rotation:

```text
corrected_logit_candidate =
    raw_logit_candidate
    - alpha * prior_logit_candidate
```

Then map:

```text
candidate space
→ semantic option space
```

Then aggregate.

This matters because prior bias belongs to the candidate token / position, not directly to the semantic option.

---

# Temperature scaling

Apply `T_choice` after semantic aggregation.

Preferred sequence:

```text
raw candidate logits
    ↓
candidate prior correction
    ↓
semantic remapping
    ↓
average semantic logits across rotations
    ↓
temperature scaling
    ↓
softmax
```

Do not independently temperature-scale each rotation and then average unless explicitly implemented as a research comparison.

---

# Configuration

Add an explicit configuration mode.

Conceptually:

```text
ChoiceEnsembleMode::NONE
ChoiceEnsembleMode::CYCLIC_ROTATION
```

or the equivalent existing project style.

Do not silently enable E2 by default.

It must remain benchmarkable against the single-order baseline.

---

# Reference implementation

First implement a simple sequential correctness path.

For each choice request:

```text
for each cyclic rotation:
    build rotated candidate mapping
    build prompt
    evaluate prompt
    capture candidate logits
    apply optional prior correction
    remap candidate logits to semantic option order
    store semantic scores

average semantic scores
apply temperature
softmax
select argmax
```

This sequential implementation is the reference behavior.

Keep it straightforward and auditable.

---

# Parallel / batched evaluation

Support an optimized evaluation path only where the backend safely permits it.

The preferred hierarchy is:

```text
1. sequential reference path
2. llama.cpp batch / multi-sequence evaluation
3. shared-prefix KV reuse
4. multiple independent contexts for true parallelism
```

Do not call the same mutable llama context concurrently from multiple threads unless the backend explicitly supports it.

If batch support already exists, prefer batching rotations over naïve multithreading.

---

# Shared-prefix optimization

All rotations share most of the prompt.

Conceptually:

```text
shared instruction
shared state
shared question
    ↓
rotation-specific options block
    ↓
Answer:
```

Where practical, structure the implementation so the shared prefix can later be evaluated once and reused.

Long-term target:

```text
shared prefix
    ↓
evaluate once
    ↓
fork KV state
   ↙   ↓   ↘
rot0 rot1 rot2 ...
   ↓   ↓   ↓
logits
    ↓
semantic aggregation
```

Do not make KV reuse mandatory for E2 correctness.

Correctness comes first.

---

# Rotation count and cost

For `N` options:

```text
number of evaluations = N
```

Examples:

```text
2 options -> 2 evaluations
3 options -> 3 evaluations
4 options -> 4 evaluations
5 options -> 5 evaluations
```

Do not generate all permutations.

For four choices:

```text
cyclic E2 cost ≈ 4x naïve baseline inference
```

before optimization.

Record this cost explicitly in benchmark output.

---

# Diagnostics

Expose enough information to understand whether rotation is helping.

Per sample, record where practical:

```text
rotation_count

for each rotation:
    candidate_to_semantic_mapping
    raw candidate logits
    corrected candidate logits
    semantic logits
    semantic prediction

ensemble:
    mean semantic logits
    final probabilities
    final prediction
```

Also calculate:

```text
rotation disagreement count
rotation disagreement rate
semantic winner frequency
```

---

# Rotation disagreement

For each rotation, convert the winning candidate back to semantic option identity.

Then determine whether all rotations agree.

For one sample, examples:

```text
rot0 -> option 2
rot1 -> option 2
rot2 -> option 2
rot3 -> option 2
```

means stable.

Whereas:

```text
rot0 -> option 2
rot1 -> option 0
rot2 -> option 2
rot3 -> option 3
```

indicates strong order sensitivity.

Aggregate metrics should include:

```text
samples_with_any_rotation_disagreement
rotation_disagreement_rate
mean_number_of_distinct_winners
```

---

# Semantic score variance

Add a rotation-stability diagnostic.

For each semantic option `i`, compute variance across rotations:

```text
Var_r(S_r(option_i))
```

Optionally summarize per sample:

```text
mean_semantic_logit_variance
max_semantic_logit_variance
```

Then aggregate across the dataset.

Low variance means candidate binding is relatively stable.

High variance means the semantic score depends heavily on option order.

This metric is useful even when final accuracy is unchanged.

---

# Signed winner margin

For labeled `choice` examples, compute a ground-truth margin.

Let:

```text
S_gt = ensemble score for ground-truth option
S_best_wrong = maximum ensemble score among all incorrect options
```

Then:

```text
signed_winner_margin =
    S_gt - S_best_wrong
```

Interpretation:

```text
> 0  correct side
< 0  incorrect side
larger positive = stronger classification
```

Report:

```text
mean signed winner margin
median signed winner margin
p10 signed winner margin
minimum signed winner margin
```

This is the multiclass equivalent of the signed semantic margin used for `noul`.

---

# E2 margin gain

When baseline and E2 results can be matched by source ID:

```text
margin_gain =
    signed_winner_margin_E2
    - signed_winner_margin_baseline
```

Report:

```text
mean_margin_gain
median_margin_gain
improved_margin_count
degraded_margin_count
unchanged_margin_count
```

This allows E2 to show improvement even if accuracy is unchanged.

---

# Metrics

For baseline and E2, report at least:

```text
accuracy
NLL
Brier
ECE

mean signed winner margin
median signed winner margin
p10 signed winner margin
min signed winner margin

rotation disagreement rate
mean semantic logit variance

eval_ms
```

Use the existing metric implementation where possible.

Do not create incompatible duplicate definitions of NLL, Brier, or ECE.

---

# Multiclass Brier score

If not already implemented, for `K` options use:

```text
Brier =
    mean_samples(
        sum_i (P_i - Y_i)^2
    )
```

where `Y_i` is one-hot ground truth.

Document whether the implementation uses the summed or normalized-by-`K` convention.

Use one convention consistently across experiments.

---

# Tests

Add focused unit tests.

## 1. Rotation mapping

For four options:

```text
semantic options:
0 1 2 3
```

verify generated rotations:

```text
rot0: 0 1 2 3
rot1: 1 2 3 0
rot2: 2 3 0 1
rot3: 3 0 1 2
```

Verify inverse mapping as well.

---

## 2. Semantic remapping

Given a rotation:

```text
A -> semantic 2
B -> semantic 3
C -> semantic 0
D -> semantic 1
```

and candidate logits:

```text
A = 1
B = 2
C = 3
D = 4
```

verify semantic logits become:

```text
semantic 0 = 3
semantic 1 = 4
semantic 2 = 1
semantic 3 = 2
```

---

## 3. Position-bias cancellation

Construct synthetic logits where every rotation strongly prefers candidate `A` regardless of meaning.

Example:

```text
A = 5
B = 1
C = 1
D = 1
```

for every rotation.

Because each semantic option appears exactly once as `A`, cyclic averaging should distribute this candidate-position bias symmetrically across semantic options.

The final semantic scores should become equal or approximately equal.

This is a key E2 test.

---

## 4. Semantic evidence survives rotation

Construct synthetic logits where the same semantic option receives the strongest semantic score in every rotation even though its candidate letter changes.

Verify the ensemble strongly selects that semantic option.

---

## 5. Prior correction

Verify candidate priors are subtracted before semantic remapping and aggregation.

---

## 6. Temperature

Verify `T_choice = 1` reproduces unscaled ensemble softmax.

---

## 7. Sequential vs optimized equivalence

If an optimized batched / KV-reuse path is implemented, verify:

```text
semantic ensemble logits
probabilities
selected option
```

match the sequential reference within floating-point tolerance.

---

# JevBench experiment

Run a controlled comparison:

```text
Baseline:
single option ordering

E2:
full cyclic rotation
```

Use the same:

```text
model
prompt wording
candidate scheme
temperature
prior correction
few-shot configuration
```

Do not change multiple experimental variables simultaneously.

Evaluate separately on:

```text
easy
original
hard
```

where those splits exist.

---

# Required report

Produce a comparison similar to:

```text
choice / original

metric                       baseline      E2
------------------------------------------------
accuracy                     ...
NLL                          ...
Brier                        ...
ECE                          ...

mean winner margin           ...
median winner margin         ...
p10 winner margin            ...
min winner margin            ...

rotation disagreement rate   n/a           ...
mean semantic variance       n/a           ...

eval_ms                      ...
```

Also report:

```text
baseline correct -> E2 wrong
baseline wrong -> E2 correct
```

counts.

---

# Acceptance criteria

E2 is correctly implemented when:

1. `N` cyclic rotations are generated for `N` options.
2. Every semantic option appears exactly once in every candidate-label position.
3. Candidate logits are remapped into semantic-option space correctly.
4. Semantic logits are aggregated, not raw candidate labels.
5. Candidate-position bias cancels in synthetic tests.
6. Consistent semantic evidence survives rotation.
7. Prior correction works in candidate space.
8. Existing choice behavior remains available as baseline.
9. Sequential reference tests pass.
10. Any optimized path matches reference output numerically.
11. JevBench can compare baseline and E2 cleanly.

---

# Non-goals

Do not in this task:

- implement all permutations;
- modify E1 `noul` behavior;
- add few-shot examples;
- optimize example selection;
- change model architecture;
- redesign calibration;
- alter multilingual behavior;
- make E2 default automatically.

This experiment should answer one question:

> Does cyclic candidate rotation improve `choice` quality by reducing option-position and candidate-token bias?

---

# Deliverables

Provide:

1. cyclic rotation generator;
2. explicit semantic mapping;
3. sequential reference ensemble;
4. optional safe batched / optimized evaluation;
5. semantic-logit aggregation;
6. E2 configuration switch;
7. unit tests;
8. JevBench baseline-vs-E2 report;
9. latency comparison;
10. rotation-stability diagnostics.

At completion, report:

```text
files changed
tests run

baseline:
  accuracy
  NLL
  Brier
  ECE
  winner-margin metrics
  eval_ms

E2:
  accuracy
  NLL
  Brier
  ECE
  winner-margin metrics
  rotation disagreement rate
  semantic logit variance
  eval_ms

implementation:
  sequential / batched / parallel
  number of rotations
  whether KV reuse is used
```

Do not promote E2 to the default path based only on training/tuning data.

Use held-out JevBench results and latency cost to decide whether it is worth enabling.
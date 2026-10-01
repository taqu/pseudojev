# Phase 0 — Experimental Decision Core

## Objective

Build the smallest possible experimental implementation that proves whether Bonsai 1.7B, running through llama.cpp, can perform Jev-style typed decisions using constrained candidate logits.

This phase is strictly an inference experiment.

Do **not** build the final CLI, HTTP server, background daemon, single-binary distribution, model embedding, or production architecture yet.

The only question Phase 0 must answer is:

> Can Bonsai 1.7B produce useful `noul`, `choice`, and `score` decisions by evaluating a constrained set of single-token candidates and converting their logits into probabilities?

The implementation should be intentionally small, easy to modify, and optimized for experimentation rather than product architecture.

---

## Background

pseudojev is intended to become a local, Jev-compatible decision engine.

The long-term design is expected to use:

- Go
- llama.cpp statically linked
- Bonsai 1.7B as the bundled default model
- CPU-first inference
- multilingual input
- Jev-compatible HTTP API
- one-file distribution
- an automatically managed background process to hide model-loading overhead

None of those product-level concerns are part of Phase 0.

Phase 0 exists only to validate the decision mechanism before investing in the surrounding product.

---

## Core Idea

Do not ask the model to generate probability values as text.

Instead:

1. Construct a prompt containing the state, question, and candidate meanings.
2. Map each possible decision to a known single-token internal candidate.
3. Restrict output to those candidate tokens.
4. Obtain the logits for those candidate tokens.
5. Normalize the candidate logits with softmax.
6. Use the resulting distribution as the decision probability distribution.
7. Optionally generate exactly one constrained token as a sanity check that the selected token matches the maximum-probability candidate.

The model is therefore being used as a constrained semantic classifier rather than as a text generator.

Conceptually:

```text
state + question + candidate descriptions
                |
                v
            prompt
                |
                v
         Bonsai 1.7B
                |
                v
     candidate token logits
                |
                v
       restricted softmax
                |
                v
       probability vector
```

---

## Scope

Implement support for exactly three experimental decision types:

1. `noul`
2. `choice`
3. `score`

No additional decision types are required.

---

# 1. Noul

`noul` represents a binary decision.

Example conceptual input:

```json
{
  "state": "The customer says they were charged twice and want their money back.",
  "question": "Is the customer requesting a refund?"
}
```

Internally map the two possible outcomes to two known single-token candidates.

For example:

```text
A = false
B = true
```

The actual candidate tokens do not need to be `A` and `B`. Choose tokens only after verifying that they are each represented by exactly one tokenizer token.

Obtain:

```text
logit_false
logit_true
```

Then calculate:

```text
p_false = exp(logit_false) /
          (exp(logit_false) + exp(logit_true))

p_true = exp(logit_true) /
         (exp(logit_false) + exp(logit_true))
```

Return both probabilities internally.

The primary `noul` value should be:

```text
p_true
```

Example experimental result:

```json
{
  "type": "noul",
  "probabilities": {
    "false": 0.04,
    "true": 0.96
  },
  "value": 0.96,
  "selected": true
}
```

Exact output formatting is not important yet.

---

# 2. Choice

`choice` represents selection among N unordered alternatives.

Example:

```json
{
  "state": "The customer was charged twice and wants the duplicate charge returned.",
  "question": "Which queue should handle this ticket?",
  "choices": {
    "billing": "Invoices, charges, payments, and refunds",
    "technical": "Product failures and API errors",
    "sales": "Questions before purchasing"
  }
}
```

Do not attempt to generate the strings:

```text
billing
technical
sales
```

directly.

Instead map them to internal single-token candidates:

```text
A = billing
B = technical
C = sales
```

The semantic meaning must be present in the prompt, while the generated/evaluated output tokens remain simple internal IDs.

For example:

```text
A: Invoices, charges, payments, and refunds
B: Product failures and API errors
C: Questions before purchasing
```

At the decision position, retrieve the logits corresponding to:

```text
A
B
C
```

Normalize only these candidate logits:

```text
P(candidate_i) =
    exp(logit_i) /
    sum(exp(logit_j))
```

Return:

- the probability of every candidate,
- the selected candidate,
- optionally the raw candidate logits for debugging.

Example:

```json
{
  "type": "choice",
  "probabilities": {
    "billing": 0.93,
    "technical": 0.05,
    "sales": 0.02
  },
  "selected": "billing"
}
```

---

# 3. Score

`score` represents an ordered set of levels.

Example:

```json
{
  "state": "I have contacted support three times and nobody has fixed this. This is ridiculous.",
  "question": "How frustrated is the customer?",
  "levels": [
    "Calm",
    "Mildly annoyed",
    "Frustrated",
    "Very angry"
  ]
}
```

Map the ordered levels to internal candidates:

```text
A = Calm
B = Mildly annoyed
C = Frustrated
D = Very angry
```

Retrieve and normalize the candidate logits exactly as for `choice`.

Then calculate the expected score over the zero-based level indices:

```text
score =
    sum(index_i * probability_i)
```

For:

```text
P(A) = 0.05
P(B) = 0.20
P(C) = 0.65
P(D) = 0.10
```

the expected score is:

```text
0 * 0.05 +
1 * 0.20 +
2 * 0.65 +
3 * 0.10
= 1.80
```

Return both:

- the complete probability distribution,
- the expected score.

Example:

```json
{
  "type": "score",
  "probabilities": [
    0.05,
    0.20,
    0.65,
    0.10
  ],
  "score": 1.80,
  "selected_level": 2
}
```

---

# Candidate Token Requirements

Candidate token selection is critical.

Do not assume that arbitrary strings are single tokens.

Before using any candidate token, verify through the llama.cpp tokenizer that:

```text
candidate text -> exactly one token ID
```

The experimental code should expose or log the mapping:

```text
candidate A -> token ID ...
candidate B -> token ID ...
candidate C -> token ID ...
```

Avoid using arbitrary user-provided labels as output tokens.

User labels may:

- contain multiple tokenizer tokens,
- be multilingual,
- contain whitespace,
- overlap semantically,
- have tokenizer-dependent behavior.

Instead, always separate:

```text
semantic label
```

from:

```text
internal candidate token
```

For Phase 0, supporting a limited maximum number of candidates is acceptable.

For example:

```text
maximum choice candidates: 10
maximum score levels: 10
```

The exact limit is not important as long as it is documented.

---

# Logit Handling

The decision probability must be calculated from the candidate logits only.

Do not normalize over the entire vocabulary.

For candidate logits:

```text
l_1, l_2, ..., l_n
```

calculate a numerically stable softmax:

```text
m = max(l_i)

p_i =
    exp(l_i - m) /
    sum(exp(l_j - m))
```

Validate that:

```text
sum(p_i) ~= 1.0
```

within normal floating-point tolerance.

Reject or clearly report invalid results such as:

- NaN
- infinity
- empty candidate set
- duplicate candidate token IDs

---

# Constrained One-Token Generation

A preliminary experiment has already established that logits can be retrieved and used for probability calculation.

It should also be possible to use logit bias or equivalent llama.cpp sampling controls to constrain generation to the candidate set and generate exactly one token.

Implement this only as a validation mechanism.

For each decision, verify where practical that:

```text
generated constrained token
```

matches:

```text
argmax(candidate probabilities)
```

The probability calculation itself must remain based on the candidate logits.

Do not derive probabilities from repeated sampling.

Do not perform Monte Carlo sampling.

Do not generate explanatory text.

---

# Prompt Design

Keep prompt construction explicit and easy to modify.

Do not create a large abstraction framework in this phase.

A simple structure is sufficient.

For example:

```text
You are performing a classification task.

State:
<state>

Question:
<question>

Possible answers:
A: <description>
B: <description>
C: <description>

Return exactly one answer token.

Answer:
```

This example is not mandatory.

Experiment with formatting only as necessary to obtain a functional baseline.

Do not spend significant time optimizing prompts during Phase 0. Prompt optimization belongs to a later phase.

The important property is that the prompt clearly communicates:

- the state,
- the decision question,
- the meaning of every candidate,
- that exactly one candidate must be selected.

---

# Multilingual Input

The implementation must not make assumptions that the state or question is English.

It must accept arbitrary UTF-8 input.

At minimum, manually verify that the inference pipeline works mechanically with:

- English
- Japanese

This Phase does not require proving multilingual quality.

The goal is only to ensure that multilingual input does not break tokenization, prompt construction, or decision extraction.

Actual multilingual benchmarking belongs to a later phase.

---

# Experimental Interface

Use the simplest interface that makes repeated testing convenient.

A small executable that accepts JSON input is sufficient.

For example:

```bash
./phase0-test case.json
```

or:

```bash
echo '<json>' | ./phase0-test
```

A hard-coded test program is also acceptable initially if it accelerates development.

Do not implement:

- a public CLI UX,
- subcommand architecture,
- HTTP,
- daemon IPC,
- configuration files,
- installation logic.

The interface only needs to support developer experimentation.

---

# Suggested Internal Data Model

Keep the internal representation small.

Something similar to the following is sufficient:

```text
DecisionRequest
    state
    question
    type
    candidates

Candidate
    key
    description
    token_id

DecisionResult
    probabilities
    selected_index
    raw_logits
```

For `score`, additionally calculate:

```text
expected_score
```

For `noul`, additionally expose:

```text
p_true
```

Do not over-engineer type hierarchies.

---

# Debug Output

Phase 0 should make debugging easy.

Provide a verbose/debug mode that can show:

```text
prompt
candidate label -> internal token
candidate token ID
candidate raw logit
candidate normalized probability
selected candidate
generated constrained token
inference duration
```

Example:

```text
candidate:
  internal: A
  label: billing
  token_id: 32
  logit: 8.132
  probability: 0.9271
```

Debug information does not need a stable format.

---

# Determinism

Repeated evaluation of the same input should produce the same probability distribution within expected floating-point tolerance.

Use deterministic inference settings wherever possible.

Avoid introducing unnecessary randomness.

If one-token generation is used for validation, configure it so that the selected candidate is deterministic.

---

# Minimal Test Cases

Add small deterministic smoke tests or test fixtures covering at least the following.

## Noul

Clearly positive example:

```text
State:
"Please refund the duplicate charge."

Question:
"Is the user asking for a refund?"
```

Expected qualitative result:

```text
P(true) > P(false)
```

Clearly negative example:

```text
State:
"Thanks, everything is working now."

Question:
"Is the user reporting a problem?"
```

Expected qualitative result:

```text
P(false) > P(true)
```

## Choice

Example:

```text
State:
"The customer was billed twice."

Choices:
billing
technical
sales
```

Expected qualitative result:

```text
P(billing) > P(technical)
P(billing) > P(sales)
```

## Score

Use one obviously low and one obviously high example and verify that the expected scores are ordered correctly.

Do not encode exact probability thresholds into tests at this stage.

The purpose is to catch broken prompting, candidate mapping, and logit extraction.

---

# Small JevBench Smoke Test

After the three primitives work, run a very small subset of JevBench through the experimental core.

Do not implement the complete benchmark integration yet.

Select enough examples to include:

- several `noul` cases,
- several `choice` cases,
- several `score` cases.

The objective is only to confirm that real Jev-style tasks can flow through the system end to end.

Record:

```text
expected answer
predicted answer
candidate probabilities
latency
```

A simple CSV, JSONL, or console table is sufficient.

---

# Performance Measurements

Record basic timing information for each request.

At minimum distinguish:

```text
model load time
prompt evaluation time
decision/logit evaluation time
total inference time
```

Also record approximate process RSS if convenient.

Do not optimize these numbers yet.

They are baseline measurements for later phases.

---

# Non-Goals

The following are explicitly out of scope for Phase 0.

Do not implement them unless absolutely necessary for the experiment.

## No HTTP Server

Do not implement:

```text
POST /api/v1/systemone/
```

That begins in Phase 1.

## No Jev API Compatibility Layer

Internal experimental data structures are sufficient.

Do not spend time matching exact Jev request or response JSON.

## No Production CLI

Do not design the final pseudojev command-line interface.

## No Background Process

Do not implement:

- daemonization,
- Unix domain sockets,
- named pipes,
- automatic process startup,
- idle shutdown.

## No Single-Binary Packaging

The model may remain an external GGUF file.

Do not embed the model into the executable.

## No Static-Linking Requirement

Use the easiest llama.cpp integration available for the experiment.

Static linking will be addressed later.

## No Model Downloading

Assume the Bonsai GGUF already exists locally.

## No Authentication

No API keys or authorization handling.

## No Concurrency

Single-request sequential inference is sufficient.

## No Production Logging

Simple debug output is enough.

## No Calibration Training

Do not implement:

- temperature scaling,
- isotonic regression,
- Platt scaling,
- per-language calibration,
- per-task calibration.

Use raw restricted-softmax probabilities.

Calibration is a later phase.

## No Extensive Prompt Optimization

Do not perform broad prompt searches.

A functional baseline is sufficient.

## No Alternative Model Benchmarking

Use Bonsai 1.7B only.

Model comparison belongs after a baseline exists.

---

# Implementation Priorities

When tradeoffs are necessary, use this priority order:

1. Correct candidate logit extraction
2. Correct restricted softmax calculation
3. Reliable single-token candidate mapping
4. Correct `noul`, `choice`, and `score` behavior
5. Easy experimentation and debugging
6. Deterministic output
7. Basic measurement
8. Code cleanliness

Do not prioritize production architecture over experimental velocity.

---

# Phase 0 Deliverables

The completed Phase 0 should contain:

1. A runnable experimental program using Bonsai 1.7B through llama.cpp.
2. Candidate-token validation.
3. Candidate logit extraction.
4. Restricted softmax calculation.
5. `noul` implementation.
6. `choice` implementation.
7. `score` implementation including expected-value calculation.
8. Optional constrained one-token generation used as an argmax sanity check.
9. Debug output showing candidate logits and probabilities.
10. Basic English and Japanese smoke cases.
11. A small JevBench smoke run.
12. Basic latency measurements.
13. A short README explaining how to run the experiment and interpret its output.

---

# Exit Criteria

Phase 0 is complete when all of the following are true.

### Functional

All three decision primitives work:

```text
noul
choice
score
```

### Candidate handling

Every output candidate used by the inference path is verified to correspond to exactly one tokenizer token.

### Probability handling

For every successful decision:

```text
0 <= probability_i <= 1
```

and:

```text
sum(probability_i) ~= 1
```

No NaN or infinite values occur in normal cases.

### Noul

The implementation returns a binary probability distribution and exposes the probability of the positive/true outcome.

### Choice

The implementation returns an N-way probability distribution and identifies the maximum-probability choice.

### Score

The implementation returns the level probability distribution and computes the expected numeric score correctly.

### Constrained decoding

At least one test demonstrates that generation can be restricted to the candidate-token set and limited to one generated token.

The generated token should agree with the maximum candidate logit under deterministic decoding.

### Real-task smoke test

A small mixed subset of JevBench runs successfully through the experimental implementation.

The purpose is not to reach any particular benchmark score yet.

### Multilingual mechanics

At least one English case and one Japanese case run successfully through the complete inference pipeline.

### Reproducibility

Repeated evaluation of identical input produces stable results.

---

# Do Not Optimize Prematurely

The most important rule for this phase is:

> Do not build pseudojev yet. Prove that its inference primitive works.

If a design decision does not materially help answer whether Bonsai 1.7B can perform Jev-style decisions from constrained logits, postpone it.

The expected Phase 0 result is a small experimental codebase that may later be refactored or partially discarded.

That is acceptable.

The goal of this phase is evidence, not architecture.

---

# Final Report

At the end of the phase, provide a short report containing:

## Implementation

Describe:

- how candidate tokens were chosen,
- how logits were obtained,
- how candidate restriction was applied,
- how probabilities were calculated,
- how one-token constrained decoding was implemented.

## Results

Report:

- `noul` smoke-test results,
- `choice` smoke-test results,
- `score` smoke-test results,
- English/Japanese smoke-test observations,
- small JevBench smoke-test results,
- model load time,
- representative inference latency.

## Problems Found

Document any issues involving:

- tokenizer behavior,
- whitespace-sensitive candidate tokens,
- BOS/EOS behavior,
- candidate ordering,
- unexpected tokenization,
- llama.cpp logit access,
- logit bias,
- deterministic decoding,
- numerical stability.

## Recommendation

Conclude with one of:

```text
Proceed to Phase 1
```

or:

```text
Do not proceed yet
```

and explain the technical reason.

Do not base this recommendation on production polish.

Base it only on whether the constrained-logit decision approach appears technically viable.

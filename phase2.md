# Phase 2 — Decision Quality Baseline

## Objective

Implement Phase 2 of **pjev**.

The purpose of this phase is to understand and improve the decision quality limitations observed in Phase 0/Phase 1.

The goal is NOT to replace the model.

The goal is:

> identify which part of the decision formulation causes JevBench failures.

The current hypothesis is that many failures are not purely model capability problems.

The experiments should determine whether errors come from:

```text
candidate representation
prompt formulation
option ordering bias
candidate token priors
score encoding
```

The primary evaluation target is the existing JevBench workflow.

Do not introduce a new benchmark.

Do not optimize for a single benchmark score without understanding the mechanism.

---

# 1. Project Context

The project command name is:

```bash
pjev
```

Use `pjev` consistently in:

- documentation
- command examples
- scripts
- benchmark commands
- diagnostics
- configuration names

Do not use previous temporary names such as:

```text
pseudojev-server
pseudojev-cli
jev-compatible-server
```

unless referring to historical architecture.

The intended product direction is:

```text
pjev
 ├── server
 ├── decision engine
 ├── benchmark tooling
 └── future CLI
```

Phase 2 should continue using the Phase 1 native C++ architecture.

---

# 2. Phase 1 Assumptions

Assume Phase 1 has already implemented:

```text
HTTP API
    ↓
Jev compatibility layer
    ↓
DecisionEngine
    ↓
PromptStrategy
    ↓
LlamaBackend
    ↓
llama.cpp
```

Do not bypass these boundaries.

All experiments should run through the same decision path used by the pjev server.

Do not create separate experimental inference implementations.

The benchmark should measure the actual pjev behavior.

---

# 3. Main Research Questions

Implement infrastructure to answer:

## Question A

How much error comes from candidate token representation?

Example:

```text
semantic answer
        ↓
candidate token
```

Is the mapping itself introducing bias?

---

## Question B

How much error comes from option ordering?

Example:

Same question:

```text
A B C D
```

versus:

```text
D C B A
```

Does the model choose different answers?

---

## Question C

How much error comes from prompt layout?

Example:

```text
state
question
options
```

versus:

```text
question
options
state
```

---

## Question D

How much error comes from candidate token priors?

Does a token have a preference independent of the actual decision?

---

## Question E

Why is score prediction unstable?

Can better representation improve:

```text
accuracy
MAE
QWK
ordering consistency
```

---

# 4. Experimental Framework Requirement

Before implementing individual experiments, create a reusable experiment framework.

The framework should allow:

```text
experiment configuration
        ↓
pjev DecisionEngine
        ↓
JevBench evaluation
        ↓
metrics collection
        ↓
comparison report
```

Do not implement each experiment as a one-off script.

The goal is to make future formulation experiments cheap.

---

# 5. Configuration-Based Experiments

Experimental variables should be configurable.

Avoid hard-coded branches.

At minimum support:

```yaml
prompt:
  layout:
    auto
    state-first
    state-last

candidate:
  scheme:
    natural
    letters
```

The exact configuration format is flexible.

Possible implementations:

```text
JSON
YAML
command line options
C++ configuration object
```

Choose what fits the existing pjev architecture.

---

# 6. Candidate Binding Experiments

Implement Phase 2A.

The purpose is to compare candidate representations.

Support experiments with:

```text
A/B/C/D

0/1/2/3

Yes/No

True/False

custom single-token IDs
```

The important question:

> Does the semantic answer survive the conversion into candidate tokens?

---

## Required Metrics

For each candidate scheme, measure:

```text
choice accuracy
noul accuracy
score MAE
```

Also record:

```text
invalid candidate rate
tokenization behavior
candidate token IDs
```

because candidate-token mistakes can invalidate the experiment.

---

# 7. Candidate Token Validation

Extend existing validation.

For every candidate representation, verify:

```text
candidate text
        ↓
tokenizer
        ↓
single token?
        ↓
candidate logit extraction possible?
```

If a candidate is not a valid single token:

- report it clearly;
- do not silently continue;
- allow the experiment runner to skip or mark it invalid.

Do not hide tokenization failures.

---

# 8. Option Order Sensitivity Experiment

Implement Phase 2B.

For a given benchmark sample, generate multiple option permutations.

At minimum:

```text
original order

reverse order

random permutation
```

The experiment runner should keep the underlying question identical.

Only option order should change.

---

## Required Metrics

Measure:

```text
prediction stability

probability variance

accuracy variance
```

Also calculate:

```text
semantic answer consistency
```

Meaning:

Does pjev produce the same semantic answer after option ordering changes?

---

# 9. Prompt Layout Experiment

Implement Phase 2C.

Support controlled comparison of prompt layouts.

At minimum:

```text
state-first:

state
question
options
```

and:

```text
state-last:

question
options
state
```

Also support:

```text
question
state
options
```

if the current PromptStrategy abstraction makes this easy.

---

## Important Constraint

Do not create a massive prompt search system.

This phase is about understanding major formulation effects.

Use a small number of clearly defined templates.

---

# 10. Noul Prior Correction Experiment

Implement Phase 2D.

The goal is to determine whether candidate-token priors distort prediction.

Create a content-free baseline.

Measure:

```text
prior(candidate)
```

Conceptually:

```text
corrected_logit_i =
    raw_logit_i - prior_logit_i
```

The exact implementation may differ.

Do not implement a full calibration system.

That belongs to Phase 3.

The goal here is:

> separate candidate-token prior effects from semantic evidence.

---

# 11. Score Formulation Experiment

Implement Phase 2E.

Score is currently the least stable primitive.

Evaluate alternative representations.

Support:

```text
ordinal labels

digits

letters

semantic anchor text
```

while preserving the existing expected-value approach where possible.

---

## Required Metrics

For score experiments collect:

```text
accuracy

MAE

QWK

ordering consistency
```

Do not report only accuracy.

Score quality is not a classification problem only.

---

# 12. Experiment Result Format

Create a machine-readable result format.

Each experiment run should record:

```json
{
  "experiment": "...",
  "model": "...",
  "layout": "...",
  "candidate_scheme": "...",
  "metrics": {
    "choice_accuracy": 0.0,
    "noul_accuracy": 0.0,
    "score_mae": 0.0
  }
}
```

The exact schema can differ.

The important requirement:

Results from different experiments must be comparable.

---

# 13. pjev Commands

Add experiment commands under the pjev namespace.

Do not create unrelated standalone binaries.

Examples:

```bash
pjev benchmark run
```

or:

```bash
pjev experiment candidate-binding
```

or equivalent.

Choose a command structure consistent with the existing Phase 1 design.

The command naming should clearly communicate that these are pjev operations.

---

# 14. Benchmark Integration

JevBench remains the primary evaluation system.

The workflow should be:

```text
modify formulation
        ↓
run pjev experiment
        ↓
run JevBench
        ↓
compare results
```

Do not create a private benchmark that replaces JevBench.

---

# 15. Regression Protection

Every formulation experiment must preserve the ability to return to the Phase 1 default configuration.

The default pjev behavior should remain reproducible.

Store experiment configuration explicitly.

Do not modify global defaults during experiments.

---

# 16. Testing Requirements

Add tests for:

## Candidate Binding

```text
candidate scheme parsing

single-token validation

candidate mapping

invalid candidate handling
```

## Prompt Layout

```text
state-first rendering

state-last rendering

configuration switching
```

## Prior Correction

```text
prior calculation

corrected logit calculation
```

## Experiment Framework

```text
configuration loading

result serialization

metric collection
```

## Integration

At least one test should verify:

```text
pjev configuration
        ↓
DecisionEngine
        ↓
llama.cpp backend
        ↓
result output
```

---

# 17. Performance Constraints

Do not optimize performance in Phase 2.

Do not implement:

```text
KV cache optimization

batch scheduling

thread tuning

model replacement

GPU acceleration work
```

Those belong to later phases.

Correct experiments are more important than fast experiments.

---

# 18. Model Replacement Restriction

Do not compare other models in Phase 2.

The purpose of this phase is:

> determine whether Bonsai 1.7B performance can be improved by better formulation.

Model comparison belongs to Phase 5.

Do not conclude that the model is insufficient before formulation issues are investigated.

---

# 19. Phase 2 Exit Criteria

Phase 2 is complete when:

```text
[ ] pjev has a reusable experiment framework.

[ ] Candidate representation experiments can be executed.

[ ] Option-order sensitivity can be measured.

[ ] Prompt layout can be changed through configuration.

[ ] Candidate prior effects can be measured.

[ ] Score formulation alternatives can be compared.

[ ] All experiments use the same pjev DecisionEngine path.

[ ] Results are stored in a comparable format.

[ ] JevBench can be run after each formulation change.

[ ] The best findings are documented.
```

---

# 20. Final Implementation Report

When complete, provide:

## Architecture Changes

Explain:

```text
new experiment framework
new configurations
new pjev commands
```

## Experiments Implemented

List:

```text
candidate binding
option ordering
prompt layout
prior correction
score formulation
```

## Commands

Provide exact examples:

```bash
pjev ...
```

## Results

For each experiment report:

```text
baseline
changed variable
metrics
interpretation
```

Do not only report the best score.

Report what was learned.

## Remaining Questions

Document unresolved issues for Phase 3 and Phase 5.

---

# Final Instruction

The purpose of Phase 2 is not "make the benchmark number higher".

The purpose is:

> make pjev's decision mechanism understandable enough that future improvements are based on evidence.

Prioritize:

```text
measurement
reproducibility
controlled experiments
clean configuration
```

over:

```text
quick score improvements
large refactors
new features
```

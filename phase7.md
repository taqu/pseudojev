# Phase 7 — Performance Optimization

## Objective

Implement Phase 7 of **pjev**.

The purpose of this phase is to improve runtime performance of the already-correct and production-hardened pjev server.

The command name is:

```bash
pjev
```

The primary runtime remains:

```bash
pjev serve
```

The main optimization target is:

> prompt evaluation cost

Previous measurements showed that decision math itself is effectively negligible compared with model prompt evaluation.

Therefore, optimize the parts of the runtime that reduce or avoid repeated llama.cpp prompt processing.

Do not change decision semantics merely to make benchmarks faster.

---

# 1. Project Context

Assume previous phases have already established:

```text
Phase 1:
native C++ / llama.cpp architecture

Phase 2:
stable decision formulation

Phase 3:
calibration

Phase 4:
multilingual validation

Phase 5:
selected default model

Phase 6:
production server
```

Phase 7 must preserve all behavior validated in those phases.

The architecture remains:

```text
HTTP API
   ↓
Jev compatibility layer
   ↓
DecisionEngine
   ↓
PromptStrategy / Calibration
   ↓
LlamaBackend
   ↓
llama.cpp
```

Do not introduce a second inference path.

---

# 2. Primary Optimization Targets

Investigate and optimize:

```text
prompt length
KV reuse
multiple-question batching
context sizing
thread count
llama.cpp runtime settings
```

The highest-value hypothesis is:

```text
same state
+
multiple questions
```

may allow reuse of shared prompt/KV work.

Prioritize measurement and evidence.

Do not optimize blindly.

---

# 3. Core Principle

Every optimization must preserve:

```text
semantic prediction
candidate mapping
candidate logits within expected numerical tolerance
probabilities
calibration behavior
Jev API behavior
```

Performance improvements must not silently change decision quality.

If an optimization changes outputs, treat that as a correctness issue until proven otherwise.

---

# 4. Establish a Performance Baseline First

Before changing implementation, measure the current Phase 6 runtime.

Create a reproducible baseline covering:

```text
model load time
single-request latency
prompt evaluation time
candidate/logit processing time
total decision time
RSS
prompt token count
```

Measure separately for:

```text
noul
choice
score
```

where useful.

Do not use historical Phase 0 measurements as the final baseline.

Re-measure the current implementation.

---

# 5. Measurement Environment

Record enough information to reproduce every performance result.

At minimum:

```text
OS
CPU model
architecture
logical/physical cores where available
RAM
pjev version / git commit
llama.cpp revision
model hash
quantization
thread configuration
context size
batch settings
```

Do not compare measurements collected under materially different environments without labeling them.

---

# 6. Benchmark Workloads

Define representative workloads rather than one synthetic request.

Include:

```text
short prompt
typical prompt
long prompt
```

and:

```text
single question
multiple questions sharing one state
```

Where possible, derive workloads from real JevBench examples.

Do not create benchmarks that are much easier than actual pjev usage.

---

# 7. Separate Cold and Warm Performance

Measure separately:

```text
cold startup / model load
warm inference
```

Phase 7 should primarily optimize warm request processing.

Do not mix model startup time into ordinary request latency statistics.

---

# 8. Timing Boundaries

Instrument coarse but meaningful timing boundaries.

At minimum measure:

```text
request parsing
prompt construction
tokenization
prompt evaluation
logit extraction
restricted softmax / decision math
response serialization
```

Avoid overly granular profiling unless necessary.

The goal is to identify real bottlenecks.

---

# 9. Decision Math Sanity Check

Verify again that:

```text
candidate-logit extraction
restricted softmax
argmax
expected-value calculation
calibration
```

are negligible compared with prompt evaluation.

Do not spend time micro-optimizing sub-millisecond decision math unless measurements show it has become significant.

---

# 10. Prompt Length Optimization

Investigate whether current prompts contain unnecessary repeated text.

Measure:

```text
prompt token count
prompt evaluation latency
quality metrics
```

for any proposed reduction.

Possible targets include:

```text
redundant instructions
repeated formatting
unnecessary whitespace
duplicated context
verbose candidate explanations
```

Do not change semantic instructions casually.

Any prompt shortening must be validated against the existing quality benchmark.

---

# 11. Prompt Optimization Guardrail

A shorter prompt is not automatically better.

For each prompt-length optimization report:

```text
tokens before
tokens after
latency before
latency after
quality before
quality after
```

Reject changes that materially damage:

```text
choice accuracy
noul accuracy
score MAE
calibration
multilingual behavior
```

unless explicitly justified.

---

# 12. Prompt Template Stability

Do not turn Phase 7 into another Phase 2 prompt search.

Only remove or restructure text for measured performance reasons.

Do not broadly experiment with new semantic formulations.

The selected Phase 5/6 formulation remains the reference.

---

# 13. KV Cache Reuse

Investigate llama.cpp KV-cache reuse for requests that share prompt prefixes.

The main target is:

```text
same state
+
multiple questions
```

For example:

```text
shared state prefix
        ↓
evaluate once
        ↓
reuse cached state representation
        ↓
evaluate question A suffix
        ↓
evaluate question B suffix
        ↓
evaluate question C suffix
```

The exact implementation depends on llama.cpp APIs and current prompt layout.

---

# 14. Do Not Assume KV Reuse Is Free

Measure:

```text
reuse setup overhead
memory impact
cache copy/restore cost
latency savings
correctness
```

A complex KV strategy is not worthwhile if savings are small.

Use evidence.

---

# 15. Prefix Identity Requirement

Only reuse KV state when the shared token prefix is exactly compatible.

Do not assume semantically similar strings are reusable.

Reuse must depend on exact tokenized prefix identity or another provably safe mechanism supported by llama.cpp.

---

# 16. Prompt Layout Implications

Existing prompt layout may affect KV reuse.

For example:

```text
state-first
```

may offer a reusable shared prefix more naturally than:

```text
state-last
```

Do not automatically change the selected layout for performance.

Instead measure the tradeoff:

```text
quality
vs
reuse opportunity
```

If changing layout would improve performance but reduce decision quality, report the tradeoff rather than silently switching.

---

# 17. Multiple Questions Per Request

The Jev API may contain multiple questions in one request.

Optimize this case explicitly.

Avoid treating every question as an entirely independent model evaluation if significant prompt state is shared.

Create an execution path conceptually like:

```text
request
  ↓
shared state preparation
  ↓
question 1
question 2
question 3
```

while keeping single-question behavior unchanged.

---

# 18. Batch Semantics

Do not confuse:

```text
multiple independent HTTP requests
```

with:

```text
multiple questions within one semantic Jev request
```

Phase 7 should first optimize the latter because shared state is explicit and safe to exploit.

Cross-request batching is not required.

---

# 19. Question Batching

Investigate llama.cpp batching APIs where they provide a real benefit.

Potential opportunities:

```text
parallel suffix evaluation
batched token evaluation
shared-prefix branching
```

Use the APIs available in the pinned llama.cpp version.

Do not rely on undocumented assumptions.

---

# 20. Numerical Equivalence for Reuse

For KV reuse/batching, compare optimized output against the unoptimized reference.

For each question verify:

```text
candidate token IDs identical
candidate logits equivalent within tolerance
argmax identical
probabilities equivalent within tolerance
score expected value equivalent within tolerance
```

Do not accept silent semantic drift.

---

# 21. Deterministic Reference Mode

Maintain a simple reference execution mode.

Conceptually:

```text
no KV reuse
no batching
baseline settings
```

This mode should remain available for:

```text
correctness comparison
regression tests
debugging
```

Do not remove the simple path immediately after introducing optimization.

---

# 22. Context Sizing

Investigate whether the current llama.cpp context size is unnecessarily large.

Measure effects of context size on:

```text
RSS
startup cost
inference performance
maximum supported request size
```

Use the smallest context that safely supports intended workloads plus reasonable margin.

Do not shrink context below real request requirements.

---

# 23. Context Size Policy

Avoid a hard-coded oversized context if a smaller default is sufficient.

Where useful, support explicit configuration such as:

```bash
pjev serve --context-size ...
```

or the project's existing configuration equivalent.

Validate the requested value before server startup.

---

# 24. Dynamic Context Sizing

Do not build complicated dynamic context allocation unless measurements justify it.

A well-chosen configurable default is sufficient for Phase 7.

Prefer simplicity.

---

# 25. Thread Count Optimization

Benchmark llama.cpp thread counts.

At minimum compare a small useful set around:

```text
physical core count
logical core count
current default
```

Do not assume "more threads = faster."

Prompt evaluation may stop scaling or regress due to contention.

---

# 26. Thread Benchmark Method

For each tested thread count record:

```text
median latency
p90/p95 latency
CPU utilization where practical
RSS
```

Use the same workloads.

Run enough iterations to reduce noise.

---

# 27. Default Thread Policy

If evidence supports a better default, implement it.

Prefer an automatic policy based on CPU detection if the code already has reliable platform information.

Otherwise keep explicit configuration simple.

Do not introduce fragile topology detection solely for this phase.

---

# 28. llama.cpp Settings

Review runtime settings relevant to prompt evaluation.

Potential areas may include:

```text
n_threads
n_threads_batch
n_batch
n_ubatch
context size
memory mapping behavior
backend settings
```

Only optimize settings actually supported and meaningful for the pinned llama.cpp revision.

Do not copy tuning advice blindly from unrelated model/runtime configurations.

---

# 29. Pinned llama.cpp Version

Performance experiments must use a known llama.cpp revision.

Do not simultaneously:

```text
upgrade llama.cpp
and
change runtime tuning
```

unless necessary.

That makes causality difficult to interpret.

If upgrading llama.cpp is explicitly tested, treat it as its own experimental variable.

---

# 30. Memory Mapping

If the current runtime uses mmap, measure its actual behavior.

Do not disable or enable mmap simply based on intuition.

Record effects on:

```text
load latency
RSS
warm latency
```

where relevant.

---

# 31. Model Load Is Secondary

Model load time matters, but it is not the primary Phase 7 target.

Future background runtime support can hide startup cost.

Focus first on costs paid repeatedly per request.

Do not sacrifice warm inference quality/performance solely to shave a small amount from startup.

---

# 32. RSS Monitoring

Track memory impact of all optimizations.

KV caches and larger batches may trade latency for memory.

For every major optimization report:

```text
latency delta
RSS delta
```

Do not optimize latency without observing memory cost.

---

# 33. Performance Artifact

Produce a machine-readable performance artifact.

Conceptually:

```json
{
  "configuration": {
    "threads": 8,
    "context_size": 4096,
    "kv_reuse": true,
    "batch_questions": true
  },
  "workload": {
    "questions": 4,
    "prompt_tokens": 620
  },
  "metrics": {
    "p50_ms": 0,
    "p95_ms": 0,
    "rss_bytes": 0
  }
}
```

Exact schema may differ.

The important requirement is reproducible comparison.

---

# 34. Baseline vs Optimized Reporting

Every optimization experiment should report:

```text
baseline
variant
delta
percentage delta
```

For:

```text
latency
prompt evaluation time
RSS
token count
```

where relevant.

Also report quality regression checks.

---

# 35. Avoid a Single Synthetic Score

Do not create a composite "performance score."

Keep:

```text
latency
memory
quality
```

visible separately.

A faster configuration that uses much more memory should remain obviously different.

---

# 36. pjev Benchmark Commands

Add performance benchmarking under the `pjev` command namespace.

Conceptually:

```bash
pjev benchmark performance ...
```

or:

```bash
pjev benchmark latency ...
```

Use the command structure already established by the project.

Do not create a separate benchmarking executable unless technically required.

---

# 37. Benchmark Repetition

Allow repeated runs.

A useful command should support or internally perform enough iterations for stable measurements.

Report at least:

```text
count
median
p90
p95
min/max where useful
```

Do not report only the fastest run.

---

# 38. Warm-Up

Explicitly define warm-up behavior.

For example:

```text
load model
run N warm-up requests
discard warm-up timings
measure subsequent requests
```

Use the same method across variants.

Document it.

---

# 39. CPU Frequency Noise

Do not overengineer system benchmarking infrastructure, but acknowledge normal CPU timing noise.

Where practical:

```text
run multiple repetitions
avoid mixing cold/warm results
report medians
```

This is sufficient.

---

# 40. Single-Question Regression

Any optimization for multi-question requests must not significantly regress single-question latency.

Measure both:

```text
1 question
N questions sharing state
```

A special multi-question path may be justified, but ordinary requests must remain healthy.

---

# 41. Multi-Question Scaling

Measure total and per-question latency for:

```text
1 question
2 questions
4 questions
8 questions
```

or another practical set supported by the API.

Report:

```text
total latency
latency per question
speedup vs independent evaluation
```

This is one of the most important Phase 7 outputs.

---

# 42. Speedup Definition

Define speedup explicitly.

For example:

```text
speedup =
    independent_total_time /
    optimized_total_time
```

Do not use ambiguous statements such as:

```text
"2x faster"
```

without defining the baseline.

---

# 43. Shared-State Benchmark

Create a dedicated benchmark for:

```text
one state
multiple questions
```

This should become a permanent performance regression test.

Use realistic state lengths.

---

# 44. Different-State Control Benchmark

Also benchmark:

```text
different state
multiple questions
```

as a control.

This helps confirm that gains actually come from shared-prefix reuse rather than unrelated changes.

---

# 45. Cache Correctness

KV reuse must never leak state between unrelated requests or questions.

Add tests ensuring:

```text
request A state
```

cannot affect:

```text
request B
```

This is both a correctness and privacy requirement.

---

# 46. Cache Lifetime

Define exactly how long reusable KV state exists.

Prefer narrow lifetime:

```text
within one request
```

for Phase 7 unless there is strong evidence for cross-request reuse.

Do not create a persistent global prompt cache by default.

---

# 47. Cross-Request Caching

Cross-request KV caching is NOT required in Phase 7.

It introduces:

```text
memory growth
cache invalidation
privacy concerns
complexity
```

Focus first on deterministic within-request reuse.

---

# 48. Concurrency Interaction

Phase 6 may serialize inference.

Ensure performance optimizations remain compatible with that concurrency model.

Do not mix:

```text
KV reuse work
```

with a major server concurrency redesign.

High concurrency remains outside the immediate optimization target.

---

# 49. Request Batching Across Clients

Do not implement dynamic batching across unrelated HTTP clients unless measurements show it is essential and the complexity is justified.

The roadmap target is primarily shared state within a Jev request.

Stay focused.

---

# 50. Cancellation / Shutdown Safety

Any new batching or KV state must remain compatible with Phase 6 graceful shutdown.

Do not introduce resources that can outlive:

```text
DecisionEngine
llama context
request scope
```

unexpectedly.

Use RAII ownership.

---

# 51. Error Behavior Preservation

Optimized execution must preserve structured error semantics.

Examples:

```text
invalid request
context overflow
inference failure
timeout
```

must still produce the same API-level error behavior.

Do not bypass validation through a fast path.

---

# 52. Context Overflow Handling

When batching multiple questions or reusing prefixes, carefully calculate context usage.

Do not accidentally allow cumulative token usage to exceed context limits.

Fail deterministically before unsafe evaluation.

---

# 53. Multilingual Performance

Measure at least basic English/Japanese performance cost.

Japanese tokenization may produce different prompt lengths.

Report:

```text
token count
latency
```

for representative English and Japanese requests.

Do not optimize one language at the expense of correctness in the other.

---

# 54. No New Model Search

Do not compare alternative models in Phase 7.

The Phase 5 model decision is frozen.

Optimization should target the selected default model.

Changing the model would make results incomparable.

---

# 55. No Calibration Retuning Unless Needed

Do not retune calibration merely because runtime code changed.

If candidate logits remain equivalent, existing calibration should remain valid.

If optimization changes logits beyond expected floating-point tolerance, investigate correctness first.

---

# 56. No Product Feature Work

Do not implement:

```text
CLI UX expansion
background daemon
model downloader
single-binary embedding
installer
authentication
```

during this phase.

Stay focused on runtime efficiency.

---

# 57. Regression Test Matrix

For every major optimization, verify:

```text
noul
choice
score
English
Japanese
single question
multi-question
```

where practical.

Not every test needs the full benchmark dataset.

Use representative fixtures plus full benchmark validation at milestones.

---

# 58. Numerical Regression Tests

Create tests comparing:

```text
reference path
optimized path
```

on the same input.

Compare:

```text
candidate logits
probabilities
prediction
score expected value
```

within documented tolerances.

---

# 59. JevBench Quality Regression

After selecting optimized defaults, rerun the established JevBench quality evaluation.

Compare against Phase 6 / Phase 5 reference metrics.

At minimum check:

```text
choice accuracy
noul accuracy
score MAE
```

and relevant calibration metrics.

Do not ship a faster default with an unexplained quality regression.

---

# 60. Server Integration Benchmark

Measure through the real server path as well as direct DecisionEngine benchmarks.

The final product latency includes:

```text
HTTP parse
validation
DecisionEngine
serialization
```

A direct inference microbenchmark alone is insufficient.

---

# 61. Direct Backend Benchmark

Also maintain a direct DecisionEngine/backend benchmark.

This distinguishes:

```text
inference cost
```

from:

```text
HTTP/server overhead
```

The two measurements serve different purposes.

---

# 62. Optimization Order

Use this order unless measurements strongly suggest otherwise:

```text
1. establish baseline
2. profile timing boundaries
3. prompt length
4. within-request KV reuse
5. multi-question batching
6. context sizing
7. thread count
8. llama.cpp runtime settings
9. select optimized defaults
10. rerun quality/regression suite
```

Do not change all variables simultaneously.

---

# 63. One Variable at a Time

When experimenting, prefer changing one major parameter at a time.

For example:

```text
threads = 4
threads = 8
threads = 12
```

while keeping:

```text
context
batch
prompt
model
```

fixed.

This makes results interpretable.

---

# 64. Small Parameter Sweeps

Small systematic sweeps are allowed for:

```text
thread count
context size
batch size
```

Do not build a huge auto-tuning framework.

A handful of sensible values is enough.

---

# 65. Auto-Tuning Is Not Required

Do not add startup-time benchmarking that automatically searches for optimal settings on every user's machine.

That belongs to later UX/runtime polish if ever needed.

Phase 7 should determine good defaults and expose useful configuration.

---

# 66. Default Selection

After experiments, select defaults based on evidence.

Document:

```text
why this thread count
why this context size
whether KV reuse is enabled
whether question batching is enabled
```

Do not leave tuned defaults unexplained.

---

# 67. Configuration Overrides

Allow expert overrides for important performance settings.

Conceptually:

```bash
pjev serve \
  --threads ... \
  --context-size ...
```

or existing config equivalents.

Do not expose every obscure llama.cpp internal knob directly.

Keep the supported surface small.

---

# 68. Stable Configuration Surface

Expose pjev-level concepts rather than raw llama.cpp internals where possible.

For example:

```text
threads
context_size
batch_size
```

are reasonable.

Avoid making users configure internal implementation details that may change across llama.cpp versions.

---

# 69. Performance Logging

Optionally add debug-level timing output.

For example:

```text
prompt_tokens
prompt_eval_ms
decision_ms
total_ms
```

Do not log this verbosely by default if it creates noise.

A benchmark command is preferable for detailed measurements.

---

# 70. Diagnostics

If a diagnostics command already exists or is easy to extend, expose effective runtime settings.

For example:

```bash
pjev diagnostics
```

could show:

```text
model
threads
context size
batch settings
KV reuse enabled/disabled
```

Do not build the entire Phase 11 diagnostics feature set here.

Only expose what materially helps performance verification.

---

# 71. Memory Safety

Pay special attention to llama.cpp ownership around:

```text
KV cache snapshots
batch objects
temporary token buffers
contexts
```

Use RAII where possible.

Performance work must not introduce use-after-free or lifetime ambiguity.

---

# 72. Avoid Unnecessary Copies

Profile large copies in:

```text
prompt strings
token vectors
candidate arrays
KV state handling
```

If measured as significant, reduce them.

Do not rewrite code for hypothetical copy costs without measurement.

---

# 73. String Construction

If prompt construction becomes measurable, reduce unnecessary repeated allocations.

But treat this as secondary to llama.cpp prompt evaluation.

Do not micro-optimize string concatenation before confirming it matters.

---

# 74. Batch Memory Bound

If batching increases memory proportional to question count, enforce reasonable limits.

Do not allow a single request with many questions to create unbounded allocations.

Reuse Phase 6 request validation/limits where possible.

---

# 75. Performance Failure Handling

If an optimized path cannot safely handle a particular request, prefer:

```text
fallback to reference path
```

where correctness can be preserved.

Do not return incorrect results merely to stay on the fast path.

Document fallback behavior.

---

# 76. Feature Flag During Development

During implementation, keep major optimizations selectable.

For example:

```text
kv_reuse: true/false
question_batching: true/false
```

This allows direct A/B comparison.

After validation, optimized behavior may become the default.

Retain a reference/debug override where useful.

---

# 77. Benchmark Artifact Directory

Organize performance results consistently.

Conceptually:

```text
artifacts/
  performance/
    baseline/
    prompt-shortening/
    kv-reuse/
    batching/
    threads/
    final/
```

Do not rely only on terminal output.

Preserve machine-readable results.

---

# 78. Reproducible Benchmark Command

The final repository should contain a documented command or set of commands capable of reproducing the selected performance profile.

For example:

```bash
pjev benchmark performance \
  --dataset ... \
  --iterations 100 \
  --output artifacts/performance/final
```

Use the actual implemented syntax.

---

# 79. Phase 7 Success Criteria

The phase should produce meaningful evidence of reduced repeated prompt-evaluation cost.

There is no required fixed speedup target.

A valid result may be:

```text
KV reuse gives large multi-question gains
```

or:

```text
KV reuse complexity is not worthwhile with the current prompt layout
```

Both are acceptable if measured rigorously.

Do not force a positive result.

---

# 80. Definition of Done

Phase 7 is complete when:

```text
[ ] A reproducible Phase 6 performance baseline exists.

[ ] Timing is separated into meaningful runtime stages.

[ ] Prompt token count has been measured.

[ ] Prompt-length optimization has been evaluated.

[ ] Within-request KV reuse has been evaluated.

[ ] Same-state / multiple-question reuse has been evaluated.

[ ] Multi-question batching has been evaluated.

[ ] Context-size tradeoffs have been measured.

[ ] Thread-count scaling has been measured.

[ ] Relevant llama.cpp settings have been evaluated.

[ ] Cold and warm latency are reported separately.

[ ] Single-question and multi-question latency are reported separately.

[ ] Multi-question speedup vs independent evaluation is reported.

[ ] RSS impact is measured for major optimizations.

[ ] English/Japanese token/latency differences are measured where practical.

[ ] Optimized and reference paths have numerical equivalence tests.

[ ] KV state cannot leak across unrelated requests.

[ ] Phase 6 structured errors and lifecycle behavior remain intact.

[ ] JevBench quality regression checks pass.

[ ] Selected optimized defaults are documented.

[ ] Important performance settings are configurable.

[ ] Machine-readable performance artifacts are generated.

[ ] Final `pjev serve` performance is measured through the real HTTP path.
```

---

# 81. Final Implementation Report

When implementation is complete, provide the following.

## Baseline

Report:

```text
model
hardware
pjev revision
llama.cpp revision
thread count
context size
batch settings
```

and:

```text
model load latency
warm p50
warm p95
RSS
```

## Timing Breakdown

Report:

```text
tokenization
prompt evaluation
decision math
HTTP overhead
```

for representative workloads.

## Prompt Optimization

Report:

```text
token count before/after
latency before/after
quality before/after
```

## KV Reuse

Explain:

```text
what prefix is reused
how KV state is managed
cache lifetime
correctness validation
memory cost
speedup
```

## Multi-Question Optimization

Report performance for:

```text
1 question
2 questions
4 questions
8 questions
```

or the actual tested set.

Show:

```text
total latency
per-question latency
speedup vs independent inference
```

## Context Size

Show tested values and tradeoffs.

## Thread Count

Show tested values and measured scaling.

## llama.cpp Settings

List every relevant setting changed and why.

## Multilingual Performance

Report representative:

```text
English token count / latency
Japanese token count / latency
```

## Numerical Regression

Confirm:

```text
noul
choice
score
```

results remain equivalent to the reference implementation.

## JevBench Regression

Report current:

```text
choice accuracy
noul accuracy
score MAE
calibration metrics
```

and compare them with the pre-optimization baseline.

## Selected Defaults

State the final chosen:

```text
thread policy
context size
batch settings
KV reuse setting
question batching setting
```

and the evidence for each.

## Commands

Provide exact `pjev` commands used to reproduce benchmarks.

## Tests

Provide exact test commands and outcomes.

## Known Limitations

Document issues such as:

```text
KV reuse only works within one request
specific prompt layouts limit prefix sharing
batching increases memory
thread scaling plateaus
platform-specific performance variance
```

Do not hide negative findings.

---

# Final Instruction

Phase 7 is not about making every line of C++ faster.

It is about reducing the dominant cost in pjev:

> repeated prompt evaluation.

Prioritize:

```text
fewer evaluated tokens
reuse of shared work
efficient handling of multiple questions
sensible llama.cpp settings
measured CPU scaling
```

over:

```text
micro-optimizing decision math
complex cross-request caching
high-concurrency redesign
new product features
new model experiments
```

The most important experiment is:

```text
same state
+
multiple questions
```

Determine whether pjev can reuse shared prompt computation safely and meaningfully.

Preserve correctness first.

Measure every optimization.

Keep a reference path.

The final result should be a faster `pjev serve` whose performance improvements are reproducible, explainable, and verified not to damage the decision behavior established in earlier phases.
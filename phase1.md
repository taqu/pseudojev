# Phase 1 — Minimal Jev-Compatible Native C++ Server

## Objective

Implement Phase 1 of pseudojev as a **native C++ application using llama.cpp directly**.

The goal of this phase is:

> establish a stable experimental API surface compatible with Jev

The goal is **not** to improve model quality.

At the end of this phase, JevBench should be able to run the same test suite against either:

```text
Jev
```

or:

```text
pseudojev
```

by changing endpoint/configuration only.

Do not perform Phase 2 research or optimization work in this phase.

---

# 1. Core Architecture Decision

From Phase 1 onward, pseudojev should be implemented natively in C++.

Do not introduce:

```text
Go
cgo
CGo wrappers
language FFI
RPC to a separate inference process
Python inference bindings
```

for the core runtime.

Use llama.cpp directly from C++.

The intended architecture is:

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

All layers should live in one native process.

The purpose of this architecture is to keep the experimental implementation close to the eventual production/distribution implementation and avoid a cross-language boundary around inference.

---

# 2. Primary Deliverable

Implement:

```text
POST /api/v1/systemone/
```

as a minimal Jev-compatible HTTP endpoint.

The endpoint should:

1. accept the request format expected by the existing JevBench adapter;
2. translate that request into calls to the native `DecisionEngine`;
3. execute the existing Phase 0 constrained-logit decision mechanism;
4. serialize the result into the response format expected by JevBench.

Do not invent a new pseudojev-specific API if the existing Jev contract already defines the required behavior.

The existing JevBench adapter and tests should be treated as the primary compatibility specification.

---

# 3. Repository Inspection First

Before modifying code:

1. inspect the existing repository structure;
2. locate the Phase 0 decision implementation;
3. locate the llama.cpp integration;
4. locate the JevBench adapter;
5. identify the exact Jev request and response schema currently expected;
6. identify existing build/test commands;
7. identify existing Phase 0 tests and benchmarks.

Do not redesign working Phase 0 behavior unnecessarily.

Prefer extracting clean interfaces around existing code rather than rewriting inference from scratch.

---

# 4. Build System

Use **CMake** as the primary build system.

The expected developer workflow should be approximately:

```bash
cmake -S . -B build
cmake --build build
```

Tests should integrate with CMake/CTest where practical.

For example:

```bash
ctest --test-dir build --output-on-failure
```

Integrate llama.cpp as a native C/C++ dependency.

Prefer a reproducible repository-controlled integration over requiring users to install llama.cpp globally.

Suitable approaches include:

```text
git submodule
vendored dependency
CMake FetchContent with pinned revision
```

Use the approach that best fits the existing repository.

Do not depend on an unpinned moving llama.cpp revision.

---

# 5. Cross-Platform Constraint

The implementation should avoid unnecessary platform-specific assumptions.

Target architectural compatibility with:

```text
Linux x86_64
Linux arm64
macOS arm64
Windows x86_64
```

Full release packaging for all platforms is not required in Phase 1.

However, do not introduce design choices that unnecessarily prevent those targets from working later.

In particular:

```text
avoid POSIX-only APIs in core logic
avoid shelling out to external processes
avoid hard-coded filesystem layouts
avoid compiler-specific behavior where unnecessary
```

If OS-specific behavior is needed, isolate it behind a narrow boundary.

---

# 6. Suggested Code Structure

A structure similar to the following is recommended:

```text
src/
  main.cpp

  server/
    http_server.*
    jev_api.*

  decision/
    decision_engine.*
    prompt_strategy.*
    candidate_scheme.*

  inference/
    llama_backend.*

tests/
  ...
```

Exact filenames are not mandatory.

What matters is dependency direction.

The intended dependency flow is:

```text
HTTP
  ↓
Jev compatibility
  ↓
DecisionEngine
  ↓
PromptStrategy + LlamaBackend
```

Do not allow HTTP-specific types to leak into the decision engine.

Do not put prompt construction inside route handlers.

Do not put Jev response serialization inside llama.cpp integration code.

---

# 7. DecisionEngine

Create or refactor toward a reusable native C++ `DecisionEngine`.

Conceptually, it should expose functionality equivalent to:

```cpp
noul(...)
choice(...)
score(...)
```

The exact API is up to you.

It should operate on domain-level inputs rather than HTTP request objects.

For example, conceptually:

```cpp
DecisionResult decision_engine.choice(...);
```

rather than:

```cpp
HttpResponse decision_engine.choice(HttpRequest&);
```

This distinction is important because the same engine will later be reused by:

```text
HTTP server
CLI
background runtime
benchmark tooling
```

Do not implement those later interfaces now.

Only make the boundary reusable.

---

# 8. Preserve Phase 0 Decision Mechanics

Phase 0 already demonstrated a working constrained-logit mechanism.

Preserve existing behavior including:

```text
noul
choice
score

single-token candidate validation
candidate-logit extraction
restricted softmax
deterministic argmax
constrained one-token generation
```

Do not replace constrained candidate-logit decisions with ordinary free generation.

Do not change the mathematical decision behavior unless required to fix a correctness bug.

If a refactor changes any Phase 0 numerical output unexpectedly, investigate it.

---

# 9. Native llama.cpp Backend

Wrap direct llama.cpp usage behind a small `LlamaBackend` or equivalent abstraction.

That layer should own llama.cpp-specific concerns such as:

```text
backend initialization
model loading
context creation
tokenization
prompt evaluation
logit access
candidate-token lookup
resource cleanup
```

Use RAII for llama.cpp resources where appropriate.

Avoid raw ownership patterns that make model/context lifetime unclear.

Conceptually:

```text
application start
    ↓
load model once
    ↓
create/use context
    ↓
serve decisions
    ↓
clean shutdown
```

Do not reload the model for every HTTP request.

Do not add a daemon/background worker abstraction yet.

The Phase 1 server process itself can own the loaded model.

---

# 10. PromptStrategy Boundary

Prompt formulation must remain explicitly configurable.

Phase 0 showed that decision quality changes materially depending on prompt layout and candidate representation.

Do not hard-code one formulation throughout the codebase.

Introduce a small strategy/configuration abstraction capable of representing at minimum:

```text
layout:
  auto
  state-first
  state-last

scheme:
  natural
  letters
```

The exact implementation may use:

```text
enum classes
configuration structs
small strategy classes
```

Avoid building a large plugin framework.

The important property is that future experiments can change prompt formulation without modifying:

```text
HTTP routing
Jev compatibility code
llama.cpp backend code
```

`PromptStrategy` should be the clear owner of:

```text
state placement
question placement
option rendering
candidate label rendering
candidate-token mapping
final prompt construction
```

---

# 11. `auto` Behavior

`layout=auto` does not need sophisticated optimization in Phase 1.

It may resolve to the current Phase 0 default.

The purpose of `auto` at this stage is primarily to prevent callers from depending on a specific internal layout.

Likewise, preserve the current default candidate scheme unless there is a correctness reason to change it.

Document defaults clearly.

---

# 12. Fix Phase 0 Special-Token Technical Debt

This is a required Phase 1 correctness fix.

User-controlled text must not accidentally become model chat control tokens.

The code must distinguish between:

```text
trusted chat/template structure
```

and:

```text
untrusted user-provided text
```

For example, if a user supplies:

```text
<|im_end|>
```

inside:

```text
state
question
option text
```

it must be treated as ordinary content.

It must not terminate or modify the model's chat structure.

The implementation should separate tokenization of trusted control tokens from tokenization of user content.

Do not rely only on escaping at the HTTP layer.

This must be correct at the tokenizer/prompt-construction boundary.

Add regression tests.

At minimum test special-token-looking strings in:

```text
state
question
option text
```

Verify that they cannot alter the intended chat template structure.

---

# 13. HTTP Library

Use a small C++ HTTP implementation appropriate for an embedded local server.

Keep the HTTP layer deliberately simple.

Requirements:

```text
POST support
JSON request/response support
basic status code handling
cross-platform support
low dependency complexity
```

Do not introduce a large web framework unless the repository already uses one.

The HTTP framework should remain an implementation detail.

Do not couple `DecisionEngine` to it.

---

# 14. JSON Handling

Use one consistent JSON library.

If the repository already has a JSON dependency, reuse it.

Otherwise select a lightweight, well-maintained C++ JSON library.

Keep API parsing separate from domain objects.

Conceptually:

```text
JSON request
   ↓
Jev request parser
   ↓
domain request
   ↓
DecisionEngine
   ↓
domain result
   ↓
Jev serializer
   ↓
JSON response
```

Malformed requests must result in deterministic HTTP errors rather than process termination or undefined behavior.

---

# 15. Jev Compatibility Layer

Treat the existing JevBench/Jev integration as the source of truth.

Determine exactly which parts of the Jev contract are required by the benchmark.

Implement the minimum required compatibility surface.

Do not invent unsupported semantics.

Where behavior is unclear:

1. inspect repository code;
2. inspect existing tests;
3. inspect benchmark expectations;
4. implement observed behavior;
5. document unresolved differences.

The compatibility layer should translate between:

```text
Jev wire format
```

and:

```text
pseudojev domain types
```

It should not own inference behavior.

---

# 16. Configuration

Provide the minimum configuration required to run the server.

At minimum, configuration must allow specifying the model path.

Also expose prompt strategy options where useful for experiments.

Possible mechanisms include:

```text
command-line flags
environment variables
small config object
```

Keep configuration explicit and simple.

Do not build a full configuration framework.

An example invocation may look conceptually like:

```bash
pseudojev-server \
  --model /path/to/model.gguf \
  --port 8080 \
  --layout auto \
  --scheme natural
```

Exact names may differ based on repository conventions.

Phase 1 does not require the final future CLI design.

---

# 17. Model Lifecycle

The model should be loaded once when the server starts.

Avoid:

```text
load model
handle request
unload model
```

for every request.

Model/context lifetime should be obvious from ownership structure.

Use explicit error handling for:

```text
model file not found
model load failure
invalid context configuration
tokenization failure
candidate token invalidity
inference failure
```

Do not silently continue after fatal initialization errors.

---

# 18. Concurrency

High concurrency is explicitly not a Phase 1 goal.

Prefer correctness and deterministic behavior over sophisticated concurrent request execution.

A single inference execution path or serialized inference access is acceptable.

If the selected HTTP library is multithreaded by default, ensure llama.cpp context access is safe.

Do not assume a shared llama context can be mutated concurrently.

It is acceptable in Phase 1 to protect inference with a simple mutex if necessary.

Do not build a worker pool or context pool yet.

---

# 19. Error Handling

Keep errors simple but deterministic.

At minimum distinguish:

```text
malformed HTTP/JSON request
invalid Jev request
invalid decision primitive
invalid candidate configuration
model/inference failure
internal server error
```

Return appropriate HTTP status codes.

Do not expose raw stack traces or arbitrary internal implementation details in API responses.

Full production structured-error design belongs to Phase 6.

---

# 20. Testing

Add tests at clear architectural boundaries.

At minimum, cover:

```text
1. Jev-compatible request parsing
2. Jev-compatible response serialization

3. noul execution
4. choice execution
5. score execution

6. malformed request handling

7. layout selection
8. candidate scheme selection

9. state-first prompt construction
10. state-last prompt construction

11. special-token-looking text in user state
12. special-token-looking text in question text
13. special-token-looking text in option text

14. candidate single-token validation

15. HTTP → compatibility layer → DecisionEngine integration
```

Where practical, unit tests should not require loading the full model.

Use a narrow backend interface or test double where that simplifies testing.

However, retain at least one real-model integration/smoke test proving:

```text
HTTP
  ↓
Jev compatibility
  ↓
DecisionEngine
  ↓
llama.cpp
```

works end-to-end.

Do not replace existing Phase 0 real-model tests.

---

# 21. JevBench Integration

Update the benchmark configuration or adapter so it can target either implementation.

Desired structure:

```text
JevBench
   │
   ├── Jev endpoint
   │
   └── pseudojev endpoint
```

Do not create a separate pseudojev-specific benchmark suite if the same test suite can be reused.

Endpoint selection should be configurable.

The final Phase 1 acceptance test must demonstrate the same benchmark code running against pseudojev.

---

# 22. Keep Benchmarking in the Development Loop

JevBench is not a future standalone feature.

Treat it as part of the development infrastructure from this point onward.

The intended loop remains:

```text
implement
   ↓
JevBench
   ↓
analyze
   ↓
change formulation
   ↓
JevBench
```

Phase 1 establishes the API required to support that loop.

Do not optimize benchmark quality yet.

---

# 23. Explicit Non-Goals

Do not implement the following in Phase 1:

```text
final CLI
daemon
background worker
Unix socket runtime
Windows named-pipe runtime
single-file model embedding
automatic model download
authentication
TLS
production observability
high-concurrency inference
request batching
KV-cache reuse optimization
calibration
candidate-binding experiments
option permutation experiments
prior correction
systematic prompt search
alternative model evaluation
cross-platform release packaging
```

These belong to later phases.

Do not pull future roadmap work into Phase 1 merely because the new C++ architecture makes it possible.

---

# 24. Avoid Premature Generic Abstractions

Do not build:

```text
generic model provider framework
generic HTTP routing framework
generic plugin framework
generic dependency injection system
generic inference graph
```

Pseudojev currently has one inference backend:

```text
llama.cpp
```

and one model family under active evaluation.

Abstract only where there is a concrete architectural reason.

The important boundaries are:

```text
transport
Jev compatibility
decision logic
prompt formulation
llama.cpp integration
```

Keep everything else straightforward.

---

# 25. Documentation

Document enough for another developer to reproduce the Phase 1 workflow.

Include:

```text
required build tools
how llama.cpp is obtained/built
CMake configuration
how to build pseudojev
how to run tests
how to specify the GGUF model
how to start the server
how to call POST /api/v1/systemone/
how to select layout
how to select candidate scheme
how to run JevBench against pseudojev
```

Also document supported Phase 1 platforms based on what has actually been tested.

Do not claim untested platforms as verified.

---

# 26. Definition of Done

Phase 1 is complete only when all of the following are true:

```text
[ ] pseudojev builds as a native C++ application.

[ ] The primary build system is CMake.

[ ] llama.cpp is called directly from C++.

[ ] There is no Go/cgo/FFI inference boundary.

[ ] POST /api/v1/systemone/ is implemented.

[ ] The endpoint accepts the request forms required by the existing
    JevBench adapter.

[ ] noul executes through the existing constrained-logit mechanism.

[ ] choice executes through the existing constrained-logit mechanism.

[ ] score executes through the existing constrained-logit mechanism.

[ ] HTTP-specific code is separated from DecisionEngine.

[ ] Jev compatibility translation is separated from DecisionEngine.

[ ] llama.cpp-specific code is isolated behind a narrow native backend
    boundary.

[ ] PromptStrategy is independently configurable.

[ ] state-first and state-last can be selected without changing HTTP or
    inference code.

[ ] natural and letters candidate schemes can be selected without
    changing HTTP or inference code.

[ ] User-controlled text is not interpreted as trusted chat special
    tokens.

[ ] Special-token injection regression tests pass.

[ ] Model lifetime is managed safely using clear C++ ownership.

[ ] The model is loaded once for the server process rather than once per
    request.

[ ] Existing Phase 0 tests still pass.

[ ] New Phase 1 unit/integration tests pass.

[ ] At least one real-model HTTP end-to-end test succeeds.

[ ] JevBench can run the same test suite against pseudojev by switching
    endpoint/configuration.

[ ] Build and usage documentation is updated.
```

---

# 27. Final Acceptance Test

The most important acceptance criterion is:

1. build pseudojev;
2. start the native C++ server with the Bonsai GGUF model;
3. configure JevBench to target pseudojev;
4. run the existing test suite;
5. verify that the suite executes successfully through the pseudojev endpoint;
6. switch the endpoint back to Jev;
7. confirm that the same benchmark path remains usable.

Different decision-quality scores between Jev and pseudojev are expected.

Do not treat lower benchmark quality as a Phase 1 failure.

Phase 2 is responsible for decision-quality improvement.

---

# 28. Work Procedure

Follow this implementation sequence unless repository structure strongly suggests a better order:

```text
1. Inspect existing Phase 0 implementation and JevBench adapter.

2. Write down the exact request/response compatibility contract.

3. Establish CMake/native llama.cpp integration if not already cleanly
   available.

4. Refactor existing Phase 0 inference into a reusable DecisionEngine and
   LlamaBackend boundary without changing behavior.

5. Introduce PromptStrategy/configuration.

6. Fix user-content vs special-token tokenization.

7. Add unit tests for the refactored boundaries.

8. Implement the Jev compatibility layer.

9. Implement POST /api/v1/systemone/.

10. Add HTTP integration tests.

11. Run existing Phase 0 tests.

12. Run real-model smoke tests.

13. Run JevBench against pseudojev.

14. Update documentation.
```

Avoid unrelated cleanup while doing this.

If existing code must be changed substantially, keep commits or implementation steps logically separated so regressions can be identified.

---

# 29. Do Not Optimize Scores During Phase 1

Do not modify prompts merely because a new layout happens to improve JevBench results while implementing the server.

Preserve the existing default formulation unless correctness requires otherwise.

If benchmark results change after the refactor:

```text
investigate first
```

Do not immediately tune prompts around the new result.

Phase 1 should provide a reliable experimental platform.

Phase 2 will use that platform to investigate:

```text
candidate binding
position bias
prompt layout
candidate priors
score formulation
```

---

# 30. Final Implementation Report

When implementation is finished, provide a concise report containing:

## Architecture

Describe the implemented dependency structure, including:

```text
HTTP
Jev compatibility
DecisionEngine
PromptStrategy
LlamaBackend
llama.cpp
```

## Files Changed

List files added, removed, or significantly modified.

## Jev Compatibility

Document exactly which Jev request/response behaviors were implemented.

Mention any known incompatibilities.

## C++ / llama.cpp Integration

Explain:

```text
how llama.cpp is included
how model/context ownership works
how model loading works
how inference access is synchronized if necessary
```

## Prompt Strategy

Document supported values and defaults for:

```text
layout
candidate scheme
```

## Tokenization Fix

Explain exactly how trusted control tokens and user-provided content are now separated.

Include the regression tests added.

## Build

Provide the exact commands used, for example:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Use the actual commands required by the repository.

## Server

Provide the exact command used to launch pseudojev.

## JevBench

Provide the exact command/configuration used to run JevBench against pseudojev.

## Test Results

Report:

```text
Phase 0 tests
Phase 1 unit tests
Phase 1 integration tests
real-model smoke test
JevBench execution
```

Do not omit failures.

## Remaining Issues

List known technical debt or compatibility gaps that should be addressed later.

---

# Final Instruction

Prioritize:

```text
correctness
clean architectural boundaries
behavior preservation
experimental flexibility
cross-platform simplicity
```

over:

```text
feature count
benchmark score improvement
production hardening
premature performance optimization
```

Phase 1 should leave pseudojev with a small, understandable native C++ foundation that can be retained through later server, CLI, optimization, and distribution phases.

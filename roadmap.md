# pseudojev Roadmap Revision — Native C++ Implementation

## Architecture Decision

From Phase 1 onward, pseudojev will be implemented as a **native C++ application built directly on llama.cpp**.

The previous direction of using Go with a llama.cpp binding should be dropped.

Primary reason:

> pseudojev is ultimately intended to be a small, local, cross-platform, single-binary decision engine, and introducing a Go ↔ C/C++ FFI boundary creates unnecessary build, packaging, debugging, and platform-specific complexity.

Because llama.cpp is already C/C++, keeping the application layer native removes this boundary entirely.

The target architecture becomes:

```text
HTTP API
   ↓
Jev compatibility layer
   ↓
DecisionEngine
   ↓
PromptStrategy
   ↓
llama.cpp
   ↓
GGUF model
```

All of these layers should live in the same native C++ codebase.

---

# Phase 1 — Minimal Jev-Compatible C++ Server

## Goal

Make pseudojev callable in the same way as Jev.

The objective remains:

> stable experimental API surface

Decision quality is not the focus of this phase.

Implement:

```text
POST /api/v1/systemone/
```

as a minimal native C++ HTTP server.

The server should call `DecisionEngine` directly without any language-binding or IPC boundary.

Suggested logical structure:

```text
src/
  server/
    http_server.*
    jev_api.*
  decision/
    decision_engine.*
    prompt_strategy.*
    candidate_scheme.*
  inference/
    llama_backend.*
  main.*
```

Exact filenames are not mandatory.

The important requirement is separation of responsibilities.

---

## DecisionEngine

`DecisionEngine` should expose native C++ APIs conceptually equivalent to:

```cpp
noul(...)
choice(...)
score(...)
```

The HTTP layer must not contain decision logic.

The decision layer must not depend on HTTP request types.

The same `DecisionEngine` will later be reused by:

```text
HTTP server
CLI
background runtime
benchmark tooling
```

without duplicating inference behavior.

---

## llama.cpp Integration

Use llama.cpp directly from C++.

Avoid:

```text
cgo
CGo wrappers
language FFI
separate inference subprocesses
```

unless a later requirement clearly justifies them.

The application should own llama.cpp lifecycle directly:

```text
initialize backend
load model
create context
evaluate prompts
extract logits
apply constrained decision logic
destroy context/model
```

The existing Phase 0 constrained-logit behavior must be preserved.

---

## PromptStrategy

Keep prompt formulation explicitly configurable.

Do not freeze Phase 0 choices into the implementation.

At minimum support representation of:

```text
layout:
  auto
  state-first
  state-last

scheme:
  natural
  letters
```

`PromptStrategy` should control:

```text
state placement
question placement
option rendering
candidate labels
candidate token mapping
```

Future Phase 2 experiments must be possible without changing HTTP routing or llama.cpp integration.

---

## Tokenization Fix

Fix the Phase 0 technical debt before completing serverization.

Trusted chat-template tokens and user-controlled content must be handled separately.

For example, user input containing:

```text
<|im_end|>
```

must remain ordinary user text and must not be interpreted as a chat control token.

The implementation should clearly distinguish:

```text
trusted template/control content
```

from:

```text
untrusted user text
```

and tokenize them with the appropriate special-token handling.

Add regression tests.

---

## Build System

Use CMake as the primary build system.

Target a conventional workflow such as:

```bash
cmake -S . -B build
cmake --build build
```

llama.cpp should be integrated as a native dependency, preferably in a way that supports reproducible builds.

Avoid designing the build around system-installed llama.cpp.

The project should ultimately be capable of building llama.cpp together with pseudojev.

---

## Platform Targets

The architecture should avoid assumptions that prevent support for:

```text
Linux x86_64
Linux arm64
macOS arm64
macOS x86_64 where practical
Windows x86_64
```

Phase 1 does not require fully polished release artifacts for every platform.

However, platform-dependent choices should not be introduced unnecessarily.

Keep OS-specific code behind narrow abstractions.

---

# Phase 2 — Decision Quality Baseline

No fundamental change.

Continue using the native C++ `DecisionEngine` and `PromptStrategy`.

Evaluate:

```text
candidate binding
option-order sensitivity
prompt layout
noul prior correction
score formulation
```

The benchmark loop remains:

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

The advantage of the native architecture is that these experiments occur directly against the final inference implementation rather than through an FFI layer.

---

# Phase 3 — Calibration

No architectural change.

Implement calibration as another layer around logits/probabilities produced by `DecisionEngine`.

Start with:

```text
temperature scaling
```

and optionally maintain separate parameters for:

```text
noul
choice
score
```

Calibration logic should remain independent of llama.cpp.

---

# Phase 4 — Multilingual Validation

No change in goals.

Validate at minimum:

```text
English
Japanese
```

and measure language degradation relative to English.

---

# Phase 5 — Model Value Gate

No major change.

Evaluate:

```text
quality
calibration
latency
RSS
model size
multilingual quality
```

Because inference is already native, model comparisons can use the same llama.cpp backend directly.

Do not introduce model abstraction layers beyond what is needed to swap GGUF models and associated prompt/tokenization configuration.

---

# Phase 6 — Production Server

Promote the Phase 1 C++ server into a production-capable local server.

Add:

```text
graceful shutdown
request validation
timeouts
request size limits
health endpoint
version endpoint
structured errors
basic logging
```

Keep the HTTP implementation deliberately small.

The command should eventually become:

```bash
pseudojev serve
```

The server should still invoke the same in-process `DecisionEngine`.

---

# Phase 7 — Performance Optimization

Continue with:

```text
prompt length
KV reuse
batching questions
context sizing
thread count
llama.cpp settings
```

Native C++ integration makes this phase especially important because pseudojev can directly access llama.cpp contexts and KV-cache behavior without crossing an FFI boundary.

In particular, investigate reuse for:

```text
same state
+
multiple questions
```

within one Jev request.

Avoid premature optimization before benchmark evidence identifies the important paths.

---

# Phase 8 — Native Single-Binary Distribution

This phase changes substantially.

The old concept:

```text
Go
+
static llama.cpp
+
Bonsai GGUF
```

is replaced with:

```text
pseudojev C++
+
llama.cpp
+
model distribution strategy
```

There is no Go runtime or cgo boundary.

Target:

```text
download
chmod +x
run
```

where applicable.

First aim for a native executable with llama.cpp statically linked.

Example conceptual output:

```text
pseudojev
```

plus either:

```text
bonsai.gguf
```

as an adjacent model file,

or a later embedded/resource-packaging mechanism.

Do not require model embedding in the first version of this phase.

Treat these as separate problems:

```text
1. single native executable
2. model discovery/download/cache
3. optional model embedding
```

Do not complicate executable portability merely to achieve literal one-file distribution prematurely.

---

# Phase 9 — CLI

Implement the CLI directly in the same C++ executable.

For example:

```bash
pseudojev noul ...
pseudojev choice ...
pseudojev score ...
pseudojev serve
```

Architecture:

```text
CLI
  ↓
DecisionEngine
```

and:

```text
HTTP server
  ↓
DecisionEngine
```

Both interfaces must reuse exactly the same decision implementation.

No local HTTP round-trip should be required for normal CLI execution.

---

# Phase 10 — Background Runtime

Add an optional long-lived native process to keep the model loaded.

Conceptually:

```text
CLI
 ↓
local worker exists?
 ├─ yes → communicate with worker
 └─ no  → start worker
```

After an idle timeout:

```text
unload model
exit
```

Add:

```bash
pseudojev status
pseudojev stop
```

Use narrow platform-specific IPC implementations:

```text
Unix/macOS:
  Unix domain socket

Windows:
  named pipe
```

Keep platform IPC isolated from `DecisionEngine`.

---

# Phase 11 — UX / Distribution Polish

Finalize:

```text
automatic CPU detection
model discovery/cache
cross-platform packaging
versioning
friendly errors
benchmark command
diagnostics
```

Where useful, expose llama.cpp runtime information through diagnostics:

```text
CPU features
thread configuration
backend
context size
model metadata
```

The final product remains:

> one native local Jev-compatible decision engine

without requiring a Go runtime or a CGo-based integration layer.

---

# Revised Technical Principles

From Phase 1 onward:

```text
C++ is the application implementation language.
llama.cpp is used directly.
CMake is the primary build system.
DecisionEngine is independent of transport.
PromptStrategy is independent of inference execution.
HTTP and CLI are thin interfaces over DecisionEngine.
Platform-specific code stays behind narrow boundaries.
```

Avoid introducing:

```text
CGo
FFI wrappers
RPC between the server and inference engine
separate inference processes
language-specific model bindings
```

unless later measurements demonstrate a concrete need.

---

# Revised Critical Path

```text
Phase 1
Native C++ Jev-compatible API
        ↓
Phase 2
candidate binding / position bias / priors
        ↓
Phase 3
calibration
        ↓
Phase 5
Bonsai value gate
        ↓
Phase 6
production-quality C++ server
        ↓
Phase 7
native llama.cpp optimization
        ↓
Phase 8
cross-platform native distribution
```

The main architectural objective is to make the experimental implementation and the eventual distributed implementation the same codebase.

Phase 1 should therefore already establish the native C++ boundaries that later phases will retain.

# Phase 6 — Production Server

## Objective

Implement Phase 6 of **pjev**.

The purpose of this phase is to turn the existing experimental Jev-compatible HTTP server into a small, reliable production-quality local server.

The command name is:

```bash
pjev
```

The primary server command should be:

```bash
pjev serve
```

The server should remain deliberately simple.

This phase is about production hardening, not adding features, increasing benchmark quality, or optimizing inference performance.

---

# 1. Project Context

Assume previous phases have already established:

```text
Phase 1:
native C++ Jev-compatible server

Phase 2:
DecisionEngine and prompt/formulation infrastructure

Phase 3:
calibration

Phase 4:
multilingual validation

Phase 5:
default-model decision
```

Phase 6 must reuse the same architecture.

The intended runtime remains:

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
   ↓
selected GGUF model
```

Do not introduce a second server-side decision implementation.

---

# 2. Phase 6 Scope

Add production hardening around the existing server.

Required work:

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

Keep the server small.

Do not turn pjev into a general web framework.

---

# 3. Non-Goals

Do NOT implement as part of Phase 6:

```text
high-concurrency inference architecture
request batching
KV-cache reuse optimization
advanced scheduling
GPU optimization
daemon/background worker
automatic model download
single-binary model embedding
authentication
user accounts
remote multi-tenant service
metrics backend
distributed tracing
large observability stack
TLS termination
rate-limiting platform
```

Those are either later-phase work or outside the intended local-server scope.

---

# 4. Command Interface

The server must be started through:

```bash
pjev serve
```

Do not create a separate executable such as:

```text
pjev-server
pseudojev-server
server
```

All server functionality should live under the `pjev` command namespace.

A typical invocation may conceptually look like:

```bash
pjev serve \
  --model /path/to/model.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --config /path/to/pjev-config.json
```

Use the actual command structure already established by the repository.

---

# 5. Default Bind Address

Because pjev is primarily a local decision engine, prefer a safe local default.

Unless an existing project convention says otherwise, default to:

```text
127.0.0.1
```

rather than:

```text
0.0.0.0
```

Binding to non-loopback interfaces should require explicit configuration.

Do not silently expose the server to the network.

---

# 6. Existing Jev Endpoint

Preserve the existing Jev-compatible endpoint:

```text
POST /api/v1/systemone/
```

Do not change its semantics unless required to fix a correctness issue.

Existing JevBench compatibility must continue to work.

Production hardening must not break the research/benchmark workflow.

---

# 7. Health Endpoint

Add a lightweight health endpoint.

Recommended:

```text
GET /health
```

or follow existing repository routing conventions if already established.

The endpoint should indicate whether the process is capable of serving requests.

At minimum distinguish:

```text
server process running
model successfully loaded
DecisionEngine initialized
```

Do not run model inference for every health request.

Health checks must be cheap.

---

# 8. Health Semantics

A successful health response should mean:

> pjev is initialized and capable of accepting decision requests.

If model initialization fails, the process should normally fail startup rather than run permanently unhealthy.

Avoid a state where the server accepts connections but can never perform inference.

---

# 9. Version Endpoint

Add a version endpoint.

Recommended:

```text
GET /version
```

Return machine-readable version information.

Include where available:

```text
pjev version
git/build revision
selected model identifier
configuration/schema version
```

Do not expose unnecessary filesystem paths or sensitive local information.

---

# 10. Version CLI Consistency

The HTTP version output should be consistent with:

```bash
pjev --version
```

if that command exists or is introduced.

Do not maintain unrelated version strings in multiple source locations.

Use one build/version source of truth.

---

# 11. Request Validation

Validate all incoming Jev API requests before inference.

Reject invalid inputs before they reach `DecisionEngine`.

Validation should include where applicable:

```text
valid JSON
required fields
valid primitive
valid option counts
valid field types
string size constraints
candidate constraints
score constraints
unsupported values
```

Do not rely on downstream crashes or assertions for request validation.

---

# 12. Validation Layer Boundary

Keep request validation in the HTTP/Jev compatibility layer.

Do not contaminate core DecisionEngine APIs with HTTP-specific validation rules.

Domain-level invariants may still be enforced inside DecisionEngine.

Conceptually:

```text
HTTP JSON
   ↓
wire-format validation
   ↓
Jev request parsing
   ↓
domain validation
   ↓
DecisionEngine
```

---

# 13. Structured Errors

Introduce a consistent structured error response.

Conceptually:

```json
{
  "error": {
    "code": "invalid_request",
    "message": "..."
  }
}
```

The exact schema may follow existing project conventions.

Use stable machine-readable error codes.

Do not require clients to parse human-readable error strings.

---

# 14. Error Categories

At minimum distinguish:

```text
invalid_json
invalid_request
unsupported_primitive
request_too_large
timeout
model_error
inference_error
internal_error
```

Add more only where useful.

Avoid exposing raw C++ exception text directly to clients.

---

# 15. HTTP Status Codes

Use appropriate HTTP status codes.

Conceptually:

```text
400  malformed/invalid request
404  unknown endpoint
405  wrong HTTP method
408/appropriate timeout response where applicable
413  request too large
500  unexpected internal error
503  model/runtime unavailable
```

Use one consistent mapping.

Document it.

---

# 16. Exception Boundary

No uncaught exception should escape the request handling boundary and terminate the server process for an ordinary bad request.

Catch unexpected exceptions at an appropriate outer request boundary.

Log enough information for diagnosis.

Return a generic structured internal error to the client.

Do not expose:

```text
stack traces
filesystem internals
raw exception dumps
```

in normal HTTP responses.

---

# 17. Startup Failure Behavior

Fatal initialization errors should fail fast.

Examples:

```text
model file missing
model load failure
invalid configuration
invalid calibration artifact
incompatible model configuration
port bind failure
```

The process should exit non-zero with a clear error.

Do not start a partially functional server.

---

# 18. Graceful Shutdown

Implement graceful shutdown.

Handle standard termination signals where supported.

At minimum:

```text
SIGINT
SIGTERM
```

on Unix-like systems.

Use the appropriate Windows equivalent/event handling where necessary.

---

# 19. Shutdown Sequence

The intended shutdown sequence is:

```text
receive shutdown request
        ↓
stop accepting new requests
        ↓
allow active request to finish or expire
        ↓
stop HTTP server
        ↓
destroy DecisionEngine / llama.cpp context
        ↓
release model
        ↓
exit cleanly
```

Do not abruptly destroy llama.cpp resources while inference is running.

---

# 20. Shutdown Timeout

Graceful shutdown must not wait forever.

Use a bounded shutdown timeout.

If active work does not finish within the configured grace period, terminate cleanly according to the HTTP/runtime library capabilities.

Keep this mechanism simple.

---

# 21. Request Timeout

Add request/inference timeout support.

A malformed or pathological request must not occupy the server indefinitely.

Separate where practical:

```text
HTTP request/read timeout
inference execution timeout policy
```

Do not assume llama.cpp computation can always be forcibly interrupted safely.

If inference cancellation is not technically safe with the current backend, document the limitation and enforce timeouts at the safest available boundary.

Do not introduce unsafe thread termination.

---

# 22. Timeout Configuration

Expose sensible configuration for timeouts.

Conceptually:

```text
request_read_timeout
request_execution_timeout
shutdown_grace_period
```

Do not hard-code arbitrary values throughout the codebase.

Use one server configuration object.

---

# 23. Request Size Limits

Add a maximum HTTP request body size.

Reject oversized requests before parsing/inference.

Return:

```text
413 Payload Too Large
```

or equivalent structured behavior.

This protects against accidental or malicious huge prompts.

---

# 24. Input Field Limits

Where reasonable, validate semantic input limits in addition to HTTP body size.

Examples:

```text
maximum state length
maximum question length
maximum option count
maximum option length
```

Do not introduce unnecessarily strict limits without evidence.

Make limits configurable or centrally defined.

---

# 25. Context Limit Awareness

If an otherwise valid request exceeds the selected model/context capacity, return a deterministic structured error.

Do not rely on llama.cpp failure deep inside inference.

Where practical, estimate or detect tokenized prompt size before full evaluation.

Report the error clearly.

---

# 26. Basic Logging

Add basic structured or consistently formatted server logging.

At minimum log:

```text
startup
model load result
listening address
shutdown
request completion
request failure
unexpected internal errors
```

Avoid verbose per-token logging.

---

# 27. Request Logging

For each request, useful fields may include:

```text
request ID
method
path
primitive
status
latency
error code
```

Do not log full user state/question content by default.

pjev may receive sensitive local text.

Prefer metadata over payload content.

---

# 28. Request IDs

Introduce a simple request identifier if one does not already exist.

The ID should be included in logs.

Optionally include it in error responses.

Do not require a distributed tracing system.

A lightweight local request ID is sufficient.

---

# 29. Logging Levels

Support a minimal level concept:

```text
error
warn
info
debug
```

Default server operation should not require debug logging.

Do not introduce a heavyweight logging framework unless the repository already uses one.

---

# 30. Avoid Logging Secrets / Content

By default, do not log:

```text
full state
full question
full option contents
raw prompt
raw model logits
```

Normal production logs should record operational metadata only.

Debug modes may expose more detail only if explicitly requested and clearly documented.

---

# 31. Model Lifecycle

The selected model must be loaded once during server startup.

The lifecycle should remain:

```text
pjev serve starts
        ↓
load model
        ↓
initialize DecisionEngine
        ↓
start HTTP listener
        ↓
serve requests
        ↓
shutdown
        ↓
release model
```

Do not lazy-load the model on the first request unless there is a strong existing architectural reason.

Startup success should mean the model is ready.

---

# 32. Readiness vs Liveness

If useful, distinguish:

```text
liveness
readiness
```

but do not overengineer this.

For a small local process, one `/health` endpoint may be sufficient if it clearly means "ready to serve."

Only introduce separate endpoints if the server architecture genuinely benefits.

---

# 33. Concurrency Policy

High concurrency remains a non-goal.

Correctness comes first.

If `DecisionEngine` / llama.cpp context is not safe for concurrent inference, serialize inference explicitly.

A simple mutex or single execution lane is acceptable.

Do not create a context pool in Phase 6.

---

# 34. Concurrent HTTP Requests

The HTTP server may accept multiple connections, but inference access must follow the actual thread-safety guarantees of the current implementation.

Do not assume:

```text
shared llama_context
```

is safe for simultaneous mutation.

Document whether inference is:

```text
serialized
single-threaded
or otherwise protected
```

---

# 35. Queue Behavior

If requests are serialized, avoid an unbounded queue.

Where supported by the HTTP architecture, limit pending work or reject overload deterministically.

Do not build a sophisticated scheduler.

A small bounded policy is sufficient.

---

# 36. Overload Errors

If the server cannot accept more work safely, return a structured error.

For example:

```text
server_busy
```

with an appropriate HTTP status.

Do not allow memory usage to grow without bound due to queued requests.

---

# 37. Configuration Object

Centralize server configuration.

It should contain where relevant:

```text
host
port
model path
pjev decision config
calibration artifact
request body limit
timeouts
logging level
concurrency/queue limit
```

Avoid global mutable configuration.

---

# 38. Configuration Precedence

If configuration can come from multiple sources, define precedence clearly.

For example:

```text
defaults
  ↓
config file
  ↓
environment
  ↓
CLI arguments
```

Use the project’s existing conventions if they already exist.

Document the actual precedence.

---

# 39. Configuration Validation

Validate configuration before loading the model or starting the listener.

Examples:

```text
invalid port
invalid timeout
missing model
invalid calibration artifact
unsupported candidate configuration
invalid request limit
```

Fail startup clearly.

---

# 40. pjev Configuration Artifact

Continue using the selected versioned decision configuration produced by earlier phases.

Do not move tuned values back into source code.

The server should load or resolve:

```text
prompt configuration
candidate scheme
prior correction
calibration
language overrides
```

through the established configuration system.

---

# 41. HTTP Library Discipline

Keep the existing Phase 1 HTTP library unless there is a demonstrated correctness or cross-platform reason to replace it.

Do not rewrite the server around a new framework merely because Phase 6 is called "production."

A small dependency surface is a product advantage.

---

# 42. Cross-Platform Behavior

Maintain compatibility with the intended native targets:

```text
Linux x86_64
Linux arm64
macOS arm64
Windows x86_64
```

Phase 6 does not require final release packaging for all platforms.

However, graceful shutdown, networking, path handling, and logging should not unnecessarily assume POSIX behavior.

---

# 43. No cgo / FFI

The server remains native C++.

Do not introduce:

```text
Go
cgo
Python runtime
separate inference process
FFI layer
```

Keep llama.cpp in-process.

---

# 44. Health Test

Add an integration test verifying:

```text
start pjev server
        ↓
GET /health
        ↓
success
```

The response should indicate readiness.

The test should also confirm the endpoint is inexpensive and does not run inference.

---

# 45. Version Test

Add a test for:

```text
GET /version
```

Verify:

```text
valid response schema
pjev version present
```

Avoid brittle assertions on dynamic build metadata unless appropriate.

---

# 46. Invalid Request Tests

Add tests for at least:

```text
invalid JSON
missing required field
wrong field type
unsupported primitive
invalid option structure
oversized request
unsupported method
unknown endpoint
```

Verify both HTTP status and structured error code.

---

# 47. Inference Error Test

Use a controlled backend failure/test double to verify that an inference failure:

```text
does not terminate server
returns structured error
logs request failure
```

Do not require corrupting the actual model file to test this.

---

# 48. Shutdown Test

Add an integration test proving that:

```text
server starts
request succeeds
shutdown signal sent
server stops accepting requests
resources are released
process exits successfully
```

Keep platform-specific test differences isolated.

---

# 49. Active-Request Shutdown Test

Where practical, test graceful shutdown while one request is in progress.

Verify:

```text
no use-after-free
no llama.cpp resource destruction during active inference
bounded shutdown behavior
```

Use a test double if necessary to simulate a slow inference call.

---

# 50. Timeout Tests

Add tests for:

```text
request read timeout
configured execution timeout behavior
shutdown timeout
```

Only test behavior that is safely implementable.

If hard cancellation of active llama.cpp inference is unsupported, test the documented fallback semantics rather than unsafe forced cancellation.

---

# 51. Request Size Test

Test:

```text
body below limit → accepted
body above limit → rejected before inference
```

Verify:

```text
HTTP 413
structured error
```

or the project's chosen equivalent.

---

# 52. Logging Tests

Do not over-test exact log formatting.

Test important properties where practical:

```text
startup message exists
request failures are logged
request ID is present
user content is not logged by default
```

Avoid tests that break on harmless wording changes.

---

# 53. JevBench Regression

After hardening, rerun the existing JevBench integration.

The same benchmark adapter must continue to target:

```text
POST /api/v1/systemone/
```

without production-server changes breaking compatibility.

Phase 6 is not complete if the server is hardened but no longer compatible with the established research workflow.

---

# 54. Calibration Regression

Verify the server can still run with the selected calibration configuration.

Test both:

```text
calibration enabled
calibration disabled
```

where the existing runtime supports both.

Do not allow server refactoring to change numerical decision behavior unexpectedly.

---

# 55. Numerical Behavior Preservation

For a fixed model, prompt configuration, and calibration artifact:

```text
Phase 5 decision output
```

and:

```text
Phase 6 server decision output
```

should remain equivalent within expected floating-point tolerance.

Production hardening should not alter model semantics.

---

# 56. Performance Regression Guard

Measure basic latency before and after Phase 6.

Do not optimize it yet.

The goal is only to ensure that server hardening has not introduced a major accidental regression.

Record:

```text
warm request latency
health endpoint latency
```

Do not start Phase 7 optimization work.

---

# 57. Security Scope

Phase 6 should implement basic defensive server behavior:

```text
safe local bind default
request size limits
input validation
bounded timeouts
structured errors
no payload logging by default
```

Do not attempt to turn pjev into an internet-facing hardened SaaS server.

Authentication and broader remote-service security are outside the current roadmap.

---

# 58. Fuzz-Friendly Boundaries

Where practical, keep JSON parsing and request validation functions pure enough that they can later be fuzz tested.

Actual fuzzing infrastructure is optional.

Do not introduce it if it expands scope significantly.

---

# 59. Server State

Keep server lifecycle state explicit.

Conceptually:

```text
initializing
ready
shutting_down
stopped
```

Avoid scattered boolean flags.

This is especially useful for health responses and shutdown correctness.

Do not overengineer a complex state machine.

---

# 60. Exit Codes

Use meaningful process exit behavior.

At minimum distinguish:

```text
clean shutdown → 0
startup/configuration failure → non-zero
fatal runtime failure → non-zero
```

Do not return success after a failed model load or failed bind.

---

# 61. Signal Safety

Signal handlers must not directly perform complex llama.cpp destruction or logging if that is unsafe.

Prefer:

```text
signal/event
   ↓
set shutdown request
   ↓
normal control flow performs cleanup
```

Use platform-appropriate safe mechanisms.

---

# 62. Resource Ownership

Use explicit RAII ownership for:

```text
HTTP server
model
llama.cpp context
DecisionEngine
configuration
```

Shutdown order should be obvious from the code.

Avoid raw global pointers.

---

# 63. Server Startup Logging

On successful startup, log useful operational information such as:

```text
pjev version
bind address
port
model identifier
model loaded
decision config identifier
```

Do not dump the entire config if it may contain local paths or unnecessary details.

---

# 64. Startup Readiness

Do not announce:

```text
server ready
```

before the model and DecisionEngine have finished initialization.

The listener should ideally become externally ready only after inference initialization succeeds.

---

# 65. Documentation

Update documentation so a developer/user can:

```text
build pjev
start pjev serve
specify the model
specify the decision/calibration config
change host/port
query /health
query /version
call /api/v1/systemone/
stop the server cleanly
understand request limits/timeouts
interpret structured errors
```

Use actual implemented commands.

---

# 66. Example Documentation

Include working examples equivalent to:

```bash
pjev serve \
  --model ./models/model.gguf \
  --config ./config/pjev.json
```

and:

```bash
curl http://127.0.0.1:8080/health
```

and:

```bash
curl http://127.0.0.1:8080/version
```

and a valid Jev-compatible POST example.

Use actual routes and fields implemented by the project.

---

# 67. Production Server Definition of Done

Phase 6 is complete when:

```text
[ ] `pjev serve` starts the production server.

[ ] The existing Jev-compatible endpoint remains functional.

[ ] Default binding is safe for a local application.

[ ] Request JSON is validated before inference.

[ ] Invalid requests return structured errors.

[ ] Request body size is bounded.

[ ] Context/prompt overflow is handled deterministically.

[ ] HTTP/request timeouts are configured.

[ ] Graceful shutdown is implemented.

[ ] Shutdown is bounded.

[ ] Active inference is not destroyed unsafely during shutdown.

[ ] `/health` is implemented.

[ ] `/version` is implemented.

[ ] Model initialization happens before readiness.

[ ] Fatal startup failures exit non-zero.

[ ] Basic request/startup/shutdown logging exists.

[ ] Full user content is not logged by default.

[ ] Request IDs or equivalent correlation identifiers exist.

[ ] Inference thread safety is explicit.

[ ] Pending work cannot grow without bound.

[ ] llama.cpp model/context ownership is RAII-safe.

[ ] Phase 5 numerical behavior is preserved.

[ ] JevBench compatibility still passes.

[ ] Unit and integration tests pass.

[ ] Documentation contains working `pjev serve` instructions.
```

---

# 68. Required Implementation Sequence

Follow approximately this order:

```text
1. Inspect the current Phase 1 server and Phase 5 runtime configuration.

2. Freeze existing API behavior with regression tests.

3. Introduce a centralized ServerConfig.

4. Implement request validation and structured errors.

5. Add request/body limits.

6. Add timeout configuration.

7. Add explicit server lifecycle state.

8. Implement graceful shutdown.

9. Add /health.

10. Add /version.

11. Add basic logging and request IDs.

12. Add/verify concurrency protection around DecisionEngine.

13. Add integration tests.

14. Rerun JevBench compatibility.

15. Verify numerical output against the Phase 5 baseline.

16. Update documentation.
```

Avoid unrelated refactoring.

---

# 69. Final Implementation Report

When implementation is complete, provide the following.

## Architecture

Describe:

```text
HTTP server
Jev compatibility layer
DecisionEngine
llama.cpp ownership
server lifecycle
```

## Files Changed

List added and significantly modified files.

## Server Command

Provide the exact working command:

```bash
pjev serve ...
```

## Endpoints

Document:

```text
POST /api/v1/systemone/
GET /health
GET /version
```

and any other endpoint actually implemented.

## Validation

Document request validation rules and size limits.

## Error Schema

Show the exact structured error format and stable error codes.

## Timeouts

Document:

```text
read timeout
request/execution timeout policy
shutdown grace period
```

including any technical limitation around cancelling active llama.cpp inference.

## Graceful Shutdown

Explain:

```text
signals/events handled
new-request behavior during shutdown
active-request behavior
resource cleanup order
```

## Concurrency

Document whether inference is:

```text
serialized
mutex-protected
single-threaded
```

and why.

## Logging

Document:

```text
log levels
request IDs
default payload privacy behavior
```

## Tests

Provide exact commands and results for:

```text
unit tests
HTTP integration tests
shutdown tests
timeout tests
JevBench regression
real-model smoke tests
```

## Numerical Regression

Confirm whether Phase 5 decision outputs were preserved.

Report any intentional differences.

## Known Limitations

Include any remaining issues such as:

```text
active inference cannot be forcibly cancelled safely
limited concurrent throughput
platform-specific shutdown differences
HTTP library limitations
```

Do not hide them.

---

# Final Instruction

Phase 6 is not a server rewrite.

It is a hardening pass over the existing pjev server.

Prioritize:

```text
correct lifecycle
safe input handling
deterministic errors
bounded resource use
clean shutdown
operational visibility
behavior preservation
```

over:

```text
high concurrency
maximum throughput
complex observability
new product features
performance optimization
```

At the end of Phase 6, `pjev serve` should be a small, predictable, production-quality local server suitable for real use while retaining the same DecisionEngine, Jev compatibility, and model behavior validated in earlier phases.
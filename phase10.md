# Phase 10 — Background Runtime

## Objective

Implement Phase 10 of **pjev**.

The goal of this phase is to hide repeated model-loading cost for direct CLI usage by introducing an optional local background runtime.

The command name is:

```bash
pjev
```

The intended user experience is:

```text
CLI invocation
    ↓
local pjev worker available?
    ├─ yes → reuse loaded model
    └─ no  → start background worker, then use it
```

After an idle timeout:

```text
unload model
exit background process
```

The background runtime should remain strictly local to the machine.

---

# 1. Project Context

Assume previous phases have already established:

```text
Phase 1:
native C++ DecisionEngine

Phase 6:
production server lifecycle and structured errors

Phase 7:
optimized llama.cpp runtime

Phase 8:
native distribution

Phase 9:
direct CLI commands

  pjev noul
  pjev choice
  pjev score
```

Phase 10 should accelerate those existing CLI commands.

Do not replace or remove the direct in-process execution path.

---

# 2. Core Architecture

The intended architecture is:

```text
pjev CLI
   ↓
BackgroundRuntimeClient
   ↓
local IPC
   ↓
pjev background worker
   ↓
DecisionEngine
   ↓
llama.cpp
```

The worker should host the same `DecisionEngine` used by:

```text
pjev serve
```

and direct Phase 9 CLI execution.

Do not create another implementation of decision logic.

---

# 3. Direct Mode Must Remain Available

The Phase 9 path:

```text
CLI
 ↓
DecisionEngine
```

must remain usable.

Background runtime is an optimization, not a correctness dependency.

Support an explicit direct mode or equivalent fallback.

Conceptually:

```bash
pjev choice ... --direct
```

or:

```bash
pjev --no-worker choice ...
```

Use the CLI style that fits the current project.

This is required for:

```text
debugging
recovery
testing
environments where IPC is unavailable
```

---

# 4. Primary Commands

Add:

```bash
pjev status
pjev stop
```

The existing commands:

```bash
pjev noul ...
pjev choice ...
pjev score ...
```

should transparently benefit from the background runtime according to the selected default policy.

Do not create a separate user-facing executable for the worker.

---

# 5. Worker Process

The background worker should be another mode of the existing `pjev` executable.

For example, internally:

```text
pjev internal-worker
```

or a hidden/internal command.

Do not require a second distributed binary.

The worker command does not need to be prominently documented as a user command.

---

# 6. Worker Responsibility

The worker should own:

```text
model loading
DecisionEngine lifetime
llama.cpp context lifetime
selected runtime configuration
local IPC listener
idle timeout
clean shutdown
```

It should not own unrelated user-facing CLI formatting.

The client CLI remains responsible for:

```text
argument parsing
human output
JSON output
shell-facing exit behavior
```

---

# 7. Local IPC

Use platform-native local IPC.

Unix-like systems:

```text
Unix domain socket
```

Windows:

```text
named pipe
```

Do not use a TCP localhost server for this background runtime unless there is a compelling technical limitation.

The Phase 6 HTTP server remains a separate product interface.

---

# 8. IPC Boundary

Keep IPC behind a narrow abstraction.

Conceptually:

```cpp
class LocalTransport {
    connect(...)
    send(...)
    receive(...)
}
```

with platform-specific implementations such as:

```text
UnixSocketTransport
NamedPipeTransport
```

Do not spread Unix socket or named-pipe details through CLI/DecisionEngine code.

---

# 9. IPC Protocol

Define a small versioned local protocol.

The protocol should support at minimum:

```text
noul request
choice request
score request
ping/status
shutdown
```

Prefer a simple framed format.

For example:

```text
length-prefixed JSON
```

or another explicit message framing scheme.

Do not rely on:

```text
connection close = message boundary
```

without a clearly safe protocol design.

---

# 10. Protocol Version

Include a protocol version.

Conceptually:

```json
{
  "protocol_version": 1,
  "type": "choice",
  ...
}
```

This is important because the CLI and worker may accidentally come from different pjev versions.

Do not silently communicate across incompatible protocol versions.

---

# 11. Client/Worker Version Mismatch

If an existing worker is incompatible with the invoking CLI:

```text
detect mismatch
        ↓
stop or replace incompatible worker safely
        ↓
start compatible worker
```

where practical.

At minimum, fail clearly rather than parsing an incompatible protocol.

Do not send arbitrary requests to a worker with an unknown protocol version.

---

# 12. Worker Identity

The client must be able to determine whether the discovered local endpoint belongs to a valid pjev worker.

A handshake should return information such as:

```text
protocol version
pjev version
worker PID
model identity
config identity
```

Do not trust endpoint existence alone.

---

# 13. Worker Discovery

Define deterministic per-user endpoint locations.

For Unix-like systems, use an appropriate user-local runtime directory where available.

Possible concepts:

```text
$XDG_RUNTIME_DIR
user temp/runtime directory
```

For Windows, use a user-scoped named pipe.

Do not use one global system-wide endpoint shared by all users.

---

# 14. Multi-User Isolation

Worker endpoints must be isolated by OS user.

One user's pjev CLI must not accidentally connect to another user's worker.

Use:

```text
filesystem permissions
user-scoped runtime directories
named-pipe security
```

as appropriate.

Keep the implementation simple but safe.

---

# 15. Endpoint Naming

Use a deterministic endpoint name derived from pjev identity.

Conceptually:

```text
pjev-runtime-v1
```

If configuration/model-specific workers are supported, include a stable configuration identity.

Avoid random endpoint names that make discovery difficult.

---

# 16. One Worker vs Multiple Workers

Prefer one active worker per user/configuration identity.

Do not create an uncontrolled new worker for every invocation.

A worker identity may need to include:

```text
model
decision config
runtime settings
```

if those materially affect results.

---

# 17. Configuration Identity

The worker must know which model/configuration it has loaded.

Record a stable identity based on:

```text
model hash or resolved identity
decision configuration
calibration configuration
language configuration
important llama.cpp runtime settings
```

A CLI request must not silently use a worker loaded with incompatible settings.

---

# 18. Config Mismatch Behavior

If a CLI invocation requests configuration different from the active worker:

Preferred options are:

```text
start/replace worker with requested configuration
```

or:

```text
fall back to direct mode
```

depending on project policy.

Do not silently ignore explicit CLI configuration.

Document the chosen behavior.

---

# 19. Worker Startup

For a normal CLI decision:

```text
CLI parses request
        ↓
resolve effective model/config
        ↓
attempt worker connection
        ↓
worker available and compatible?
        ├─ yes → send request
        └─ no  → start worker
                  ↓
                 wait for readiness
                  ↓
                 send request
```

Startup coordination must be bounded.

Do not wait indefinitely for a worker.

---

# 20. Race-Free Startup

Handle two CLI invocations starting simultaneously.

Do not allow both to permanently create competing workers for the same endpoint.

Use a simple coordination mechanism appropriate for the platform.

Possible techniques include:

```text
exclusive endpoint creation
lock file
named mutex
atomic bind
```

Prefer OS-level exclusivity where possible.

---

# 21. Startup Winner / Loser Behavior

If two processes attempt worker startup:

```text
one becomes worker
other detects worker readiness
```

Both CLI invocations should complete correctly.

Do not treat normal startup contention as an unrecoverable error.

---

# 22. Worker Spawn

Spawn the background process using native process APIs or the project's existing cross-platform process helper.

Do not construct shell command strings.

Avoid shell-dependent behavior.

Pass arguments safely.

---

# 23. Worker Detachment

The worker must survive after the invoking CLI exits.

Use platform-appropriate detached/background process behavior.

However, do not overengineer full daemonization.

The worker is a local ephemeral helper process, not a system daemon.

---

# 24. No System Service Installation

Do not implement:

```text
systemd service
launchd service
Windows service
```

Phase 10 worker startup is demand-driven.

The worker should begin when needed and exit automatically when idle.

---

# 25. Readiness

The worker must not be considered ready until:

```text
model loaded
DecisionEngine initialized
IPC listener active
```

A client should never send a decision request to a half-initialized worker.

---

# 26. Readiness Protocol

Use an explicit readiness mechanism.

Examples:

```text
successful IPC handshake
ready message
endpoint appears only after initialization
```

Do not rely solely on arbitrary sleeps.

---

# 27. Startup Timeout

Worker startup must have a bounded timeout.

If model loading fails or startup hangs:

```text
CLI exits non-zero
```

with an actionable error.

Do not leave the client waiting forever.

---

# 28. Worker Startup Errors

Errors such as:

```text
model missing
invalid configuration
model load failure
IPC bind failure
incompatible artifact
```

must be returned or otherwise surfaced to the original CLI invocation.

Do not hide background startup errors.

---

# 29. Decision Request Flow

For a normal request:

```text
CLI request
   ↓
serialize domain request
   ↓
local IPC
   ↓
worker
   ↓
DecisionEngine
   ↓
serialize domain result
   ↓
CLI
   ↓
human/JSON formatter
```

Do not route worker requests through the Jev HTTP compatibility layer.

Use domain-level request/result semantics.

---

# 30. CLI / Worker / Server Equivalence

For the same:

```text
model
configuration
semantic input
```

these paths must produce equivalent results:

```text
direct CLI
background worker CLI
HTTP server
```

Compare:

```text
semantic prediction
probabilities
score expected value
```

within expected floating-point tolerance.

---

# 31. Worker Serialization Types

Prefer shared domain serialization types.

Do not serialize internal llama.cpp pointers, tokenizer state, or other implementation details.

The IPC protocol should contain only stable request/result data.

---

# 32. Error Serialization

Worker-side errors should be returned in a structured local error format.

Reuse existing pjev error categories where appropriate:

```text
invalid_request
model_error
inference_error
timeout
internal_error
```

The CLI should map them to concise stderr messages and non-zero exit codes.

---

# 33. Worker Crash Behavior

If the worker dies unexpectedly:

```text
client detects broken connection
```

The client may:

```text
retry once by spawning a fresh worker
```

where safe.

Do not retry indefinitely.

A single controlled recovery attempt is sufficient.

---

# 34. Request Retry Safety

Only retry requests when doing so cannot create harmful duplicate side effects.

Current pjev decision primitives are read-only inference operations, so retry is generally safe.

Keep this assumption explicit.

---

# 35. Idle Timeout

Implement worker idle timeout.

After no active requests for a configured interval:

```text
worker initiates shutdown
        ↓
release model/context
        ↓
remove IPC endpoint
        ↓
exit
```

This is a core Phase 10 requirement.

---

# 36. Idle Definition

Idle should mean:

```text
no active request
AND
no completed request within idle timeout
```

Do not start the timeout while an inference request is still running.

---

# 37. Idle Timeout Configuration

Provide a sensible default.

Make it configurable through existing pjev configuration mechanisms.

Conceptually:

```text
worker_idle_timeout
```

Do not scatter the duration through source code.

---

# 38. Disable/Adjust Idle Timeout

If useful, allow expert configuration for:

```text
shorter timeout
longer timeout
disabled automatic timeout
```

but keep the default behavior straightforward.

Do not expose unnecessary worker knobs.

---

# 39. Idle Shutdown Race

Handle the case:

```text
worker begins idle shutdown
while
new client attempts connection
```

deterministically.

Possible acceptable behavior:

```text
client fails handshake
then starts fresh worker
```

Do not require a complex restart cancellation state machine.

---

# 40. `pjev status`

Implement:

```bash
pjev status
```

It should report whether a compatible worker is running.

Useful information may include:

```text
running / not running
PID
pjev version
model identity
uptime
idle duration
effective config identity
```

Keep default output concise.

---

# 41. `pjev status --json`

Where the CLI already supports JSON conventions, provide machine-readable status output.

Example conceptually:

```json
{
  "running": true,
  "pid": 1234,
  "model": "...",
  "uptime_seconds": 42
}
```

Do not require scripts to parse prose.

---

# 42. Status Must Not Start Worker

Important:

```bash
pjev status
```

must never start a worker.

If no worker exists, report that state and exit successfully according to the chosen CLI semantics.

---

# 43. `pjev stop`

Implement:

```bash
pjev stop
```

This should request clean worker shutdown.

Do not kill by PID immediately under normal conditions.

Preferred flow:

```text
connect
 ↓
send shutdown request
 ↓
worker stops accepting work
 ↓
active request completes or expires
 ↓
release model
 ↓
remove endpoint
 ↓
exit
```

---

# 44. Stop When No Worker Exists

If no worker is running:

```bash
pjev stop
```

should respond cleanly.

Do not treat "nothing to stop" as a catastrophic error.

Choose and document whether exit code is 0 or a distinct non-zero informational status.

Prefer script-friendly behavior.

---

# 45. Forced Stop

A forced-kill option is not required.

If implemented for recovery, keep it explicit, such as:

```bash
pjev stop --force
```

and use platform-safe process validation before killing anything.

Do not kill arbitrary PIDs from stale files.

---

# 46. Stale Endpoint Cleanup

Handle stale runtime artifacts.

Examples:

```text
Unix socket exists but worker is gone
lock file remains
metadata file remains
```

The client should validate worker liveness before trusting artifacts.

Clean stale artifacts safely.

---

# 47. PID Reuse Safety

Never trust a stored PID alone.

OS process IDs can be reused.

Use:

```text
IPC handshake
worker identity
protocol/version validation
```

before assuming a process is the pjev worker.

---

# 48. Endpoint Cleanup

Normal worker shutdown must remove its Unix socket/runtime metadata.

Named pipes should close naturally according to platform semantics.

Do not leave persistent stale endpoints after clean exit.

---

# 49. Crash Cleanup

If a worker crashes, the next client should recover stale local artifacts.

Do not require the user to manually delete socket files in normal failure scenarios.

---

# 50. Worker Logging

Use existing pjev logging infrastructure.

Worker logs may include:

```text
startup
model load
worker ready
request failures
idle shutdown
explicit stop
unexpected fatal errors
```

Do not log full user request content by default.

---

# 51. Background Log Destination

Do not write worker logs to the invoking CLI's stdout.

Choose a consistent strategy.

Possible simple policies:

```text
stderr inherited only during startup
then quiet
```

or:

```text
user-local log file
```

Only introduce a persistent log file if useful and maintainable.

Avoid large uncontrolled logs.

---

# 52. Log Rotation

Do not build a full log rotation system in Phase 10.

If persistent worker logs are used, keep them bounded or minimal.

Prefer existing logging policy where possible.

---

# 53. Worker Resource Ownership

Use RAII for:

```text
IPC listener
model
llama context
DecisionEngine
runtime metadata
```

Shutdown order must be explicit.

Do not rely on process termination to clean up normal resources.

---

# 54. Worker Lifecycle State

Use a small explicit lifecycle model.

Conceptually:

```text
starting
ready
shutting_down
stopped
```

Avoid scattered flags.

This helps with:

```text
status
idle shutdown
explicit stop
startup coordination
```

---

# 55. Worker Concurrency

Follow the validated Phase 6/7 inference concurrency policy.

Do not redesign model concurrency in Phase 10.

If inference is serialized, the worker should serialize it.

Local IPC does not imply parallel llama.cpp contexts.

---

# 56. Multiple Clients

Support multiple CLI clients connecting over time.

Concurrent client connections may exist, but inference behavior must respect the existing thread-safety model.

Do not create one model instance per client.

---

# 57. Queue Bounds

Do not permit unbounded pending requests.

Reuse or adapt the Phase 6 bounded-work policy where possible.

If busy, return a structured:

```text
server_busy
```

or equivalent local error.

---

# 58. Client Timeout

The CLI client must not wait forever for:

```text
connect
handshake
worker startup
decision response
shutdown acknowledgement
```

Reuse or define bounded local IPC timeouts.

---

# 59. Inference Timeout

Do not introduce unsafe forced thread cancellation.

Use the Phase 6 timeout semantics.

If active llama.cpp inference cannot be forcibly cancelled safely, preserve that limitation.

---

# 60. Graceful Process Exit

The worker must respond correctly to:

```text
idle timeout
pjev stop
OS termination signal/event
```

and release model resources safely.

Reuse the Phase 6 graceful shutdown infrastructure where possible.

---

# 61. Unix Domain Socket Permissions

On Unix-like systems, ensure the socket cannot be accessed broadly by unrelated users.

Use:

```text
user-private runtime directory
appropriate permissions
```

Do not create a world-writable pjev control socket.

---

# 62. Named Pipe Security

On Windows, configure named-pipe access so that the current user's pjev CLI can connect without exposing the worker unnecessarily to other users.

Keep security descriptors as narrow and maintainable as practical.

---

# 63. No Network Listener

The worker must not open:

```text
TCP port
HTTP port
network interface
```

for local CLI acceleration.

Use only local IPC.

`pjev serve` remains the HTTP mode.

---

# 64. Worker and `pjev serve`

Do not automatically reuse the Phase 6 HTTP server process as the CLI worker.

Keep:

```text
pjev serve
```

and:

```text
background CLI worker
```

as distinct roles even if they share `DecisionEngine`.

This avoids tying CLI acceleration to HTTP configuration.

---

# 65. Server + Worker Coexistence

It should be possible for:

```text
pjev serve
```

and the local worker to exist simultaneously if users explicitly do so.

Do not share mutable llama contexts across processes.

Each process owns its own model instance.

Document the resulting memory cost.

---

# 66. Avoid Accidental Double Model Load

For ordinary CLI use, ensure one compatible background worker is reused.

Do not spawn a new model process on every command because discovery failed due to a minor race.

Add tests for repeated invocation.

---

# 67. Direct Mode Performance Baseline

Before enabling background runtime, measure Phase 9 direct invocation:

```text
process startup
model load
decision
total latency
```

This is the baseline.

---

# 68. Worker Performance Metrics

Measure:

```text
first invocation latency
subsequent invocation latency
IPC overhead
worker startup time
decision latency
```

Separate:

```text
cold worker start
warm worker reuse
```

---

# 69. Main Success Metric

The key performance metric is:

> warm CLI invocation latency after the worker is already running.

Compare:

```text
Phase 9 direct invocation
vs
Phase 10 worker reuse
```

Do not hide the first-call startup cost.

---

# 70. IPC Overhead

Measure local IPC overhead separately where practical.

It should be small relative to model inference.

Do not prematurely optimize protocol serialization unless measurements show it matters.

---

# 71. First Invocation

The first invocation may still need to:

```text
spawn worker
load model
```

This is expected.

Do not claim the background runtime eliminates cold startup.

It amortizes it across subsequent CLI calls.

---

# 72. Prewarming

Do not add automatic login/startup prewarming.

The worker should start on demand.

OS startup integration belongs outside this phase.

---

# 73. Manual Start Command

A dedicated public:

```text
pjev start
```

is optional.

It is not required if normal CLI commands automatically start the worker.

Do not add unnecessary commands.

---

# 74. Direct/Worker Policy

Choose and document a clear default.

Recommended behavior:

```text
normal CLI decision → use/start worker
explicit --direct   → run Phase 9 direct path
```

If project requirements prefer opt-in worker usage initially, that is acceptable during rollout.

But final Phase 10 behavior should be clear and consistent.

---

# 75. Environment Control

If useful, allow a simple environment override for CI/debugging, but do not create a large environment variable surface.

For example, an internal/test mechanism to disable worker reuse may be useful.

Prefer explicit CLI/config options for users.

---

# 76. JSON Output Preservation

Background runtime must not alter Phase 9 output contracts.

For:

```bash
pjev choice ... --json
```

stdout should remain the same logical JSON whether execution is:

```text
direct
or
worker-backed
```

Do not expose transport details in normal result output.

---

# 77. Exit Code Preservation

Worker-backed CLI commands must preserve Phase 9 exit semantics.

For example:

```text
noul=false
```

is still:

```text
exit 0
```

Transport failures or model failures should be non-zero.

---

# 78. CLI Error Translation

Users should see pjev-level errors, not low-level:

```text
ECONNREFUSED
broken pipe
ERROR_PIPE_BUSY
```

unless debug mode is enabled.

Translate local IPC failures into concise actionable messages.

---

# 79. Worker Recovery

If IPC fails because the worker disappeared:

```text
detect stale worker
clean artifacts
start new worker
retry once
```

This should be transparent where safe.

If recovery fails, surface the final error.

---

# 80. Protocol Framing Tests

Add tests for:

```text
partial reads
partial writes
multiple sequential messages
invalid frame length
truncated frame
invalid JSON/protocol payload
```

Do not assume local IPC always transfers an entire message in one read.

---

# 81. Message Size Limits

Bound IPC message sizes.

Reuse semantic/request limits from Phase 6.

Do not trust a local peer blindly.

This also protects against corrupted/stale clients.

---

# 82. Invalid Protocol Request

The worker should reject malformed local requests without crashing.

Return a structured protocol error where possible.

Do not allow malformed IPC to terminate the worker.

---

# 83. Worker Authentication Scope

A full authentication system is not required.

OS-user-scoped local IPC permissions are sufficient for Phase 10.

Do not build token/password authentication for the local worker.

---

# 84. Configuration Changes

If the user explicitly supplies:

```text
different --model
different --config
different performance settings
```

the worker behavior must be predictable.

Do not let a warm worker override the user's explicit request.

---

# 85. Model Switching

Phase 10 does not need live model hot-swapping inside one worker.

If configuration changes materially:

```text
stop incompatible worker
start new worker
```

is acceptable.

Keep lifecycle simple.

---

# 86. Worker Restart Cost

Measure configuration-switch restart cost, but do not optimize it aggressively.

The common case should be repeated use of one default configuration.

---

# 87. Model Hash / Identity Check

Use the existing Phase 8/5 model identity metadata.

Do not rely only on model filename when deciding worker compatibility.

Two different files may share a name.

---

# 88. Calibration/Prompt Identity

Worker compatibility must include the effective decision configuration.

A worker using:

```text
prompt A
temperature X
```

must not serve a CLI invocation explicitly requesting:

```text
prompt B
temperature Y
```

without restarting or otherwise honoring the requested configuration.

---

# 89. Performance Settings Identity

Decide which performance options are worker-defining.

Likely examples:

```text
threads
context size
batch settings
```

Changing them may require worker restart.

Document the rules.

---

# 90. Status Configuration Visibility

`pjev status` should expose enough effective configuration to explain why a CLI invocation may require worker restart.

Do not dump every internal field.

A concise config/model identity is enough.

---

# 91. Status Human Output

Example conceptual output:

```text
pjev worker: running
pid: 12345
model: bonsai-1.7b-q4
uptime: 03:12
idle: 8s
```

Keep it concise.

---

# 92. Stop Timeout

`pjev stop` must not wait forever.

Use a bounded shutdown wait.

If clean shutdown fails, return a clear error.

Forced termination remains optional.

---

# 93. Idle Timeout Testability

Make idle timeout configurable enough for tests.

Tests should use a very short timeout.

Do not make tests wait for production-length idle periods.

---

# 94. Cross-Platform Abstraction

Keep platform-specific code in a narrow area:

```text
process spawn
endpoint discovery
Unix socket / named pipe
user runtime directory
process identity where needed
```

The rest of the worker/client protocol should be shared C++.

---

# 95. Unix Tests

On Unix-like systems test:

```text
worker spawn
socket permissions
status
repeated CLI reuse
stop
idle exit
stale socket recovery
simultaneous startup race
```

---

# 96. Windows Tests

On Windows test:

```text
worker spawn
named-pipe connection
user isolation where practical
status
stop
idle exit
paths with spaces
simultaneous startup
```

Do not claim Windows support based only on compilation.

---

# 97. Distribution Test

Test Phase 10 from the Phase 8 packaged distribution.

The worker must not depend on:

```text
source tree
build directory
developer-only files
```

It must correctly spawn the packaged `pjev` executable.

---

# 98. Executable Self-Spawn

When spawning the worker, resolve the current executable reliably.

Do not assume:

```text
"pjev"
```

is present on PATH.

This is especially important for portable distributions.

Use the resolved current executable path.

---

# 99. Relocation Test

Move the Phase 8 package.

Then run:

```bash
pjev choice ...
pjev status
pjev stop
```

Worker self-spawn must still work after relocation.

---

# 100. Working Directory Independence

Worker startup must not depend on the caller's current working directory.

Model/config discovery should continue using established Phase 8 rules.

---

# 101. Read-Only Install Directory

The worker must not need to write:

```text
socket files
locks
logs
state
```

next to the executable.

Use per-user writable runtime locations.

The installation directory may be read-only.

---

# 102. Runtime Directory

Create a small abstraction for the per-user runtime directory.

It may store:

```text
Unix socket
lock metadata
minimal worker metadata
```

Do not store the model there.

---

# 103. Runtime Cleanup

Clean worker runtime artifacts on:

```text
normal shutdown
idle timeout
explicit stop
```

and recover stale artifacts after crashes.

Do not delete unrelated files from the runtime directory.

---

# 104. Privacy

Do not persist:

```text
state
question
option contents
model prompts
```

in worker metadata.

Local runtime artifacts should contain operational information only.

---

# 105. No Request Cache

Phase 10 is not a semantic response-cache project.

Do not cache prior decision results by input.

The worker's value is model persistence, not answer caching.

---

# 106. No Cross-Request KV Cache by Default

Do not automatically introduce persistent KV reuse between unrelated CLI commands.

Phase 7 optimized safe within-request reuse.

Persistent cross-request KV state adds complexity and privacy risk.

The background worker should primarily keep:

```text
model
context/runtime infrastructure
```

available.

---

# 107. Context Reset

Ensure each logically independent CLI decision starts with correct clean inference state according to the existing `DecisionEngine`.

Worker reuse must not leak conversation/prompt state across commands.

Add regression tests.

---

# 108. Memory Stability

Run repeated CLI commands through one worker.

Measure RSS over time.

The worker should not show unbounded memory growth.

A small stable cache/allocation plateau is acceptable.

Investigate leaks.

---

# 109. Long-Running Worker Test

Run a stress/smoke loop:

```text
many sequential noul/choice/score requests
```

against one worker.

Verify:

```text
correct outputs
stable RSS
no context contamination
no endpoint failure
```

---

# 110. Graceful Shutdown Under Load

Test:

```text
active request
+
pjev stop
```

The worker should follow the Phase 6 shutdown policy.

Do not destroy llama.cpp resources mid-inference.

---

# 111. Idle Shutdown After Load

Test:

```text
start worker
run request
wait idle timeout
verify worker exits
verify endpoint removed
```

Then run another decision and verify a new worker starts successfully.

---

# 112. Simultaneous Clients

Test multiple client processes against one worker.

Verify:

```text
no protocol corruption
bounded queue behavior
correct result routing
```

Do not assume messages remain ordered across separate connections.

---

# 113. Response Correlation

If the protocol allows multiple outstanding requests on one connection, include request IDs.

Alternatively, keep one request per connection.

Prefer the simpler design unless persistent client connections materially help.

---

# 114. Connection Model

A simple model is acceptable:

```text
one CLI command
  ↓
open IPC connection
  ↓
one request
  ↓
one response
  ↓
close
```

The model remains loaded in the worker, so persistent client connections are unnecessary for most benefit.

Prefer simplicity.

---

# 115. Status Connection

`pjev status` can use the same short-lived connection/handshake protocol.

Do not create separate inspection files as the only source of truth.

The live worker should be authoritative.

---

# 116. Benchmark Commands

Create a reproducible Phase 10 performance benchmark.

Measure at least:

```text
direct CLI cold
worker-backed cold
worker-backed warm
```

For:

```text
noul
choice
score
```

using representative inputs.

---

# 117. Main Performance Report

Report:

```text
Phase 9 direct total latency
Phase 10 first-call total latency
Phase 10 warm-call total latency
IPC overhead
```

This should make the value of the background runtime obvious.

---

# 118. No Fake Performance Win

Do not compare:

```text
Phase 9 cold
```

only against:

```text
Phase 10 internal inference
```

Compare end-to-end CLI latency.

Measure what the user experiences.

---

# 119. Startup Failure Cleanup

If worker startup fails halfway:

```text
clean temporary runtime artifacts
```

where possible.

Do not leave a lock/socket that blocks future attempts.

---

# 120. Worker Exit Codes

Worker process exit semantics should distinguish:

```text
normal idle/stop shutdown
startup failure
fatal runtime failure
```

These may primarily be useful for diagnostics/logging.

---

# 121. Internal Worker Command Safety

If the hidden worker command is user-invokable accidentally, it should fail clearly without required internal arguments/configuration.

Do not expose unsafe low-level modes.

---

# 122. Documentation

Document the user-visible behavior.

Users should understand:

```text
CLI commands may start a local background worker
the worker keeps the model loaded
it exits automatically after idle timeout
pjev status shows it
pjev stop shuts it down
--direct bypasses it
```

Do not bury this behavior.

---

# 123. Privacy Documentation

State that the worker:

```text
runs locally
uses local IPC
does not require network access
does not persist request content by default
```

Keep claims aligned with actual implementation.

---

# 124. Help Text

Update:

```bash
pjev --help
pjev status --help
pjev stop --help
```

and the direct decision help if a `--direct` option is added.

Keep help concise.

---

# 125. Phase 10 Exit Criteria

Phase 10 is complete when:

```text
[ ] pjev decision commands can reuse a background worker.

[ ] The worker uses the same DecisionEngine as direct CLI and server paths.

[ ] Direct CLI mode remains available.

[ ] The worker is spawned from the same pjev executable.

[ ] Unix domain sockets are used on supported Unix-like platforms.

[ ] Named pipes are used on Windows.

[ ] IPC endpoints are user-scoped.

[ ] A versioned IPC protocol exists.

[ ] Client/worker compatibility is checked.

[ ] Worker configuration/model identity is validated.

[ ] Simultaneous startup does not create uncontrolled duplicate workers.

[ ] Worker startup has a bounded readiness timeout.

[ ] Startup errors are surfaced to the invoking CLI.

[ ] noul/choice/score work through the worker.

[ ] Worker-backed and direct results are equivalent.

[ ] Worker-backed and HTTP results are equivalent.

[ ] pjev status is implemented.

[ ] pjev status does not start a worker.

[ ] pjev stop is implemented.

[ ] Idle timeout unloads the model and exits the worker.

[ ] Stale IPC artifacts recover automatically.

[ ] Worker crashes can be recovered by a later CLI invocation.

[ ] Worker requests have bounded sizes/timeouts.

[ ] The worker does not leak request state across invocations.

[ ] Repeated requests do not cause unbounded memory growth.

[ ] Graceful shutdown remains safe during active inference.

[ ] Worker logs do not expose full user content by default.

[ ] The packaged Phase 8 distribution can self-spawn the worker.

[ ] Working-directory changes do not break worker startup.

[ ] Read-only installation directories are supported.

[ ] English/Japanese CLI smoke tests pass through the worker.

[ ] Warm CLI latency is measured and materially avoids repeated model-load cost.

[ ] Documentation explains worker behavior, status, stop, idle timeout, and direct mode.
```

---

# 126. Required Implementation Sequence

Follow approximately this order:

```text
1. Freeze the Phase 9 direct CLI behavior with regression tests.

2. Define shared domain IPC request/result types.

3. Define protocol framing and protocol version.

4. Implement LocalTransport abstraction.

5. Implement Unix domain socket transport.

6. Implement Windows named-pipe transport.

7. Implement hidden/internal worker mode.

8. Implement worker handshake and identity.

9. Implement deterministic user-scoped endpoint discovery.

10. Implement client connection and compatibility checks.

11. Implement worker spawning and readiness wait.

12. Handle simultaneous startup safely.

13. Route noul/choice/score through the worker.

14. Keep explicit direct fallback.

15. Implement idle timeout.

16. Implement pjev status.

17. Implement pjev stop.

18. Implement stale endpoint/crash recovery.

19. Add lifecycle, race, and equivalence tests.

20. Test packaged self-spawn.

21. Benchmark cold/warm CLI latency.

22. Update documentation.
```

Avoid unrelated feature work.

---

# 127. Final Implementation Report

When implementation is complete, provide the following.

## Architecture

Describe:

```text
CLI
 ↓
BackgroundRuntimeClient
 ↓
local IPC
 ↓
worker
 ↓
DecisionEngine
```

and the direct fallback:

```text
CLI
 ↓
DecisionEngine
```

## IPC

Document:

```text
Unix transport
Windows transport
endpoint location/naming
protocol framing
protocol version
message size limits
```

## Worker Identity

Explain how compatibility is checked for:

```text
pjev version
protocol version
model
decision config
runtime settings
```

## Startup

Explain:

```text
discovery
spawn
race handling
readiness
startup timeout
failure cleanup
```

## Lifecycle

Document:

```text
start
ready
idle
explicit stop
OS shutdown
crash recovery
```

## Idle Timeout

Report the default and configuration mechanism.

## Commands

Provide exact implemented syntax for:

```bash
pjev choice ...
pjev noul ...
pjev score ...
pjev status
pjev stop
```

and direct-mode override.

## Status Output

Show actual human and JSON output if JSON status is supported.

## Equivalence

Report tests comparing:

```text
direct CLI
worker CLI
HTTP server
```

for:

```text
noul
choice
score
```

## Cross-Platform

Report status for:

```text
Linux
macOS
Windows
```

including which IPC implementation was actually tested.

## Race Tests

Report:

```text
simultaneous worker startup
multiple clients
idle-shutdown race
stale endpoint recovery
```

## Memory

Report worker RSS after:

```text
startup
many requests
idle period
```

and whether memory remains stable.

## Performance

Report:

```text
Phase 9 direct CLI latency
Phase 10 first invocation latency
Phase 10 warm invocation latency
IPC overhead
```

using end-to-end measurements.

## Distribution

Confirm self-spawn works from the packaged Phase 8 release outside the source tree.

## Tests

Provide exact commands and outcomes.

## Known Limitations

Document issues such as:

```text
first invocation still pays model-load cost
one worker is tied to one effective configuration
configuration changes restart the worker
inference remains serialized
no cross-request semantic/KV caching
```

Do not hide them.

---

# Final Instruction

Phase 10 is about amortizing model load across repeated CLI invocations.

Do not turn pjev into a permanent system daemon.

Do not require a running HTTP server.

Do not cache user answers.

Do not introduce persistent cross-request inference state.

Keep the model loaded only while it is useful.

The target behavior is:

```text
first pjev decision
    ↓
start worker + load model
    ↓
decision

next pjev decision
    ↓
reuse worker
    ↓
decision quickly

idle timeout
    ↓
unload model
    ↓
exit
```

Prioritize:

```text
correct lifecycle
safe local IPC
fast warm CLI usage
automatic recovery
configuration correctness
cross-platform behavior
```

over:

```text
complex daemon management
persistent services
cross-request semantic caching
new inference features
```

At the end of Phase 10, repeated `pjev noul`, `pjev choice`, and `pjev score` commands should feel like lightweight CLI operations while still using the same validated local DecisionEngine and automatically releasing model memory when pjev is no longer being used.
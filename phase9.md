# Phase 9 — CLI

## Objective

Implement Phase 9 of **pjev**.

The goal of this phase is to add a clean human-facing command-line interface on top of the existing native pjev runtime.

The command name is:

```bash
pjev
```

The CLI should expose the existing decision primitives directly:

```bash
pjev noul ...
pjev choice ...
pjev score ...
```

The CLI must reuse the same `DecisionEngine` already used by:

```text
pjev serve
```

Do not reimplement decision logic in CLI code.

---

# 1. Project Context

Assume previous phases have already established:

```text
Phase 1:
native C++ DecisionEngine and Jev-compatible server

Phase 2:
decision formulation

Phase 3:
calibration

Phase 4:
multilingual validation

Phase 5:
selected default model

Phase 6:
production server

Phase 7:
performance optimization

Phase 8:
native distribution
```

Phase 9 adds a user-facing CLI to this existing runtime.

The architectural goal is:

```text
CLI
 ↓
DecisionEngine
```

and separately:

```text
HTTP server
 ↓
DecisionEngine
```

The CLI and server must share the same domain logic.

---

# 2. Core Principle

Do not implement the CLI as:

```text
CLI
 ↓
HTTP request
 ↓
local pjev server
```

for normal direct execution.

The CLI should call `DecisionEngine` in-process.

This avoids:

```text
HTTP overhead
server dependency
duplicated serialization logic
requiring a running daemon
```

A later phase may add background-runtime reuse, but Phase 9 should work independently.

---

# 3. Primary Commands

Implement at minimum:

```bash
pjev noul
pjev choice
pjev score
```

Keep:

```bash
pjev serve
```

working as already implemented.

The top-level command should conceptually look like:

```text
pjev
  serve
  noul
  choice
  score
```

Do not create separate binaries for each primitive.

---

# 4. CLI Design Goal

The CLI should be:

```text
simple
scriptable
predictable
consistent
```

Prioritize machine-usable behavior over decorative terminal output.

Users should be able to call pjev from:

```text
shell scripts
CI jobs
automation
interactive terminals
```

without parsing unstable formatting.

---

# 5. DecisionEngine Reuse

All decision commands must flow through the same runtime path:

```text
CLI parser
    ↓
CLI request object
    ↓
DecisionEngine
    ↓
PromptStrategy / Calibration
    ↓
LlamaBackend
    ↓
llama.cpp
```

Do not:

```text
duplicate prompt construction
duplicate calibration
duplicate candidate mapping
duplicate score math
```

inside CLI handlers.

The CLI should be a thin adapter.

---

# 6. Model Lifecycle

For direct CLI invocation, the basic lifecycle is:

```text
start pjev
    ↓
resolve model/config
    ↓
load model
    ↓
run requested decision
    ↓
print result
    ↓
release resources
    ↓
exit
```

This is acceptable in Phase 9.

Do not implement background model persistence yet.

That belongs to Phase 10.

---

# 7. Preserve `pjev serve`

Do not make the new CLI architecture break the existing server command.

The same executable should support both:

```bash
pjev serve
```

and:

```bash
pjev choice ...
```

Command dispatch should happen before model-specific work where practical.

For example:

```bash
pjev --version
```

should not load the model.

---

# 8. Common Runtime Configuration

Reuse the configuration/path resolution established in Phase 8.

CLI commands should support the same model-selection rules as the server.

At minimum preserve:

```text
explicit model path
default packaged model
decision configuration
calibration configuration
language configuration
performance settings
```

Do not create separate CLI-only defaults that diverge from `pjev serve`.

---

# 9. Model Override

Support explicit model selection through the existing convention.

Conceptually:

```bash
pjev choice \
  --model /path/to/model.gguf \
  ...
```

Use the actual project flag syntax.

Do not force explicit `--model` when the Phase 8 packaged default can be resolved automatically.

---

# 10. Configuration Override

Where the runtime already supports an explicit decision config, expose it consistently.

Conceptually:

```bash
pjev choice \
  --config ./pjev.json \
  ...
```

Do not add a second configuration format specifically for CLI usage.

---

# 11. `noul` Command

Implement a direct command for the `noul` primitive.

Conceptually:

```bash
pjev noul \
  --state "..." \
  --question "..."
```

Adapt arguments to the actual semantic inputs required by the current `DecisionEngine`.

The command must use the existing calibrated/constrained-logit path.

Do not replace it with free-text generation.

---

# 12. `noul` Output

The default output should be concise.

For example, conceptually:

```text
true
```

or:

```text
false
```

depending on current pjev semantics.

Also support machine-readable output.

For example:

```bash
pjev noul ... --json
```

The exact flag may differ.

Do not require parsing human prose for automation.

---

# 13. `choice` Command

Implement a direct `choice` command.

It should accept:

```text
state
question
options
```

using a CLI syntax that is clear and scriptable.

Conceptually:

```bash
pjev choice \
  --state "..." \
  --question "Which action should be taken?" \
  --option "A" \
  --option "B" \
  --option "C"
```

or equivalent.

Repeatable option flags are preferable to ambiguous delimiter parsing unless repository conventions suggest otherwise.

---

# 14. Choice Semantic Output

Default human output should make the selected answer clear.

Possible output:

```text
2
```

or:

```text
B
```

or the selected option text.

Choose one primary representation and document it.

Machine-readable mode should expose enough information to distinguish:

```text
semantic option index
candidate representation
selected option text
probabilities
```

where available.

---

# 15. `score` Command

Implement a direct `score` command.

Conceptually:

```bash
pjev score \
  --state "..." \
  --question "..." \
  ...
```

Use the existing score formulation from prior phases.

Do not introduce a new score scale.

---

# 16. Score Output

Human output should clearly display the resulting score.

Machine-readable mode should preserve useful fields such as:

```text
expected score
candidate distribution
candidate values
```

where the runtime already provides them.

Do not recalculate score independently in the CLI.

---

# 17. JSON Output

Provide a stable JSON output mode for decision commands.

Conceptually:

```bash
pjev choice ... --json
```

The output should be suitable for scripts.

Possible schema:

```json
{
  "primitive": "choice",
  "result": {
    "index": 1,
    "text": "..."
  },
  "probabilities": [
    0.12,
    0.71,
    0.17
  ]
}
```

Use the actual domain types already implemented.

Do not invent data that `DecisionEngine` does not expose.

---

# 18. JSON Schema Stability

Keep CLI JSON output logically aligned with existing domain/result types.

Do not simply print the HTTP Jev response if that format is awkward for CLI use.

However, avoid unnecessary duplication.

The JSON format should be:

```text
stable
documented
versionable if needed
```

because users may script against it.

---

# 19. Human Output vs JSON Output

Keep human output and JSON output separate.

Default:

```text
minimal readable result
```

JSON mode:

```text
structured details
```

Do not print log messages or explanatory text to stdout when JSON mode is active.

---

# 20. stdout / stderr Discipline

Use:

```text
stdout
```

for command results.

Use:

```text
stderr
```

for:

```text
errors
warnings
operational diagnostics
```

This is important for shell pipelines.

For example:

```bash
result=$(pjev choice ... --json)
```

must capture valid JSON only.

---

# 21. Exit Codes

Use predictable exit codes.

At minimum:

```text
0 = successful decision
non-zero = failure
```

Where useful, distinguish:

```text
invalid CLI usage
configuration failure
model load failure
decision failure
```

Do not encode model decisions themselves as process error codes.

For example, `noul=false` is still a successful invocation and should return exit code 0.

---

# 22. Input from stdin

Support stdin where it materially improves scriptability.

For example, long state content may be supplied via:

```bash
cat state.txt | pjev choice ...
```

The exact interface may be:

```text
--state -
```

or another conventional mechanism.

Do not invent a complicated stdin protocol.

---

# 23. Avoid Ambiguous stdin Semantics

If stdin is supported, define clearly what it represents.

For example:

```text
--state -
```

means:

> read state from stdin.

Do not automatically guess whether stdin contains:

```text
state
JSON
question
```

without an explicit flag or mode.

---

# 24. Optional JSON Input

If useful for automation and easy to implement, support structured JSON input.

Conceptually:

```bash
pjev choice --input request.json
```

or:

```bash
cat request.json | pjev choice --input -
```

Only add this if it fits existing architecture cleanly.

Do not make JSON input a Phase 9 completion requirement unless needed by current workflows.

---

# 25. Shell Quoting

CLI syntax must avoid requiring fragile quoting patterns.

Prefer:

```text
repeatable --option flags
file input
stdin
```

over large delimiter-encoded argument strings.

Document shell quoting examples carefully.

---

# 26. Unicode Support

Ensure CLI arguments correctly handle:

```text
English
Japanese
Unicode option text
Unicode file paths
```

Phase 4 established multilingual behavior; Phase 9 must not regress it.

Do not assume ASCII.

---

# 27. Japanese CLI Smoke Test

Add at least one CLI smoke test with Japanese:

```bash
pjev ...
```

using Japanese state/question text.

Verify:

```text
input reaches DecisionEngine correctly
output is valid UTF-8
```

---

# 28. File Input

Where useful, allow reading large state/question content from files.

Conceptually:

```bash
pjev choice \
  --state-file state.txt \
  --question "..." \
  ...
```

Keep the interface small.

Do not add file variants for every argument if stdin already solves the practical need.

---

# 29. Argument Validation

Validate CLI arguments before model loading where possible.

Examples:

```text
missing question
choice with fewer than required options
conflicting --state and --state-file
unknown command
invalid score configuration
```

Do not spend model-load time before reporting obvious CLI errors.

---

# 30. CLI Errors

CLI errors should be concise and actionable.

Example:

```text
error: choice requires at least 2 --option values
```

Avoid:

```text
stack traces
raw exceptions
internal type names
```

for ordinary usage mistakes.

---

# 31. Help

Provide:

```bash
pjev --help
pjev choice --help
pjev noul --help
pjev score --help
pjev serve --help
```

Help text should document:

```text
required inputs
common options
JSON output
model/config overrides
examples
```

Keep help concise.

---

# 32. Version

Preserve:

```bash
pjev --version
```

from the distribution/runtime work.

This command must not load the model.

---

# 33. Consistent Options

Common options should have consistent names across:

```text
serve
noul
choice
score
```

Examples may include:

```text
--model
--config
--threads
--context-size
```

Do not invent slightly different flags for each command.

---

# 34. Global vs Command-Specific Options

Organize arguments clearly.

Conceptually:

```text
global:
  --model
  --config
  --json where appropriate

command-specific:
  --state
  --question
  --option
```

Use the CLI library's cleanest idiom.

Do not make users care about internal parser architecture.

---

# 35. Logging

Direct decision commands should be quiet by default.

Do not print:

```text
model loading...
evaluating prompt...
done
```

to stdout.

If operational logs are enabled, send them to stderr.

A normal successful invocation should produce only its result.

---

# 36. Quiet Operation

The CLI should work naturally in shell pipelines.

For example:

```bash
pjev choice ... --json | jq .
```

must not be polluted by banners or progress messages.

---

# 37. Debug Mode

If existing logging infrastructure supports debug mode, expose it consistently.

For example:

```bash
pjev choice ... --log-level debug
```

Use existing configuration rather than adding CLI-specific debugging infrastructure.

---

# 38. Probability Output

Human default output does not need to include all probabilities.

Provide them in JSON or an explicit verbose mode.

For example:

```bash
pjev choice ... --json
```

should expose probabilities if they are part of the stable DecisionResult.

Do not hide calibrated output where users need it programmatically.

---

# 39. Raw Logits

Do not expose raw logits in the default CLI output.

If existing experiment/debug tooling already supports them, they may be available through:

```text
debug
diagnostic
experimental
```

output.

Raw logits are not required for normal user CLI operation.

---

# 40. Calibration Behavior

Direct CLI decisions should use the same selected calibration configuration as `pjev serve`.

Do not create:

```text
CLI calibration
server calibration
```

as separate systems.

For a fixed model/config/input, CLI and HTTP results should match.

---

# 41. CLI vs Server Equivalence

Add regression tests comparing:

```text
pjev choice ...
```

with:

```text
POST /api/v1/systemone/
```

for equivalent semantic inputs.

Verify:

```text
same semantic prediction
same probabilities within tolerance
same score expected value
```

where applicable.

---

# 42. DecisionEngine Direct Test

Also test the CLI adapter against a test-double DecisionEngine.

The CLI parser should be testable without loading the full GGUF model for every unit test.

Keep real-model CLI smoke tests separate.

---

# 43. Model Load Cost

Accept that direct CLI invocation currently pays model load cost.

Do not hide this by implementing a daemon now.

Document the behavior if users notice startup latency.

Phase 10 exists specifically to address persistent model loading.

---

# 44. Do Not Prematurely Implement Phase 10

Do not implement:

```text
background worker
Unix domain socket
Windows named pipe
automatic process spawning
idle timeout
pjev status
pjev stop
```

in Phase 9.

Keep the direct CLI path simple first.

---

# 45. Direct Execution Must Remain Valid Later

Even after a future background runtime exists, the direct DecisionEngine path should remain usable.

Do not architect Phase 9 so tightly around future IPC that direct operation becomes difficult.

---

# 46. No Server Requirement

The following must work with no server running:

```bash
pjev noul ...
pjev choice ...
pjev score ...
```

This is a core Phase 9 requirement.

---

# 47. No Network Access

Direct CLI decisions should not require network access.

They should use the local GGUF model directly.

Do not send decision requests to any remote endpoint.

---

# 48. Scriptability

Design output so that common scripts are simple.

Examples:

```bash
ANSWER=$(pjev choice ...)

pjev choice ... --json | jq '.result.index'
```

Do not require ANSI stripping or prose parsing.

---

# 49. TTY Detection

TTY-specific pretty output is optional.

Do not make it a requirement.

If implemented, ensure redirected output remains stable and plain.

Avoid adding formatting complexity that harms scriptability.

---

# 50. Color

Color output is optional.

If used:

```text
never emit ANSI color codes in JSON mode
disable when stdout is not a TTY
```

But do not spend significant effort on styling.

---

# 51. Batch Input

Do not implement a large batch-processing system unless trivial to reuse from existing tooling.

Phase 9 is primarily about single human/scripted decisions.

Experiment/dataset commands already cover large evaluation workloads.

---

# 52. Multiple Choice Options

Allow a practical number of options consistent with the existing DecisionEngine.

Do not impose arbitrary CLI-specific limits smaller than the engine/API limits.

Reuse domain validation.

---

# 53. Choice Option Ordering

Preserve user-specified option order exactly.

Do not sort or normalize options.

Phase 2 established that order sensitivity matters.

The CLI must not silently change semantics.

---

# 54. Prompt Strategy Override

If existing runtime configuration supports experimental prompt strategy overrides, they may remain available.

However, normal CLI usage should use the selected default configuration.

Do not expose research-oriented knobs prominently unless already part of the supported configuration surface.

---

# 55. Language Configuration

If Phase 4 introduced language-specific runtime configuration, direct CLI commands should resolve it in exactly the same way as the server.

Do not create automatic language detection unless already implemented and validated.

---

# 56. Performance Settings

Reuse the Phase 7 defaults.

Allow existing expert overrides such as:

```text
threads
context size
```

where supported.

Do not expose every llama.cpp knob directly through the CLI.

---

# 57. Distribution Compatibility

The CLI must work from the Phase 8 portable distribution.

Test:

```text
unpack pjev release
        ↓
pjev choice ...
```

outside the source tree.

Do not depend on benchmark files or development-only resources.

---

# 58. Model Discovery

Direct commands must use the same Phase 8 model discovery rules.

For example:

```text
explicit --model
        ↓
packaged/default model location
```

Do not create a CLI-specific model search path.

---

# 59. Working Directory Independence

Verify direct CLI commands work when called from a directory other than the executable location.

Example:

```bash
cd /tmp
/path/to/pjev choice ...
```

should still resolve packaged defaults according to Phase 8 rules.

---

# 60. Read-Only Install Directory

Direct CLI invocation should not require modifying the installation directory.

Do not write temporary state next to the executable.

---

# 61. Security / Privacy

Do not log user state/question content by default.

This applies equally to CLI operation.

Debug behavior should remain explicit.

The CLI should not create persistent history files.

---

# 62. Environment Variables

If the project already supports configuration via environment variables, preserve that behavior.

Do not introduce a large new environment-variable surface solely for Phase 9.

CLI flags should take precedence according to the existing configuration rules.

---

# 63. Command Parser

Use the existing CLI parsing infrastructure if one already exists for `pjev serve`.

Do not replace it without a concrete reason.

The parser should support:

```text
subcommands
repeatable options
help
validation
```

cleanly.

---

# 64. Avoid Manual argv Parsing Complexity

If the current project uses a suitable argument parsing library, reuse it.

Do not write a large custom parser unnecessarily.

If parsing is currently simple and custom, only extend it as needed.

---

# 65. Domain Request Types

Prefer constructing domain-level request types.

Conceptually:

```cpp
ChoiceRequest request;
request.state = ...;
request.question = ...;
request.options = ...;

auto result = engine.choice(request);
```

Do not construct fake HTTP requests from CLI arguments.

---

# 66. Shared Result Serialization

Where practical, reuse shared helpers for:

```text
probabilities
semantic option mapping
score values
```

between CLI JSON and other machine-readable artifacts.

Do not couple CLI JSON directly to HTTP-specific response classes.

---

# 67. CLI Result Types

If useful, create a small CLI presenter/formatter layer:

```text
DecisionResult
    ↓
human formatter
JSON formatter
```

Formatting must remain separate from decision execution.

---

# 68. Error Boundary

Catch unexpected exceptions at the CLI command boundary.

Return:

```text
concise stderr error
non-zero exit code
```

Do not crash with uncaught exceptions for ordinary failures.

---

# 69. Model Initialization Error

If the model cannot load:

```text
print actionable error
exit non-zero
```

Do not print a decision-shaped result.

Use the same underlying error classification as the server where practical.

---

# 70. Structured CLI Errors

JSON mode may optionally emit structured errors.

If implemented, ensure scripts can distinguish success and failure through:

```text
exit status
```

even when an error object is printed.

Do not return exit code 0 for operational failures.

---

# 71. Performance Expectations

Phase 9 should measure but not solve direct CLI startup cost.

Report approximately:

```text
model load
decision latency
total CLI latency
```

for a representative command.

Do not optimize model persistence here.

---

# 72. CLI Benchmark

Add a small benchmark/smoke measurement for:

```bash
pjev choice ...
```

through the real executable.

This helps establish the baseline that Phase 10 will later improve.

---

# 73. Do Not Regress `serve`

After CLI changes, rerun:

```bash
pjev serve
```

server smoke tests.

Command restructuring must not break:

```text
health
version
Jev API
graceful shutdown
```

---

# 74. Do Not Regress Distribution

Run CLI smoke tests using the packaged executable layout from Phase 8.

Do not test only from the development build tree.

---

# 75. Real-Model Smoke Tests

Add at least one real-model CLI smoke test for:

```text
noul
choice
score
```

These should validate the actual path:

```text
CLI
 ↓
DecisionEngine
 ↓
llama.cpp
```

---

# 76. English and Japanese Smoke Tests

Include:

```text
English input
Japanese input
```

in CLI smoke tests.

This protects Unicode argument handling and multilingual configuration.

---

# 77. Unit Tests

Add unit tests for:

```text
subcommand parsing
required argument validation
repeatable choice options
model/config option resolution
human result formatting
JSON result formatting
stdout/stderr separation
exit-code behavior
```

Use test doubles where possible.

---

# 78. Integration Tests

Add integration tests covering:

```text
CLI → DecisionEngine
CLI → real model
CLI/server equivalence
packaged-distribution CLI invocation
```

Not every test needs to run on every fast CI job if model size makes that impractical.

---

# 79. Invalid Usage Tests

Test cases such as:

```text
pjev choice
```

with missing required fields.

Also test:

```text
unknown subcommand
invalid option
conflicting input modes
zero choice options
```

Verify concise errors and non-zero exits.

---

# 80. JSON Cleanliness Test

Add a test confirming that:

```bash
pjev choice ... --json
```

writes only valid JSON to stdout.

Operational logs must not contaminate it.

---

# 81. Unicode File/Input Tests

Where practical test:

```text
Japanese text
Unicode option text
path containing Unicode characters
```

at least on platforms where CI supports it reliably.

---

# 82. Shell Pipeline Tests

Where practical, verify output can be consumed by another tool.

For example:

```text
capture stdout
parse JSON
```

No interactive behavior should be required.

---

# 83. Documentation

Update documentation with a user-facing CLI section.

Document:

```text
pjev noul
pjev choice
pjev score
model selection
config selection
JSON output
stdin/file input if implemented
exit behavior
```

Do not make users read server API documentation to understand the CLI.

---

# 84. Required CLI Examples

Provide real working examples equivalent to:

```bash
pjev noul \
  --state "..." \
  --question "..."
```

```bash
pjev choice \
  --state "..." \
  --question "..." \
  --option "..." \
  --option "..."
```

```bash
pjev score \
  --state "..." \
  --question "..."
```

and machine-readable examples:

```bash
pjev choice ... --json
```

Use actual implemented argument syntax.

---

# 85. Japanese Example

Include at least one documented Japanese example.

This confirms that the CLI is intended for multilingual use.

Do not provide only English examples.

---

# 86. Help Examples

Subcommand help should include at least one concise invocation example where the parser/library supports it cleanly.

Avoid huge help screens.

---

# 87. Phase 9 Exit Criteria

Phase 9 is complete when:

```text
[ ] `pjev noul` is implemented.

[ ] `pjev choice` is implemented.

[ ] `pjev score` is implemented.

[ ] `pjev serve` continues to work.

[ ] Decision commands invoke DecisionEngine directly in-process.

[ ] Normal CLI decisions do not require an HTTP server.

[ ] No decision logic is duplicated in CLI handlers.

[ ] CLI and server resolve the same model/config defaults.

[ ] Explicit model/config overrides work.

[ ] Human-readable output is concise.

[ ] Stable machine-readable JSON output is available.

[ ] stdout is reserved for result data.

[ ] errors/logging go to stderr.

[ ] successful false/negative decisions still exit 0.

[ ] operational failures exit non-zero.

[ ] option order is preserved exactly.

[ ] Unicode/Japanese input works.

[ ] English/Japanese real-model smoke tests pass.

[ ] CLI/server equivalent inputs produce equivalent decision outputs.

[ ] packaged Phase 8 distribution can run direct CLI decisions.

[ ] help and usage documentation exist.

[ ] direct CLI startup/model-load cost is measured.

[ ] no Phase 10 background-worker functionality has been introduced.
```

---

# 88. Required Implementation Sequence

Follow approximately this order:

```text
1. Inspect current top-level command parsing used by `pjev serve`.

2. Define shared global runtime/model configuration options.

3. Add domain-level CLI adapters for noul/choice/score.

4. Implement human result formatting.

5. Implement JSON result formatting.

6. Add argument validation before model load.

7. Add stdin/file input only where clearly useful.

8. Add unit tests with DecisionEngine test doubles.

9. Add real-model smoke tests.

10. Add CLI/server equivalence tests.

11. Test from the Phase 8 packaged distribution.

12. Add English/Japanese smoke tests.

13. Measure direct CLI startup/decision latency.

14. Update user documentation.
```

Avoid unrelated runtime refactoring.

---

# 89. Final Implementation Report

When implementation is complete, provide the following.

## Commands

Document the exact implemented syntax for:

```bash
pjev noul
pjev choice
pjev score
pjev serve
```

## Architecture

Explain:

```text
CLI parser
 ↓
CLI adapter
 ↓
DecisionEngine
```

and confirm that no local HTTP round-trip is used for direct decisions.

## Shared Runtime

Explain how:

```text
model discovery
configuration
calibration
language settings
performance settings
```

are shared between CLI and server.

## Human Output

Show actual output examples for:

```text
noul
choice
score
```

## JSON Output

Show the actual stable JSON schema/output for all three primitives.

## stdin / File Input

If implemented, document exact behavior.

## Exit Codes

Document success/failure semantics.

## CLI / Server Equivalence

Report regression-test results showing equivalent inputs produce equivalent decisions.

## Multilingual

Show working:

```text
English
Japanese
```

examples.

## Performance

Report representative:

```text
model load time
decision time
total direct CLI invocation time
```

Do not attempt to hide model load cost.

## Distribution Test

Confirm the CLI works from the Phase 8 packaged artifact outside the source tree.

## Tests

Provide exact test commands and outcomes.

## Known Limitations

Document items such as:

```text
every direct invocation currently loads the model
startup cost is visible
batch CLI input is limited or unsupported
background model reuse is not yet implemented
```

These are expected at Phase 9.

---

# Final Instruction

Phase 9 is about giving humans and scripts a clean interface to the existing pjev decision engine.

Do not build another decision implementation.

Do not route ordinary CLI commands through HTTP.

Do not hide model startup cost with background machinery yet.

Keep the architecture:

```text
pjev noul / choice / score
          ↓
    DecisionEngine
```

simple and direct.

Prioritize:

```text
correctness
scriptability
stable output
shared runtime behavior
Unicode support
clean errors
```

over:

```text
interactive UX tricks
background processes
fancy formatting
large batch systems
new decision features
```

At the end of Phase 9, a user should be able to unpack pjev and directly run local decisions with:

```bash
pjev noul ...
pjev choice ...
pjev score ...
```

without starting a server and without changing the decision behavior validated in earlier phases.
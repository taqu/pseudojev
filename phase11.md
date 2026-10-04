# Phase 11 — UX / Distribution Polish and GitHub Release Readiness

## Objective

Implement Phase 11 of **pjev**.

This is the final polish phase.

The goal is to turn the already validated, optimized, distributed pjev runtime into a clean open-source project that is easy to:

```text
discover
build
download
run
diagnose
benchmark
update
contribute to
```

The command name is:

```bash
pjev
```

The product concept is:

> a small, local, Jev-compatible decision engine

At the end of this phase, the repository should be ready for public distribution through GitHub.

That includes:

```text
README.md
GitHub Actions
release artifacts
cross-platform build validation
clear installation instructions
friendly runtime UX
diagnostics
versioning
```

Do not redesign the decision engine or repeat earlier research phases.

---

# 1. Project Context

Assume previous phases have already established:

```text
Phase 1:
native C++ runtime and Jev-compatible server

Phase 2:
decision formulation

Phase 3:
calibration

Phase 4:
multilingual validation

Phase 5:
default model selection

Phase 6:
production server

Phase 7:
performance optimization

Phase 8:
native distribution

Phase 9:
human-facing CLI

Phase 10:
background runtime
```

Phase 11 should polish and expose those capabilities cleanly.

Do not introduce a second architecture.

---

# 2. Main Phase 11 Areas

Complete the following areas:

```text
automatic CPU/runtime detection
model discovery / cache / extraction behavior where applicable
cross-platform packaging polish
versioning
friendly errors
benchmark command polish
diagnostics
README.md
GitHub Actions
GitHub release workflow
release verification
developer onboarding
```

Prioritize user experience and distribution reliability.

---

# 3. Keep the Final Product Simple

The final user-facing model should remain:

```text
download pjev
        ↓
provide or obtain compatible GGUF
        ↓
run
```

or, for packaged releases:

```text
download release archive
        ↓
unpack
        ↓
run pjev
```

Do not make normal users understand:

```text
llama.cpp internals
KV cache implementation
candidate-token details
calibration fitting
experiment infrastructure
```

unless they are reading development/research documentation.

---

# 4. Command Surface Review

Review the full command surface.

Expected major commands include:

```bash
pjev serve

pjev noul
pjev choice
pjev score

pjev status
pjev stop

pjev benchmark ...
pjev diagnostics

pjev --version
pjev --help
```

Only document commands that actually exist.

Remove or hide obsolete experimental commands from the primary UX where appropriate.

Do not break research tooling that is still intentionally supported.

---

# 5. Top-Level Help

Make:

```bash
pjev --help
```

a useful entry point.

It should explain:

```text
what pjev is
primary commands
how to get help for subcommands
where the model comes from
```

Keep it concise.

Do not print a wall of implementation detail.

---

# 6. Friendly Error Messages

Review common failure modes and replace low-level errors with actionable pjev errors.

Examples:

```text
model not found
model incompatible
configuration invalid
calibration incompatible
context too small
worker startup failed
worker version mismatch
port already in use
unsupported CPU/runtime condition
invalid choice options
```

The user should understand:

```text
what happened
what they can do next
```

Do not expose raw C++ or llama.cpp errors as the only message.

---

# 7. Preserve Debug Information

Friendly errors should not destroy diagnostic detail.

Use a model such as:

```text
normal mode:
  concise actionable error

debug/diagnostic mode:
  underlying technical detail
```

Do not print stack traces or internal state by default.

---

# 8. Exit Code Consistency

Review exit codes across:

```text
serve
decision CLI
status
stop
benchmark
diagnostics
```

Use consistent semantics.

At minimum:

```text
0 = successful command
non-zero = operational or usage failure
```

A semantic decision such as:

```text
noul = false
```

must not become a process failure.

---

# 9. Automatic CPU Detection

Implement or polish CPU capability detection where useful for runtime defaults.

Relevant information may include:

```text
architecture
logical core count
physical core count where reliably available
supported instruction capabilities relevant to the build
```

Use this to select sensible defaults only where validated.

Do not build a complex hardware tuner.

---

# 10. CPU Detection Scope

CPU detection should primarily help with:

```text
thread defaults
diagnostics
compatibility checks
```

Do not dynamically rewrite major inference configuration without evidence.

Keep user overrides available.

---

# 11. Portable Build Safety

Do not confuse runtime CPU detection with compiler architecture requirements.

Generic release binaries must still use the Phase 8-defined CPU baseline.

Runtime detection must never cause a binary to execute unsupported instructions that were compiled in unconditionally.

---

# 12. Diagnostics Command

Implement or complete:

```bash
pjev diagnostics
```

The command should help answer:

> Is my pjev installation healthy, and what runtime configuration is it using?

Useful output includes:

```text
pjev version
build revision
OS
architecture
CPU information
thread configuration
llama.cpp revision/backend
model path
model identity/hash where available
model size
context size
candidate/runtime config identity
calibration status
background worker status
```

Do not expose sensitive user input.

---

# 13. Diagnostics JSON

Where practical, support:

```bash
pjev diagnostics --json
```

for bug reports and automation.

Keep stdout valid JSON.

Do not mix logs into JSON output.

---

# 14. Diagnostics Privacy

Do not include:

```text
user prompts
state/question contents
full recent request history
```

in diagnostics.

For filesystem paths, consider whether full local paths are useful or unnecessarily revealing.

Where practical, provide a concise path or allow a redacted mode.

---

# 15. Installation Self-Check

Diagnostics should detect common installation problems:

```text
model missing
model hash mismatch
config missing/incompatible
worker stale
unsupported runtime configuration
unreadable files
```

Provide actionable remediation.

---

# 16. Benchmark Command

Polish the existing benchmark functionality into a useful end-user/developer command.

Conceptually:

```bash
pjev benchmark
```

or existing subcommands.

The command should make it easy to measure:

```text
model load time
warm decision latency
basic throughput where relevant
RSS if supported
prompt token count
```

Do not turn the user-facing benchmark command into the full internal research harness.

---

# 17. Benchmark Output

Default human output should be concise.

Example categories:

```text
model load
choice latency
noul latency
score latency
memory
```

Also provide JSON output if practical.

---

# 18. Benchmark Reproducibility

Report enough environment information to interpret results:

```text
CPU
threads
model
quantization
context size
pjev version
```

Do not present unlabeled performance numbers.

---

# 19. Model Discovery

Polish the Phase 8 model lookup behavior.

The lookup order should be explicit and documented.

Conceptually:

```text
explicit --model
        ↓
explicit config
        ↓
user model cache
        ↓
model adjacent to executable/package
```

Use the actual project design.

Do not search arbitrary filesystem locations.

---

# 20. Model Cache

If a user-level model cache is part of the final design, define one stable cross-platform location.

Examples conceptually:

```text
Linux:
  XDG cache/data location

macOS:
  user Library location

Windows:
  LocalAppData
```

Use standard platform conventions.

Do not write large model files into temporary directories.

---

# 21. Model Cache Scope

The model cache should be used for:

```text
managed/downloaded model assets
extracted packaged assets if absolutely necessary
```

Do not use it for:

```text
user prompts
decision history
unbounded logs
```

---

# 22. Model Integrity in Cache

Cached models should retain identity metadata.

Verify hash where a release configuration specifies one.

Do not silently use a corrupted or unexpected cached GGUF.

---

# 23. Model Extraction

Only retain model extraction logic if the actual Phase 8 distribution strategy requires it.

If releases use:

```text
pjev executable + adjacent GGUF
```

then no extraction layer is needed.

Do not add extraction complexity merely because it existed in the original roadmap concept.

---

# 24. Cross-Platform Paths

Review all user-visible paths for:

```text
spaces
Unicode
different working directory
read-only installation directory
```

Test them on supported platforms.

---

# 25. Background Runtime Polish

Review Phase 10 behavior.

Ensure normal users can understand:

```text
why a worker exists
when it starts
when it exits
how to inspect it
how to stop it
how to bypass it
```

Do not expose unnecessary IPC terminology in normal help.

---

# 26. Background Worker UX

Good user-level language is:

```text
background runtime
background worker
model stays loaded temporarily
```

Avoid requiring users to understand:

```text
Unix domain sockets
named pipes
protocol framing
```

unless troubleshooting.

---

# 27. Worker Recovery

Ensure stale-worker recovery is friendly.

If pjev repairs a stale local worker state automatically, avoid noisy warnings unless useful.

If recovery fails, diagnostics should explain what happened.

---

# 28. Versioning

Establish one authoritative pjev version source.

The same version should appear in:

```text
pjev --version
GET /version
release archive names
release manifest
GitHub release
README examples where version-specific
```

Do not maintain independent version strings.

---

# 29. Version Scheme

Use the project’s chosen versioning convention consistently.

If no convention exists, use a simple SemVer-compatible scheme:

```text
MAJOR.MINOR.PATCH
```

Do not invent complex prerelease semantics unless needed.

---

# 30. Build Metadata

Where useful, expose:

```text
version
git commit
build type
dirty state for developer builds
```

Official GitHub release artifacts should correspond to clean tagged source revisions.

---

# 31. Git Tags

Prepare the repository to build releases from Git tags.

Recommended convention:

```text
v0.1.0
v0.2.0
v1.0.0
```

Use the actual project convention if already established.

Do not hard-code assumptions in multiple scripts.

---

# 32. README.md Is a Required Deliverable

Create or substantially rewrite:

```text
README.md
```

for public GitHub users.

README.md must be treated as a product surface, not internal notes.

It should let a new user understand pjev without reading the roadmap.

---

# 33. README — Required Opening

The top of README should immediately answer:

```text
What is pjev?
Why would I use it?
What does it run on?
What model/runtime does it use?
```

Keep the opening concise.

Example conceptual positioning:

> pjev is a small local Jev-compatible decision engine powered by llama.cpp and a GGUF model.

Use the actual validated project claims.

Do not claim capabilities that have not been measured.

---

# 34. README — Status / Scope

Clearly communicate project maturity.

If it is experimental, say so.

If specific platforms are verified, list only those.

Do not claim:

```text
production-ready everywhere
fully multilingual
universal CPU support
```

unless validated.

---

# 35. README — Features

Include a concise feature section covering actual capabilities such as:

```text
local CPU inference
Jev-compatible HTTP API
noul / choice / score primitives
English/Japanese validation
CLI
background model reuse
cross-platform native binaries
```

Only include implemented features.

---

# 36. README — Quick Start

Provide the shortest successful path.

For a packaged release, conceptually:

```bash
./pjev --version
./pjev choice ...
```

or:

```bash
./pjev serve
```

Use actual command syntax.

Do not begin Quick Start with source compilation if release binaries exist.

---

# 37. README — Installation

Document:

```text
GitHub Releases installation
model placement
archive extraction
platform-specific notes
```

Then provide source-build instructions separately.

---

# 38. README — GitHub Releases

Explain how users obtain official binaries from GitHub Releases.

Document artifact naming.

Example:

```text
pjev-<version>-linux-x86_64.tar.gz
pjev-<version>-macos-arm64.tar.gz
pjev-<version>-windows-x86_64.zip
```

Use actual generated names.

---

# 39. README — Model Setup

Explain:

```text
which model is expected
where to place it
how to use --model
how integrity/version matching works
```

If the model cannot legally/practically be redistributed in the GitHub release, state clearly how the user obtains it.

Do not imply the GitHub artifact includes the model unless it actually does.

---

# 40. README — CLI Examples

Include real examples for:

```bash
pjev noul ...
pjev choice ...
pjev score ...
```

Include at least one Japanese example.

Do not use pseudo syntax that differs from the implementation.

---

# 41. README — Server Example

Document:

```bash
pjev serve
```

and:

```text
POST /api/v1/systemone/
GET /health
GET /version
```

Include one minimal working request.

Do not reproduce the entire API specification in README if a separate document exists.

---

# 42. README — Background Runtime

Briefly explain:

```text
CLI commands may reuse a temporary background worker
pjev status
pjev stop
direct-mode bypass if supported
idle shutdown
```

Keep this section user-oriented.

---

# 43. README — Build from Source

Document exact supported source-build commands.

At minimum:

```bash
cmake -S . -B build
cmake --build build --config Release
```

and test commands.

Include required:

```text
compiler
CMake version
platform dependencies
```

Use actual requirements.

---

# 44. README — Tests

Document a straightforward test command.

For example:

```bash
ctest --test-dir build --output-on-failure
```

Use the actual test workflow.

Do not require a model for unit tests unless truly necessary.

Explain real-model tests separately.

---

# 45. README — Performance

Include only measured representative numbers.

Every number must identify enough context to be meaningful:

```text
CPU
model/quantization
threads
workload
```

Do not copy historical Phase 0 values as current performance.

If stable public numbers are not yet appropriate, omit them.

---

# 46. README — Configuration

Explain the supported normal-user configuration surface.

Examples:

```text
model path
threads
context size
host/port
background worker behavior
```

Do not document every research-only internal knob in the main README.

---

# 47. README — Troubleshooting

Include common problems:

```text
model not found
worker appears stale
port in use
unsupported model/config
slow first CLI invocation
```

Link users to:

```bash
pjev diagnostics
```

where useful.

---

# 48. README — Project Architecture

Include a small architecture overview.

For example:

```text
CLI / HTTP
    ↓
DecisionEngine
    ↓
PromptStrategy + Calibration
    ↓
llama.cpp
    ↓
GGUF
```

Keep it brief.

This helps contributors understand why decision logic is not duplicated.

---

# 49. README — Development / Research Context

Optionally include a short section explaining that pjev was developed through:

```text
constrained-logit decisions
JevBench evaluation
calibration
multilingual validation
```

Do not turn the README into a research report.

Link to separate reports/docs if they exist.

---

# 50. README — License

Clearly state the repository license.

Also clarify model licensing separately if model assets have their own terms.

Do not conflate source license and model license.

---

# 51. README — Contributing

If the project is accepting contributions, add a concise contribution section or link to:

```text
CONTRIBUTING.md
```

Do not invent contribution policy if the repository is intentionally closed to contributions.

---

# 52. README — Badges

Add only useful badges.

Reasonable examples:

```text
CI
latest release
license
```

Avoid a large badge wall.

All badges must point to actual workflows/repository metadata.

---

# 53. README Verification

All README commands must be tested.

Do not publish commands that are aspirational.

As part of CI or release validation where practical, verify key examples remain valid.

---

# 54. Additional Documentation

Create separate docs where README would become too large.

Possible structure:

```text
docs/
  cli.md
  server.md
  configuration.md
  development.md
  benchmarking.md
```

Only create files that provide real value.

README should remain the primary entry point.

---

# 55. GitHub Actions Are a Required Deliverable

Create or polish workflows under:

```text
.github/workflows/
```

At minimum provide:

```text
CI
release build
```

Use separate workflows if that improves clarity.

---

# 56. CI Workflow

Create a normal CI workflow triggered by:

```text
push
pull_request
```

on relevant branches.

It should run fast enough for routine development.

---

# 57. CI Build Matrix

Run native builds on supported CI platforms where practical.

Recommended initial matrix:

```text
ubuntu
macos
windows
```

Architecture depends on available GitHub-hosted runners and project needs.

Do not claim an architecture is tested if CI does not actually run it.

---

# 58. CI Tasks

Normal CI should include:

```text
checkout
configure CMake
build
unit tests
lightweight integration tests
```

Avoid downloading a huge production GGUF model for every normal PR unless there is a strong reason.

---

# 59. CI Model-Free Tests

Structure tests so most CI can run without the real model.

Use:

```text
test doubles
fixtures
synthetic logits
small protocol tests
```

for:

```text
CLI parsing
HTTP validation
calibration math
IPC
artifact handling
configuration
```

Real-model validation belongs in a separate workflow if expensive.

---

# 60. Real-Model CI

If practical, add a manually triggered or scheduled workflow for real-model smoke tests.

Possible triggers:

```text
workflow_dispatch
schedule
release
```

Do not burden every pull request with large model downloads if it is slow or costly.

---

# 61. CI Caching

Use caching where safe for:

```text
CMake/dependency downloads
compiler caches if already supported
```

Do not cache generated research artifacts as if they were build dependencies.

Be conservative with model caching in public CI.

---

# 62. CI Dependency Pinning

Pin important third-party GitHub Actions to stable major versions or commits according to repository policy.

Avoid obviously floating/unreviewed dependencies.

Also keep llama.cpp revision pinned through the existing project mechanism.

---

# 63. CI Warnings

Build with appropriate warning levels.

Where the codebase is ready, consider treating project-code warnings as errors.

Do not turn upstream llama.cpp warnings into an unmanageable blocker.

Apply strictness selectively.

---

# 64. CI Build Types

At minimum test a standard release-like build.

Optionally include Debug on one platform if useful.

Do not explode the matrix without value.

---

# 65. Sanitizers

Where practical, add a Linux sanitizer job for project code:

```text
ASan
UBSan
```

if compatible with llama.cpp and existing tests.

Do not make sanitizers a Phase 11 blocker if upstream interaction makes them impractical.

Document skipped areas.

---

# 66. Formatting

If the project already uses a formatter, add a formatting check.

Typical:

```text
clang-format
```

Do not introduce a new style migration solely for Phase 11 unless needed.

---

# 67. Static Analysis

Optional lightweight static analysis may include:

```text
clang-tidy
```

only if the project already supports it cleanly.

Do not create hundreds of unrelated warnings and scope creep.

---

# 68. Release Workflow

Create a GitHub Actions release workflow triggered by version tags.

Conceptually:

```text
tag v*
    ↓
build release artifacts
    ↓
run smoke tests
    ↓
package
    ↓
generate checksums
    ↓
publish GitHub Release assets
```

Use the repository's version/tag convention.

---

# 69. Release Builds Must Use Phase 8 Packaging

Do not create a second packaging implementation inside GitHub Actions.

GitHub Actions should invoke the repository's existing release/package scripts or CMake targets.

There must be one packaging source of truth.

---

# 70. Release Matrix

Build official artifacts for each verified supported target available through GitHub Actions.

At minimum, where supported:

```text
Linux x86_64
macOS arm64 or available verified macOS target
Windows x86_64
```

Handle other architectures only where reliable builders exist.

Do not fake architecture labels.

---

# 71. Linux arm64

If Linux arm64 is a supported Phase 8 target but not conveniently available on standard GitHub runners, choose an explicit strategy:

```text
self-hosted runner
cross-build
separate/manual release path
not yet automated
```

Document actual status.

Do not pretend the standard CI tests it.

---

# 72. Release Smoke Tests

Before publishing an artifact, run the packaged binary outside the build tree.

At minimum:

```text
pjev --version
pjev diagnostics or basic self-check
```

and where model availability allows:

```text
noul
choice
score
```

plus server health/version tests.

---

# 73. Release Without Bundled Model

If GitHub release artifacts do not include the GGUF model, release smoke tests should still verify:

```text
binary starts
version works
diagnostics explains missing model cleanly
```

Real-model smoke tests may run separately before release.

Do not force model bundling purely for CI convenience.

---

# 74. GitHub Release Notes

Generate or prepare concise release notes.

Include:

```text
version
major user-visible changes
supported platforms
model requirement
known limitations
```

Do not dump raw commit logs as the only release notes.

Automatic generated notes may supplement curated text.

---

# 75. Release Assets

Attach:

```text
platform archives
checksums
release manifest where useful
```

Do not attach development build trees or temporary artifacts.

---

# 76. Checksums

Generate cryptographic checksums after final archive creation.

Prefer:

```text
SHA-256
```

unless project convention says otherwise.

Publish them with the GitHub Release.

---

# 77. GitHub Artifact Retention

CI artifacts may be used for debugging, but official downloadable binaries should come from GitHub Releases.

Do not make normal users search Actions artifacts.

---

# 78. Release Version Validation

The release workflow must verify:

```text
git tag
pjev --version
release manifest version
archive filename version
```

are consistent.

Fail the release if they disagree.

---

# 79. Dirty Builds

Official GitHub release workflow should build from the tagged clean checkout.

Do not publish artifacts carrying a dirty-development version marker.

---

# 80. GitHub Workflow Permissions

Use minimal GitHub Actions permissions.

Normal CI should not receive release-write permissions.

Release workflow should get only what it needs to create releases/assets.

Avoid broad write permissions.

---

# 81. Secrets

Do not require repository secrets unless actually necessary.

If code signing or notarization is added later, keep secrets isolated to release jobs.

Never echo secrets in logs.

---

# 82. macOS Signing

Do not make codesigning/notarization a hard requirement unless the project already has credentials and policy for it.

If official releases are unsigned, document that clearly.

Do not create fake signing steps.

---

# 83. Windows Signing

Same principle.

Do not implement placeholder certificate logic.

Unsigned artifacts are acceptable if documented.

---

# 84. GitHub Security Basics

Add or review:

```text
dependency pinning
workflow permissions
untrusted PR behavior
```

Do not execute untrusted fork code in privileged release jobs.

Keep release triggers limited to trusted refs/tags.

---

# 85. Pull Request Validation

PR CI should catch:

```text
build failures
unit test regressions
CLI regressions
IPC regressions
server regressions
packaging-script breakage where cheap
```

Do not make contributors manually run platform builds that CI can cover.

---

# 86. CMake Install / Package Target

Where useful, make release packaging available locally through a clear command.

For example:

```bash
cmake --build build --target package
```

or repository scripts.

GitHub Actions should use the same mechanism.

---

# 87. Developer Setup

Document contributor setup.

A new developer should be able to:

```text
clone
configure
build
run tests
```

without reverse-engineering CI.

README or `docs/development.md` should match GitHub Actions.

---

# 88. Dependency Setup in CI

Keep platform setup explicit.

Examples:

```text
compiler
CMake
Ninja if used
```

Do not rely accidentally on tools preinstalled on one runner image unless documented.

---

# 89. Build Generator

Choose a consistent CMake generator in CI where useful.

For example:

```text
Ninja
```

or native platform defaults.

Do not create needless differences between CI and documented local builds.

---

# 90. Release Reproducibility

Document exactly how a GitHub Release artifact is produced.

The same repository scripts should be usable locally where possible.

This helps debug failed releases.

---

# 91. Artifact Manifest

Every official release artifact should retain Phase 8 provenance information:

```text
pjev version
commit
platform
architecture
llama.cpp revision
model expectation/hash if applicable
```

Do not lose this metadata in GitHub packaging.

---

# 92. Model Distribution Documentation

GitHub README and Release notes must make model distribution explicit.

Possible cases:

## Case A — Model bundled

Document archive size and included model.

## Case B — Model separate

Document exactly how to obtain/place the model.

Do not leave users with a binary that fails without explanation.

---

# 93. License / Notices in GitHub Release

Ensure release packages include required:

```text
LICENSE
third-party notices
model license/notice where applicable
```

Do not rely only on the repository web page.

---

# 94. Repository Root Polish

Review root-level files.

At minimum consider:

```text
README.md
LICENSE
.gitignore
CMakeLists.txt
```

and, where useful:

```text
CONTRIBUTING.md
SECURITY.md
CHANGELOG.md
```

Do not create empty boilerplate files just to look complete.

---

# 95. `.gitignore`

Ensure large/generated assets are excluded appropriately:

```text
build directories
release output
benchmark artifacts
local GGUF models
temporary runtime state
```

Do not accidentally ignore source/config files needed by the project.

---

# 96. Model Files Must Not Be Accidentally Committed

Add appropriate ignore rules for local GGUF files unless the project intentionally versions one.

Be careful not to hide required small fixtures.

---

# 97. Release Assets Must Not Depend on Git LFS Accidentally

If model distribution uses GitHub Releases, do not accidentally require cloning Git LFS assets just to build the executable.

Keep source checkout manageable.

---

# 98. Changelog

If a changelog is useful, establish a minimal process.

Possible:

```text
CHANGELOG.md
```

or GitHub Releases as the authoritative history.

Do not maintain two detailed histories manually unless needed.

---

# 99. Security Documentation

If `SECURITY.md` is added, keep it factual.

Explain how to report vulnerabilities.

Do not claim unsupported response SLAs.

---

# 100. Contribution Documentation

If accepting contributions, document:

```text
build
test
format
platform expectations
```

Keep contribution guidance consistent with CI.

---

# 101. User-Facing Configuration Review

Review configuration names for clarity.

Avoid exposing internal experimental terminology where a stable product term exists.

Examples:

```text
threads
context-size
model
config
idle-timeout
```

are understandable.

Do not rename working options unnecessarily if it breaks compatibility.

---

# 102. Backward Compatibility

Avoid gratuitously breaking:

```text
CLI flags
JSON output
HTTP API
config schemas
```

during Phase 11 polish.

If a breaking change is necessary, version/document it.

This is the phase where users may begin depending on the public interface.

---

# 103. JSON Output Stability

Review JSON output for:

```text
CLI decisions
status
diagnostics
benchmark
```

Document which formats are intended as stable machine-readable interfaces.

Add schema version fields where justified.

---

# 104. Human Output Stability

Human output may remain less strict, but keep it concise and predictable.

Do not add decorative banners to commands intended for scripting.

---

# 105. Shell Completion

Shell completion is optional.

Do not delay Phase 11 completion for:

```text
bash completion
zsh completion
PowerShell completion
```

unless the CLI framework can generate them trivially.

---

# 106. Man Pages

Man pages are optional.

README/help quality is more important.

Do not create manual-page infrastructure unless it adds clear value.

---

# 107. Package Managers

Do not make package-manager distribution a Phase 11 requirement.

Examples:

```text
Homebrew
winget
apt
brew tap
```

may come later.

GitHub Releases are sufficient for this phase.

---

# 108. Automatic Updates

Do not implement automatic self-update.

Version detection may tell users what version they are running, but update logic is out of scope.

---

# 109. Telemetry

Do not add telemetry or analytics.

pjev should remain local by default.

No usage reporting is required for Phase 11.

---

# 110. Network Access

Normal:

```text
pjev noul
pjev choice
pjev score
background worker
```

must remain local.

GitHub distribution does not imply runtime network dependencies.

---

# 111. Release Documentation Verification

Before declaring Phase 11 complete, follow README Quick Start exactly from a clean release archive.

Do not use hidden developer knowledge.

Record any missing step and fix the documentation or distribution.

---

# 112. Clean-Machine Acceptance Test

The strongest acceptance test is:

```text
fresh supported environment
        ↓
download GitHub Release
        ↓
unpack
        ↓
follow README
        ↓
pjev works
```

No source tree or compiler should be required for binary users.

---

# 113. Source-Build Acceptance Test

Separately:

```text
fresh development environment
        ↓
clone repository
        ↓
follow README Build from Source
        ↓
build succeeds
        ↓
unit tests pass
```

README must support both user types.

---

# 114. GitHub Actions File Structure

A reasonable structure is:

```text
.github/
  workflows/
    ci.yml
    release.yml
```

Optionally:

```text
    real-model.yml
```

if expensive real-model tests need separation.

Keep workflow responsibilities obvious.

---

# 115. `ci.yml`

Normal CI should roughly perform:

```text
checkout
        ↓
install/setup build tools
        ↓
configure CMake
        ↓
build
        ↓
run unit tests
        ↓
run lightweight integration tests
```

Use a platform matrix where practical.

---

# 116. `release.yml`

Release workflow should roughly perform:

```text
tag push
        ↓
verify version/tag
        ↓
build release matrix
        ↓
package each artifact
        ↓
smoke test packaged artifact
        ↓
generate checksum
        ↓
upload artifacts
        ↓
create/update GitHub Release
```

Do not duplicate packaging logic.

---

# 117. Release Failure Policy

If one official supported platform fails to build or smoke test:

```text
fail the release
```

rather than silently publishing a partial release under the same support claim.

If partial releases are intentionally allowed, make that explicit.

---

# 118. Artifacts from Matrix Jobs

Use matrix jobs to produce platform-specific packages, then gather them into the release publication job.

Do not make one platform depend unnecessarily on artifacts from another.

---

# 119. Workflow Concurrency

Use workflow concurrency controls where useful to avoid duplicate release jobs for the same tag.

Do not overcomplicate normal CI.

---

# 120. GitHub Actions Timeouts

Add reasonable job timeouts, especially for:

```text
real-model tests
release builds
```

Do not allow broken workflows to hang indefinitely.

---

# 121. Release Model Handling

If the model is too large or unsuitable for automatic GitHub Actions download, separate:

```text
binary release workflow
```

from:

```text
real-model validation workflow
```

The release process may trust a previously validated model identity if that is the project policy.

Document it.

---

# 122. Artifact Size Constraints

Be mindful of GitHub Release and Actions artifact size practicalities.

Do not build a workflow that repeatedly uploads multi-GB intermediate files unnecessarily.

Prefer attaching the model only if that is an intentional distribution choice.

---

# 123. README Release Link

README should point users to the repository's GitHub Releases page using a stable relative or canonical project link.

Do not link to temporary Actions artifacts.

---

# 124. Repository Description

If appropriate, ensure README opening language can also serve as the GitHub repository description/tagline.

Keep branding consistent:

```text
pjev
```

Do not reintroduce historical project names.

---

# 125. Screenshots

Screenshots are optional and probably unnecessary.

pjev is primarily CLI/server software.

Prefer:

```text
short terminal examples
```

over decorative screenshots.

---

# 126. Examples Must Be Copy-Pasteable

README code blocks should be valid commands.

Avoid placeholders that look executable without clearly marking them.

Use variables/placeholders consistently.

---

# 127. Documentation of First-Run Cost

Explain that the first direct/background-backed decision may load the model and therefore be slower.

Explain that the background runtime makes subsequent CLI decisions faster.

Do not let users mistake first-run latency for steady-state behavior.

---

# 128. Documentation of Memory

Mention that keeping the background worker alive keeps model memory resident until idle timeout.

This is expected behavior.

Provide:

```bash
pjev stop
```

for users who want to release memory immediately.

---

# 129. Friendly Status UX

`pjev status` should be understandable without reading architecture docs.

Example concepts:

```text
Background runtime: running
Model: ...
Memory: ...
Idle for: ...
```

Only include memory if already measurable reliably.

---

# 130. Benchmark UX

A normal user should be able to run:

```bash
pjev benchmark
```

and receive a useful summary.

Research benchmark commands may remain more detailed under subcommands.

Do not expose JevBench internals as the only benchmark experience.

---

# 131. Diagnostics for GitHub Issues

Design:

```bash
pjev diagnostics --json
```

so users can attach its output to GitHub issues without exposing prompt content.

This is one of the most useful Phase 11 UX improvements.

---

# 132. GitHub Issue Templates

Optional.

If the repository will accept bug reports, consider a minimal bug-report template asking for:

```text
pjev version
platform
diagnostics output
reproduction steps
```

Do not require it for completion.

---

# 133. Known Limitations Section

README should contain or link to known limitations.

Examples might include:

```text
CPU-only current release
first-run model load latency
limited verified architectures
model file size
experimental multilingual scope
```

Only list real current limitations.

---

# 134. API Stability Statement

If the HTTP/CLI interfaces are not yet stable, say so.

Do not implicitly promise long-term compatibility if the project is still pre-1.0.

---

# 135. Release Candidate Testing

Before the first public release, create a release candidate artifact and test it like a user.

Possible tag:

```text
v0.x.y-rc.1
```

only if consistent with the project's versioning policy.

Not required if the project prefers direct releases.

---

# 136. Documentation Linting

Optional.

If README/docs contain many shell commands or links, add lightweight validation where practical.

Do not add a large documentation toolchain solely for this.

---

# 137. CMake Version Export

Ensure the build system can inject version metadata from one source.

Do not edit source files during CI merely to set the release version unless this is the project’s established method.

---

# 138. Generated Files

Avoid committing generated release artifacts.

Git should contain:

```text
source
build scripts
packaging scripts
workflow definitions
documentation
```

not built archives.

---

# 139. Release Script Idempotence

Running the packaging command twice from a clean source state should produce the same logical package layout.

Avoid hidden dependence on stale output directories.

---

# 140. Artifact Cleanliness

Before archive creation, verify packages do not contain:

```text
.git
build logs
CMake cache
object files
benchmark data
developer paths
temporary model files
```

Only ship intended user assets.

---

# 141. README Model License Clarity

If the selected model has separate licensing or redistribution restrictions, clearly distinguish:

```text
pjev source license
```

from:

```text
model license
```

Do not make legal conclusions beyond known metadata.

---

# 142. GitHub Release License Assets

If model redistribution is allowed and the model is bundled, include its required notice/license information.

If not bundled, document the upstream source and requirement without redistributing it.

---

# 143. Dependency Notices

Ensure bundled third-party dependencies have required notices.

At minimum review llama.cpp requirements.

Do not omit upstream licenses from distributed packages.

---

# 144. Phase 11 Exit Criteria

Phase 11 is complete when:

```text
[ ] pjev has polished top-level and subcommand help.

[ ] Common runtime errors are actionable and user-friendly.

[ ] `pjev diagnostics` is implemented or completed.

[ ] Diagnostics include runtime/build/model information without user content.

[ ] `pjev diagnostics --json` exists if practical.

[ ] CPU/runtime detection supports sensible defaults and diagnostics.

[ ] Model discovery behavior is stable and documented.

[ ] Model cache behavior is implemented only if actually needed.

[ ] Cross-platform path handling is verified.

[ ] `pjev benchmark` provides a useful user-facing performance summary.

[ ] Versioning has one authoritative source.

[ ] `pjev --version`, `/version`, manifests, and release names agree.

[ ] README.md has been written for public GitHub users.

[ ] README includes Quick Start.

[ ] README includes GitHub Release installation.

[ ] README explains model setup.

[ ] README documents noul/choice/score.

[ ] README documents `pjev serve`.

[ ] README explains the background runtime.

[ ] README includes source-build instructions.

[ ] README includes test instructions.

[ ] README includes troubleshooting and diagnostics.

[ ] README states verified platforms and known limitations accurately.

[ ] README includes license/model-license information.

[ ] README commands have been tested.

[ ] Normal GitHub Actions CI exists.

[ ] CI runs on a useful platform matrix.

[ ] CI builds and runs model-free tests.

[ ] Expensive real-model testing is separated if needed.

[ ] GitHub Actions release workflow exists.

[ ] Tagged releases build official platform artifacts.

[ ] Release workflow uses the Phase 8 packaging implementation.

[ ] Packaged artifacts are smoke-tested before publishing.

[ ] Release checksums are generated.

[ ] GitHub Release assets use stable version/platform names.

[ ] Workflow permissions are minimal.

[ ] Release version/tag consistency is validated.

[ ] Official releases build from clean tagged source.

[ ] Release archives include required licenses/notices.

[ ] A clean-machine binary installation test succeeds.

[ ] A clean-machine source-build test succeeds.

[ ] No previous decision-quality or runtime behavior regresses.
```

---

# 145. Required Implementation Sequence

Follow approximately this order:

```text
1. Review the final command/configuration surface from Phases 8–10.

2. Polish errors and help text.

3. Complete CPU/runtime detection.

4. Implement/polish diagnostics.

5. Implement/polish user-facing benchmark command.

6. Finalize model discovery/cache behavior.

7. Centralize version metadata.

8. Verify cross-platform path and runtime UX.

9. Write/rewrite README.md from a new-user perspective.

10. Add supporting docs only where README would become too large.

11. Create normal GitHub Actions CI.

12. Split model-free and real-model tests appropriately.

13. Create tag-driven GitHub release workflow.

14. Make release workflow reuse Phase 8 packaging scripts.

15. Add checksums and release manifests.

16. Smoke-test packaged artifacts.

17. Validate README Quick Start from the actual release artifact.

18. Validate source-build instructions from a clean checkout.

19. Review licenses/notices and release contents.

20. Produce the first release-ready artifact set.
```

Avoid unrelated decision-engine refactoring.

---

# 146. Final Implementation Report

When implementation is complete, provide the following.

## User Experience

Describe improvements to:

```text
help
errors
model discovery
background runtime UX
benchmark
diagnostics
```

## Commands

Show final user-facing command structure.

## Diagnostics

Show real examples of:

```bash
pjev diagnostics
pjev diagnostics --json
```

if both are implemented.

## CPU Detection

Explain:

```text
what is detected
which defaults it influences
how users override them
```

## Model Discovery / Cache

Document actual lookup order and cache location, if a cache exists.

## Versioning

Explain the single version source and show consistency across:

```text
pjev --version
/version
release manifest
archive names
```

## README.md

Summarize sections added.

Confirm all documented commands were tested.

## GitHub Actions

List workflow files and their triggers.

For each workflow, describe:

```text
platform matrix
build
tests
packaging
release behavior
```

## CI Results

Report actual successful jobs and any untested platforms.

## Release Workflow

Document:

```text
tag format
artifact names
checksums
release manifest
GitHub Release publication
```

## Release Artifacts

Show exact archive names and contents.

## Real-Model Validation

Explain how real-model smoke tests are performed and whether they run:

```text
on every PR
scheduled
manually
on release
```

## Clean-Machine Test

Report results of:

```text
download/unpack/run
```

without the source tree.

## Source Build Test

Report clean clone/build/test results using README instructions.

## Licensing

List included license/notice files and model-distribution status.

## Known Limitations

Document anything still incomplete, including:

```text
unsupported architectures
unsigned binaries
manual model acquisition
CI architecture gaps
model-size limitations
```

Do not overstate release maturity.

---

# Final Instruction

Phase 11 is the point where pjev stops feeling like an internal research repository and starts feeling like a coherent downloadable tool.

Do not add new decision mechanisms.

Do not restart model research.

Do not introduce unnecessary platform services or package managers.

Focus on:

```text
clarity
reliability
discoverability
diagnostics
reproducible releases
cross-platform CI
accurate documentation
```

The final acceptance test is:

> A new user discovers pjev on GitHub, reads README.md, downloads the correct release, follows the documented model setup, runs a decision successfully, understands the background runtime, and can diagnose common problems without reading the source code.

For developers, the equivalent test is:

> A new contributor clones the repository, follows README.md, builds pjev, runs the test suite, and sees the same build/test process exercised by GitHub Actions.

For maintainers, the final release test is:

> Pushing a valid release tag builds the verified platform artifacts through GitHub Actions, smoke-tests them, generates checksums, and publishes a GitHub Release whose artifacts and documentation agree on the version, platform, and runtime expectations.
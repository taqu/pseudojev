# Phase 8 — Native Distribution

## Objective

Implement Phase 8 of **pjev**.

The goal of this phase is to make pjev easy to distribute and run as a native application.

The command name is:

```bash
pjev
```

The target user experience is approximately:

```text
download
chmod +x
run
```

where applicable.

The core runtime already exists and must remain:

```text
pjev
  ↓
DecisionEngine
  ↓
llama.cpp
  ↓
GGUF model
```

This phase is about packaging and distribution.

Do not redesign inference, benchmarking, calibration, or server behavior.

---

# 1. Project Context

Assume previous phases have already established:

```text
Phase 1:
native C++ runtime

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

Phase 7:
runtime performance optimization
```

Phase 8 should package the already validated implementation.

Do not change behavior unless required for distribution correctness.

---

# 2. Primary Goal

Produce native pjev artifacts that are straightforward to distribute and run on supported platforms.

The initial target should be:

```text
one pjev executable
+
one GGUF model file
+
minimal configuration/assets if required
```

Do not require literal one-file distribution on the first iteration.

Treat these as separate concerns:

```text
1. native executable distribution
2. model distribution
3. optional model embedding
```

Solve them in that order.

---

# 3. Distribution Philosophy

Prefer:

```text
simple
predictable
portable
reproducible
```

over:

```text
clever
self-extracting
deeply customized
```

The final packaging should be easy to understand and debug.

Avoid mechanisms that make crashes, antivirus behavior, filesystem access, or mmap behavior difficult to reason about.

---

# 4. Supported Platforms

Target practical support for:

```text
Linux x86_64
Linux arm64
macOS arm64
Windows x86_64
```

Support additional targets only if the existing build is already known to work.

Do not claim a platform as supported unless it has been built and smoke-tested.

---

# 5. Build Matrix

Define an explicit build matrix.

For each supported platform record:

```text
OS
architecture
compiler/toolchain
build type
llama.cpp backend settings
artifact name
```

Example conceptual artifact names:

```text
pjev-linux-x86_64
pjev-linux-arm64
pjev-macos-arm64
pjev-windows-x86_64.exe
```

Use the repository's final naming convention consistently.

---

# 6. CMake Release Build

Create a reproducible release build path using CMake.

The expected workflow should be similar to:

```bash
cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-release --config Release
```

Adapt to the actual build system.

Document exact commands.

Do not rely on undocumented local IDE state.

---

# 7. Static Linking Strategy

Where practical, statically link pjev's native dependencies required for portability.

The goal is to minimize runtime dependencies.

However, do not force static linking where the target platform makes it inappropriate or harmful.

Document:

```text
what is statically linked
what remains dynamically linked
why
```

Do not claim "fully static" unless verified.

---

# 8. llama.cpp Integration

llama.cpp should be built as part of the pjev release build.

Do not require users to separately install llama.cpp.

Pin or otherwise reproduce the exact llama.cpp revision used by the release.

The binary and runtime behavior must correspond to the validated Phase 7 configuration.

---

# 9. CPU Backend Scope

Phase 8 should package the CPU runtime already validated by the project.

Do not add GPU backend complexity merely for distribution.

If optional accelerated backends already exist and are proven, they may remain separate artifacts.

Do not delay the core distribution work for them.

---

# 10. Model Distribution

Treat the GGUF model as a separate distribution artifact initially.

Recommended layout:

```text
pjev/
  pjev
  model.gguf
```

or equivalent.

The executable should have a simple, deterministic way to locate the model.

Do not require users to know internal repository paths.

---

# 11. Model Path Resolution

Define model resolution clearly.

A reasonable precedence may be:

```text
explicit CLI/config model path
        ↓
configured model directory
        ↓
model adjacent to executable
```

Use existing pjev configuration conventions if they already define this.

Document actual precedence.

Do not implement hidden filesystem scanning across arbitrary directories.

---

# 12. Model Identity

The distributed model must be the model selected in Phase 5.

Record:

```text
model name
GGUF filename
hash
quantization
size
```

Do not silently replace the model during packaging.

Distribution artifacts must preserve model provenance.

---

# 13. Model Integrity

Provide a way to verify the model file.

At minimum publish or include:

```text
SHA-256
```

or another existing project-standard digest.

pjev may optionally verify the expected model hash at startup if a release config requires it.

Do not silently accept an unexpected model when a frozen release configuration specifies one.

---

# 14. Model Missing Behavior

If the model cannot be found, pjev should fail clearly.

The error should tell the user:

```text
model not found
where pjev looked
how to specify --model or equivalent
```

Do not fall back to arbitrary files.

---

# 15. Distribution Configuration

Package the selected pjev decision configuration.

This includes the frozen values from previous phases, such as:

```text
prompt strategy
candidate scheme
prior correction
calibration parameters
language configuration
performance defaults
```

Do not hard-code experimental values opportunistically during packaging.

---

# 16. Configuration Packaging

Prefer one of:

```text
embedded small default configuration
```

or:

```text
small adjacent versioned config file
```

depending on the existing architecture.

The configuration should remain inspectable and versioned where practical.

Do not make runtime behavior depend on hidden build-time values that cannot be reproduced.

---

# 17. Default Experience

The ideal default experience is:

```bash
./pjev serve
```

when the expected model/configuration are placed in their documented default locations.

Avoid requiring many flags for the standard release.

Advanced users should still be able to override model and runtime configuration.

---

# 18. Do Not Break Explicit Configuration

Distribution convenience must not remove existing explicit options.

Users should still be able to specify:

```text
model path
config
host
port
threads
context size
```

where those options already exist.

Default discovery is convenience, not a replacement for explicit configuration.

---

# 19. Runtime Path Independence

Do not assume the current working directory is the installation directory.

Users may invoke pjev from another directory.

Model/config lookup relative to the executable should use the executable location, not the shell working directory, when that is the intended distribution behavior.

Implement this carefully per platform.

---

# 20. Executable Location

Create a small cross-platform abstraction for resolving the current executable path if needed.

Support:

```text
Linux
macOS
Windows
```

Keep platform-specific code isolated.

Do not spread OS-specific path logic throughout the codebase.

---

# 21. Filesystem Paths

Use native filesystem APIs and `std::filesystem` where practical.

Support:

```text
spaces
Unicode paths
relative paths
absolute paths
```

Do not construct shell commands for filesystem operations.

---

# 22. Windows Distribution

Windows artifacts should run without requiring a developer toolchain installation.

Document any unavoidable runtime dependencies.

Prefer producing:

```text
pjev.exe
```

plus the model/config assets.

Test paths containing spaces.

---

# 23. macOS Distribution

Produce a native macOS artifact for the selected architecture.

At minimum support:

```text
arm64
```

if that is the validated platform.

Document whether the artifact is:

```text
unsigned
ad-hoc signed
properly codesigned
```

based on what the project actually implements.

Do not claim notarization unless it is actually implemented.

---

# 24. Linux Distribution

Produce a native Linux artifact with a clear minimum compatibility target.

Document:

```text
architecture
libc requirements if dynamically linked
other runtime dependencies
```

Do not assume every Linux distribution has the same userspace.

---

# 25. Linux Portability

Where practical, minimize dependencies on unusually new system libraries.

If using glibc dynamically, document the effective minimum version discovered from the build environment.

If producing a more portable build, document how.

Do not claim universal Linux portability.

---

# 26. Architecture Detection

Phase 8 does not need automatic runtime architecture detection inside one binary.

Produce architecture-specific artifacts.

For example:

```text
linux-x86_64
linux-arm64
```

Do not introduce multi-architecture packaging complexity unless it is already natural on the platform.

---

# 27. macOS Universal Binary

A universal macOS binary is optional.

Do not delay Phase 8 completion to support:

```text
arm64 + x86_64
```

inside one file.

Separate artifacts are acceptable.

---

# 28. Release Artifact Layout

Define a clean release directory.

Conceptually:

```text
dist/
  pjev-<version>-linux-x86_64/
    pjev
    model.gguf
    config.json
    README.txt
```

or equivalent.

Keep it minimal.

Do not include source trees, build caches, object files, or benchmark artifacts.

---

# 29. Release Archive

Create compressed release packages where appropriate.

Examples:

```text
.tar.gz
.zip
```

Use platform-appropriate conventions.

Archive contents should unpack into one predictable directory.

---

# 30. Reproducibility Metadata

Each release package should contain or expose metadata including:

```text
pjev version
pjev git commit
llama.cpp revision
model hash
model quantization
build platform
build architecture
```

A machine-readable manifest is preferred.

---

# 31. Release Manifest

Create a release manifest.

Conceptually:

```json
{
  "pjev_version": "...",
  "pjev_commit": "...",
  "platform": "linux",
  "architecture": "x86_64",
  "llama_cpp_revision": "...",
  "model": {
    "filename": "...",
    "sha256": "...",
    "quantization": "..."
  }
}
```

Exact schema may differ.

Do not duplicate metadata inconsistently across multiple files.

---

# 32. Binary Version Metadata

`pjev --version` should expose enough information to identify the release.

At minimum:

```text
pjev version
build revision
```

Optionally:

```text
llama.cpp revision
```

if already available.

Keep output concise.

---

# 33. Release Build Identifier

Ensure release builds are distinguishable from development builds where useful.

For example:

```text
version
git commit
dirty state
```

Do not allow an uncommitted local build to masquerade as an official reproducible release without indication.

---

# 34. Debug Symbols

Do not ship large debug artifacts in the default user package unless required.

However, consider producing separate debug-symbol artifacts for development/support where practical.

Do not strip all useful diagnostic capability blindly.

---

# 35. Binary Stripping

Measure the effect of stripping release binaries.

If appropriate:

```text
strip release artifact
```

while preserving separate symbols if needed.

Do not apply platform-specific stripping commands incorrectly across systems.

---

# 36. Startup Behavior

Test startup from the unpacked distribution package.

The canonical smoke test is:

```bash
./pjev --version
./pjev serve
```

with the packaged model/configuration.

Do not test only from the build tree.

---

# 37. Relocation Test

Move the unpacked release directory to another filesystem location.

Then rerun:

```text
pjev --version
pjev serve
```

The release must not depend on the original build path.

---

# 38. Clean-Machine Principle

Where practical, test artifacts in an environment that does not contain the development checkout.

The release should not depend on:

```text
source tree
CMake build directory
developer environment variables
local llama.cpp installation
```

This is an important Phase 8 acceptance criterion.

---

# 39. Runtime Dependency Inspection

Inspect the final executable dependencies.

Use platform-appropriate tooling.

Document unexpected runtime dependencies.

Do not assume linking succeeded portably just because the binary runs on the build machine.

---

# 40. Model mmap Compatibility

The packaged model must continue to work with the validated llama.cpp model-loading strategy.

If mmap is used:

```text
adjacent GGUF file
```

should remain directly mmap-able.

Avoid packaging designs that force full extraction/copy on every startup unnecessarily.

---

# 41. Optional Model Embedding Experiment

Literal model embedding is optional.

Only evaluate it after the ordinary:

```text
executable + GGUF
```

distribution works reliably.

Do not make model embedding a Phase 8 completion requirement.

---

# 42. Model Embedding Tradeoffs

If model embedding is investigated, measure:

```text
binary size
startup behavior
memory mapping behavior
temporary extraction requirement
disk usage
update complexity
platform compatibility
```

Do not choose embedding purely because "one file" sounds cleaner.

---

# 43. Avoid Runtime Extraction by Default

Do not implement:

```text
embedded model
        ↓
extract multi-GB GGUF to temp directory every launch
```

as the default distribution strategy.

This can create significant:

```text
startup cost
disk writes
disk usage
antivirus interaction
cleanup complexity
```

Prefer an adjacent GGUF unless there is strong evidence for another design.

---

# 44. Immutable Model Resource

Treat the distributed model as read-only.

pjev must not modify the GGUF file.

Do not store runtime cache/state inside the model file.

---

# 45. Configuration Mutability

Do not modify packaged default config files silently at runtime.

If future user-specific configuration is needed, keep it separate from the immutable distribution assets.

Phase 8 does not need a full user config system.

---

# 46. Release Checksums

Generate checksums for distributable artifacts.

At minimum:

```text
release archive
model file
```

or follow the project's existing checksum convention.

Provide deterministic output names.

---

# 47. Supply Chain Metadata

Record upstream dependency revisions used in the release.

At minimum:

```text
llama.cpp
```

and any other significant bundled dependency.

Do not create a full SBOM system unless one already exists or is easy to generate.

A simple dependency manifest is sufficient.

---

# 48. License Files

Include required license/notice files for bundled code and model assets.

Do not perform new legal analysis.

Use the known project/model license information and preserve upstream notices.

If a required license is unknown, report that as a release blocker rather than guessing.

---

# 49. Build Automation

Create a reproducible release build script or CMake target.

For example:

```bash
./scripts/build-release.sh
```

or:

```bash
cmake --build build --target package
```

Use repository conventions.

Avoid requiring many undocumented manual steps.

---

# 50. Packaging Automation

The packaging process should:

```text
build release executable
collect required runtime files
copy selected config
copy or reference selected model
generate manifest
generate checksums
create archive
```

Keep the process inspectable.

---

# 51. Do Not Download During Build Unless Explicit

Avoid release builds that silently fetch an arbitrary current model or dependency.

Use pinned sources.

If downloading is unavoidable, require an explicit version/hash.

Reproducibility matters more than convenience.

---

# 52. Model Packaging Modes

Support at least one clear release mode:

```text
bundle model with release archive
```

Optionally support:

```text
binary-only artifact
```

for developers/users who already have the model.

Do not complicate this into many packaging modes unless useful.

---

# 53. Binary-Only Behavior

If distributed without a model, pjev should clearly require:

```bash
pjev serve --model /path/to/model.gguf
```

or equivalent.

Do not silently download models in Phase 8.

Automatic download/cache belongs to later UX work.

---

# 54. Release Size Reporting

Report:

```text
binary size
GGUF size
full package size
compressed archive size
```

This is part of evaluating distribution practicality.

Do not report only the executable size when the model dominates the package.

---

# 55. Performance Regression

Packaging must not materially change Phase 7 runtime behavior.

Run representative checks for:

```text
model load time
warm inference latency
RSS
```

using the packaged artifact.

Compare against the Phase 7 build.

Investigate significant differences.

---

# 56. Numerical Regression

Run the same fixed decision fixtures through:

```text
development build
release package
```

and verify equivalent:

```text
noul
choice
score
```

results.

Distribution changes must not affect model semantics.

---

# 57. Jev API Smoke Test

Run:

```bash
pjev serve
```

from the release package and execute a real request against:

```text
POST /api/v1/systemone/
```

Verify the packaged runtime remains Jev-compatible.

---

# 58. Health and Version Smoke Test

From every release artifact verify:

```text
GET /health
GET /version
```

work correctly.

The version endpoint must identify the packaged release accurately.

---

# 59. Multilingual Smoke Test

Include at least:

```text
one English decision
one Japanese decision
```

in the release smoke suite.

Do not rerun the full Phase 4 benchmark for every packaging step unless practical.

---

# 60. Cross-Platform Test Script

Create a shared smoke-test specification where possible.

It should test:

```text
pjev --version
pjev serve
health endpoint
version endpoint
noul
choice
score
clean shutdown
```

Adapt shell/PowerShell wrapper details per platform.

Keep semantic checks consistent.

---

# 61. Windows Smoke Test

Test:

```text
startup
model loading
paths with spaces
HTTP server
graceful shutdown
```

on Windows.

Do not infer Windows support from successful compilation alone.

---

# 62. macOS Smoke Test

Test:

```text
startup
model loading
HTTP server
shutdown
```

from an unpacked distribution archive.

If Gatekeeper/codesigning warnings exist, document actual behavior.

---

# 63. Linux Smoke Test

Test the release outside the source tree.

Where possible, use a clean container or VM approximating the intended compatibility baseline.

Do not accidentally depend on development packages.

---

# 64. CI Integration

If the repository has CI, add distribution build checks where practical.

A useful minimum is:

```text
build
package
basic smoke test
```

for supported hosts available in CI.

Do not require every architecture in every pull request if that is prohibitively expensive.

---

# 65. Release CI vs Normal CI

Separate expensive release packaging from fast normal development tests if needed.

For example:

```text
normal CI:
build + tests

release CI:
multi-platform package + smoke tests
```

Keep iteration speed reasonable.

---

# 66. Cross-Compilation

Cross-compilation is optional.

Prefer native builds when they are more reliable.

Do not build a complicated cross-toolchain matrix solely to say releases come from one machine.

Use CI/native builders if available.

---

# 67. Deterministic Artifact Naming

Use stable names containing:

```text
pjev version
OS
architecture
```

Example:

```text
pjev-1.0.0-linux-x86_64.tar.gz
```

Do not include random temporary identifiers in final artifact names.

---

# 68. Versioning

Use the project's chosen versioning system consistently.

The version should appear in:

```text
archive name
pjev --version
/version
release manifest
```

Do not maintain these manually in separate places.

---

# 69. No Automatic Updater

Do not implement an updater in Phase 8.

Users can download a new release manually.

Automatic update behavior belongs outside this phase.

---

# 70. No Installer Required

Do not require installers for Phase 8 completion.

A portable archive is sufficient.

Examples:

```text
tar.gz
zip
```

Installers may be considered in later UX/distribution polish.

---

# 71. No System Service

Do not install:

```text
systemd service
launchd service
Windows service
```

in this phase.

pjev remains an ordinary executable.

Background-runtime work belongs to a later phase.

---

# 72. No Shell Integration

Do not add:

```text
PATH installer
shell completions
desktop shortcuts
```

unless already trivial and established.

Focus on core distribution correctness.

---

# 73. Keep pjev Portable

The packaged executable should remain usable both as:

```text
pjev serve
```

and for any existing non-server pjev commands.

Do not create a distribution-specific server-only binary.

---

# 74. Artifact Verification Command

If useful, add a lightweight pjev command for inspecting the release/model.

For example:

```bash
pjev diagnostics
```

or:

```bash
pjev model inspect
```

Use existing commands if available.

Do not add new commands solely for packaging unless they materially help users verify installation.

---

# 75. Release Self-Check

A simple self-check mode may be useful if the project already has diagnostics.

It may verify:

```text
model found
model hash
configuration loaded
CPU backend available
```

Do not turn this into full Phase 11 diagnostics work.

---

# 76. Error Messages

Distribution-related errors should be user-friendly.

Examples:

```text
model file not found
unsupported model
config missing
config incompatible
binary/model version mismatch
```

Include actionable next steps.

Do not expose only internal exception messages.

---

# 77. Model/Config Compatibility

If a release config expects a specific model identity, verify compatibility at startup.

Do not allow:

```text
config for model A
+
model B
```

to run silently if that invalidates calibration or candidate assumptions.

Reuse existing configuration compatibility metadata from earlier phases.

---

# 78. Release Upgrade Compatibility

Do not implement full migration infrastructure.

However, version configuration schemas so incompatible future releases can fail clearly.

For example:

```text
config schema version
calibration artifact version
release manifest version
```

Avoid silently interpreting unknown schemas.

---

# 79. Embedded Defaults

Small default values may be embedded into the executable.

Examples:

```text
default local bind address
default model filename
default config filename
```

But tuned decision data should remain versioned and reproducible.

Do not bury large experimental artifacts inside arbitrary C++ constants without provenance.

---

# 80. Model Filename

Choose one stable default model filename for packaged releases.

For example conceptually:

```text
model.gguf
```

or a versioned explicit name.

Document it.

Do not use ambiguous wildcard discovery.

---

# 81. File Permissions

On Unix-like systems, ensure the executable has appropriate executable permissions in release archives.

The unpacked user experience should not require unexpected repair steps beyond:

```bash
chmod +x pjev
```

when archive/tooling cannot preserve them.

---

# 82. Temporary Files

The normal packaged runtime should not require temporary extraction of code or model assets.

Avoid unnecessary writes to:

```text
/tmp
AppData temp
system temp
```

during ordinary startup.

---

# 83. Read-Only Distribution Directory

Where practical, pjev should be able to run when its installation directory is read-only.

Do not assume runtime files can be written next to the executable.

Later caches/configs should live elsewhere if introduced.

---

# 84. Relative Config References

If config files reference model files, define whether paths are resolved relative to:

```text
config file location
current working directory
executable directory
```

Use one clear policy.

Prefer config-relative resolution for explicit config references.

Document it.

---

# 85. Security Basics

Do not weaken Phase 6 safety behavior for distribution convenience.

Preserve:

```text
local bind default
request limits
structured errors
safe logging
```

A release build must retain production-server protections.

---

# 86. Compiler Optimizations

Use standard Release optimization flags.

Do not add aggressive:

```text
-march=native
```

to generic release artifacts unless the artifact is explicitly CPU-specific.

A binary compiled for the build machine may fail on older CPUs.

Portability comes first.

---

# 87. CPU Baseline

Define the CPU instruction baseline for each release artifact.

Document any required instruction sets.

Do not accidentally enable unsupported instructions through local compiler defaults.

---

# 88. Architecture-Specific Optimized Builds

Architecture-specific optimized variants are optional.

Do not create a large matrix such as:

```text
AVX2
AVX512
Zen-specific
Intel-specific
```

in Phase 8 unless evidence strongly supports it.

Later CPU detection/polish may address this.

---

# 89. Release Performance Check

Verify that avoiding `-march=native` or using a portable CPU baseline does not introduce an unacceptable performance regression.

Record the tradeoff.

Do not secretly change CPU requirements to recover speed.

---

# 90. Packaging Tests

Add automated tests where practical for:

```text
manifest generation
checksum generation
archive contents
model/config inclusion
version consistency
path resolution
```

These should not require full benchmark execution.

---

# 91. Path Resolution Tests

Test:

```text
executable and model in same directory
explicit --model override
config-relative model path
working directory different from executable directory
path containing spaces
```

Cross-platform path bugs are a major Phase 8 risk.

---

# 92. Manifest Consistency Tests

Verify that:

```text
manifest version
binary version
archive version
model hash
```

match the actual packaged artifacts.

Do not generate metadata before final packaging if later steps can alter files.

---

# 93. Checksums After Packaging

Generate final checksums after artifacts are finalized.

Do not checksum an intermediate binary and then strip or modify it afterward.

---

# 94. Clean Release Build

The documented release procedure should begin from a clean build directory.

This helps catch hidden dependencies.

Avoid release scripts that depend on stale developer build outputs.

---

# 95. Phase 8 Exit Criteria

Phase 8 is complete when:

```text
[ ] Native release artifacts can be built reproducibly.

[ ] pjev is packaged as a native executable.

[ ] llama.cpp is included in the native build.

[ ] The selected GGUF model has a documented distribution strategy.

[ ] A standard release can run as executable + GGUF without development
    dependencies.

[ ] Model discovery/path resolution works reliably.

[ ] Explicit model-path override still works.

[ ] Release configuration is packaged/versioned.

[ ] Release manifest is generated.

[ ] Model and release checksums are generated.

[ ] pjev --version identifies the release.

[ ] /version identifies the release consistently.

[ ] Linux x86_64 artifact is built and smoke-tested if supported.

[ ] Linux arm64 artifact is built and smoke-tested if supported.

[ ] macOS arm64 artifact is built and smoke-tested if supported.

[ ] Windows x86_64 artifact is built and smoke-tested if supported.

[ ] Unsupported/unverified targets are not advertised as supported.

[ ] Packaged pjev runs outside the source/build tree.

[ ] Relocating the release directory does not break normal startup.

[ ] Paths containing spaces work on relevant platforms.

[ ] Jev-compatible API smoke test passes from packaged artifacts.

[ ] noul/choice/score smoke tests pass.

[ ] English/Japanese smoke tests pass.

[ ] Graceful shutdown still works.

[ ] Phase 7 performance does not materially regress.

[ ] Numerical outputs match the validated build.

[ ] Required license/notice files are packaged.

[ ] Model embedding is not required for Phase 8 completion.
```

---

# 96. Required Implementation Sequence

Follow approximately this order:

```text
1. Freeze Phase 7 runtime/model/configuration.

2. Define supported platform matrix.

3. Create clean Release build configuration.

4. Define release artifact naming.

5. Package native executable.

6. Package selected config.

7. Define model placement and path resolution.

8. Add release manifest/version metadata.

9. Add checksums.

10. Create platform archives.

11. Test outside the source/build tree.

12. Test relocation and path handling.

13. Run server/API smoke tests.

14. Run numerical regression checks.

15. Run performance sanity checks.

16. Add CI/release automation where practical.

17. Document distribution procedure.

18. Only then evaluate optional model embedding if still desired.
```

Avoid mixing distribution work with unrelated runtime refactoring.

---

# 97. Final Implementation Report

When implementation is complete, provide the following.

## Supported Platforms

List:

```text
platform
architecture
status
```

Clearly distinguish:

```text
verified
build-only
unsupported
```

## Build Configuration

Document:

```text
compiler
CMake options
CPU baseline
static/dynamic dependency strategy
llama.cpp revision
```

## Artifacts

List exact artifact names.

Example:

```text
pjev-<version>-linux-x86_64.tar.gz
...
```

Use actual generated names.

## Package Contents

Show the release directory/archive layout.

## Model Distribution

Explain:

```text
model filename
model location
hash
quantization
model lookup precedence
explicit override behavior
```

## Configuration

Explain which default configuration ships with the release and how compatibility is verified.

## Version Metadata

Show actual output from:

```bash
pjev --version
```

and the `/version` response.

## Dependency Inspection

Document native runtime dependencies for each verified platform.

## Smoke Tests

Report results for:

```text
startup
health
version
noul
choice
score
English
Japanese
shutdown
```

## Path Tests

Report:

```text
different working directory
relocated installation
spaces in path
explicit model override
```

## Numerical Regression

Confirm the packaged runtime matches the validated Phase 7 reference.

## Performance Regression

Report representative:

```text
model load
warm latency
RSS
```

compared with Phase 7.

## Release Size

Report:

```text
executable size
GGUF size
archive size
```

## Checksums

Document how checksums are generated and verified.

## CI / Automation

Explain how release artifacts are built and tested.

## Model Embedding

If investigated, report findings separately.

Do not make it appear required if the standard adjacent-GGUF distribution already works.

## Known Limitations

Document issues such as:

```text
platforms not yet verified
dynamic runtime dependencies
unsigned macOS artifacts
CPU baseline constraints
large model download size
```

Do not hide them.

---

# Final Instruction

Phase 8 is about making the already validated pjev runtime easy to distribute.

Do not redesign the product.

Do not optimize inference again.

Do not add user-facing convenience systems that belong to later phases.

First make this work reliably:

```text
native pjev executable
+
validated GGUF model
+
versioned configuration
```

with reproducible builds, predictable path resolution, clear version/provenance metadata, and smoke-tested artifacts.

A literal one-file executable containing the model is optional.

Prefer a boring, reliable native distribution over a clever single-file mechanism that complicates mmap, startup, updates, debugging, or cross-platform support.

The Phase 8 deliverable should be something a user can unpack and run without needing the source tree or a development environment.
# R7 — Bundle Bonsai2 2B and Its Calibration for Release

## Objective

Prepare the `pjev` release candidate with Bonsai2 2B as the bundled default inference model and a matching, pre-generated calibration configuration.

Users should be able to run `pjev` immediately after installation without manually generating calibration parameters.

Preserve the R5 task-aware routing policy and the R6 calibration generation functionality.

R7 is a release integration and packaging task, not a new model-selection or hyperparameter-tuning experiment.

## 1. Bundled Runtime Resources

The release must provide these default resources:

```text
models/
  bonsai2_2b.gguf
  nomic-embed-text-v1.5.gguf
  examplar.db
  calibration.json
```

Bonsai2 2B is the default inference model.

The calibration file must be generated specifically for this model, using the actual production candidate formulation.

Preserve existing support for user-specified GGUF models.

Users must still be able to regenerate calibration with `pjev calibrate`.

Before packaging, verify the redistribution licenses and required notices for each bundled model, database, and third-party component.

Do not assume that permission to download a resource implies permission to redistribute it.

If redistribution is prohibited or uncertain, report the blocker and do not include that asset in a public release.

## 2. Model Identity

The expected Bonsai2 2B GGUF SHA-256 from R6 is:

`36061ad406df8a4903a6db005be350c50fabfaf717f9ef0acf4b76adcd9929fa`

Verify the SHA-256 of the actual bundled GGUF file.

Do not assume the file is unchanged merely because its filename matches.

The bundled calibration must contain the SHA-256 of the exact GGUF file included in the release.

The existing production calibration loader must verify this identity before applying calibration.

Never silently apply the bundled calibration to a different model.

## 3. Generate the Release Calibration

Use the existing R6 production command:

```bash
pjev calibrate \
  --model models/bonsai2_2b.gguf \
  --database models/examplar.db \
  --output models/calibration.json \
  --seed 1 \
  --max-samples 3000 \
  --validation-ratio 0.2
```

Use the validated R6 generation configuration unless there is a documented compatibility or reproducibility reason to change it.

Do not manually copy parameter values from the R6 report into JSON.

Use the actual fitter output.

Before generating:

1. Verify the bundled model hash.
2. Verify the exemplar database is readable.
3. Verify the production formulation matches the calibration generator.
4. Preserve a backup or recoverable copy of the previous calibration.
5. Confirm there are sufficient eligible calibration samples.

The generator must atomically replace the output only after successful fitting and validation.

If generation fails, preserve the existing file and fail the release-preparation step.

## 4. Calibration Format

Use the schema currently implemented and validated by R6.

The R6 implementation retained calibration schema version 1 with task-specific parameter sets.

Do not introduce calibration schema version 2 merely for release packaging.

Preserve these fields:

- `version`
- `method`
- `model`
- `formulation`
- `parameters`

Use the production formulation:

- Candidate scheme: `letters`
- Layout: `auto`
- Prior correction: enabled

The calibration JSON must contain fitted parameters for Choice, Noul, and Score.

### Important path behaviour

R6 did not implement separate primary and fallback calibration profiles because the real calibration run produced no Choice or Noul fallback samples.

Therefore, do not claim that the release contains independently fitted fallback profiles.

Use the existing R6 runtime semantics.

Score calibration applies to its zero-shot LLM fallback, not to its primary kNN-5 classifier.

Choice and Noul calibration follow the current production implementation.

Document this limitation accurately.

## 5. Default Calibration Loading

Ensure that `pjev` automatically loads the bundled calibration during normal initialization.

Expected default:

`models/calibration.json`

The default model and default calibration must be compatible without requiring additional command-line arguments.

Loading must validate:

- JSON syntax
- Schema version
- Calibration method
- Candidate formulation
- Model identity
- Parameter ranges
- Required task parameters

Do not silently accept incompatible calibration.

Preserve the current runtime behaviour for missing calibration files and custom models.

If a custom model does not match the bundled calibration, it must run without applying that calibration, or use a separately generated compatible calibration file.

Clearly report the calibration status through existing diagnostics.

## 6. Preserve Task-Aware Routing

Do not change the selected production policy:

| Task | Primary strategy |
|---|---|
| Choice | Nearest 2-shot, Order A |
| Noul | Nearest 1-shot |
| Score | Weighted kNN-5 |

Preserve the existing zero-shot fallback when the required number of valid exemplars is unavailable.

Apply calibration only where the production implementation supports it.

Do not calibrate kNN similarity weights using LLM temperature parameters.

Do not change prompt construction, candidate scoring, retrieval filters, context size, or fallback conditions.

## 7. Package Resource Resolution

Ensure default resource paths work in an installed or extracted release package, not only when executing from the repository root.

Use the project's existing resource-path configuration.

If necessary, implement a deterministic resource lookup order such as:

1. Explicit user-provided path.
2. Resource directory configured by the application.
3. Resource directory relative to the installed executable.
4. Existing development checkout defaults.

Do not silently select resources from unrelated working directories.

Keep model, database, and calibration paths independently configurable.

Ensure paths containing spaces or non-ASCII characters are handled correctly.

## 8. Release Build and Packaging

Prepare a release package containing:

- Production executable and required libraries
- Bonsai2 2B GGUF
- Embedding model
- Exemplar database
- Matching calibration JSON
- Required licenses and notices
- Production documentation

Do not include:

- R1–R4 experimental executables
- Experimental prediction dumps
- Benchmark tuning datasets
- Temporary calibration outputs
- Debug logs
- Unnecessary development artifacts

If model files are too large for the normal source repository, use an appropriate binary release packaging mechanism.

Do not add large binaries to Git merely to make packaging convenient.

Ensure the final release archive is self-contained where redistribution rights permit.

## 9. Calibration Validation

After generating `models/calibration.json`, validate it using the real bundled model.

Required checks:

1. The generated file loads through the production calibration loader.
2. Model SHA-256 verification succeeds.
3. Formulation compatibility succeeds.
4. Calibration status is `enabled`.
5. All fitted temperatures are finite and positive.
6. All prior-alpha values are valid.
7. Calibration improves or preserves validation NLL according to the R6 acceptance gate.
8. Score kNN-5 behaviour remains unchanged.
9. Zero-shot fallback applies the appropriate task calibration.
10. No normal-path routing behaviour changes.

Use existing R6 validation artifacts when reproducibility permits, but ensure the actual packaged files are verified.

Do not treat successful JSON parsing alone as sufficient validation.

## 10. Release Smoke Tests

Run smoke tests from the final packaged directory.

Do not rely on source-tree-relative paths.

Test:

- Default startup
- Default model loading
- Default embedding model loading
- Default database loading
- Automatic calibration loading
- Choice inference
- Noul inference
- Score inference
- Missing-exemplar fallback
- Custom model override
- Calibration incompatibility detection
- `pjev calibrate` command availability
- Diagnostic JSON output

Run existing regression tests.

Preserve exact prediction checks where determinism is expected.

If calibration changes predictions relative to the uncalibrated R5/R6 golden set, use separate calibrated and uncalibrated regression expectations.

Do not overwrite old golden results without explaining the change.

## 11. Documentation

Update the production README to explain that Bonsai2 2B and its matching calibration are bundled by default.

Document:

- Default runtime resources
- Automatic calibration loading
- Model/calibration identity matching
- How to use another GGUF model
- How to regenerate calibration
- Calibration limitations
- Zero-shot fallback behaviour
- Redistribution notices

Include a short example:

```bash
pjev calibrate \
  --model models/my-model.gguf \
  --database models/examplar.db \
  --output models/my-model-calibration.json
```

Explain that users who replace Bonsai2 2B should generate calibration for their replacement model.

Do not claim that the bundled calibration is universally valid across models.

## 12. Release Validation Report

Create or update:

`docs/release-candidate.md`

Include:

- Release commit SHA
- Bundled model SHA-256
- Exemplar database identity
- Calibration file SHA-256
- Calibration generation command
- Calibration schema version
- Calibration validation metrics
- Build and test results
- Package contents
- License verification results
- Known limitations

Specifically document that the R6 Score calibration reached the fitting bounds.

Do not claim that these calibration metrics are directly comparable to R4 held-out benchmark accuracy.

## 13. Acceptance Criteria

R7 is complete when:

- Bonsai2 2B is included in the release package, subject to redistribution permission.
- The bundled GGUF hash matches the calibration model identity.
- `models/calibration.json` contains calibration generated by the R6 production command.
- Calibration is loaded automatically on default startup.
- The loader reports calibration as enabled and model-verified.
- Default Choice and Noul inference use compatible calibration.
- Score kNN-5 remains unaffected.
- Score zero-shot fallback uses Score calibration.
- Task-aware routing remains unchanged.
- Users can still generate calibration for custom GGUF models.
- Clean Release builds and regression tests pass.
- Packaged runtime smoke tests pass.
- Licenses and notices are included.
- Experimental R1–R4 executables remain excluded.
- The release validation report accurately documents the packaged artifacts.

Do not tag, publish, or push a public release without explicit authorization.

The final deliverable is a validated, self-contained release candidate containing Bonsai2 2B and its matching pre-generated calibration.
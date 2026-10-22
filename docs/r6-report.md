# R6 — Model-specific calibration generation: implementation and validation report

`pjev calibrate` generates `calibration.json` for a user-selected GGUF model from `examplar.db`, using
the production decision path's uncalibrated candidate logits and the stored ground-truth labels. The
runtime loads the result automatically and applies it only to the model it was fitted for.

## Implementation

| Area | Change |
|---|---|
| CLI | `pjev calibrate` (`--model`, `--database`, `--output`, `--seed`, `--max-samples`, `--validation-ratio`, `--batch-size`, `--threads`, `--verbose`, plus the runtime resource options). Defaults: the runtime's database and calibration paths (`models/examplar.db`, `models/calibration.json` in a checkout). |
| `src/calibration/calibration_generator.*` | Partition, query construction, logit collection, fitting, validation gate, report, atomic output. |
| `src/calibration/calibration_fit.*`, `calibration_metrics.*` | The project's fitter and metrics, restored from history (removed with the R0–R4 executables) and adapted: NLL by log-sum-exp (`log1p`), decimal alpha grid, boundary flags, alpha chosen on fitting data only, confidence metrics added. |
| `src/calibration/calibration.*` | `check_calibration_model`: an artifact's `model` (GGUF SHA-256) is verified against the loaded model; a mismatch or an unverifiable identity makes the calibration incompatible (not applied). Empty `model` (older files) is not checked. |
| `src/retrieval/exemplar_store.*` | `scan()` (row metadata, read-only) and a held-out set that removes rows from search and eligibility. |
| `src/routing/router_runtime.*` | Model identity check before the engine is built; accessors for the router, engine and store. |
| `pjev diagnostics --load --json` | Load logs go to stderr so stdout stays valid JSON (a regression from R5's calibration log lines). |

No routing policy, prompt, candidate scoring or context-size behaviour changed. No R1–R4 executable
or experiment infrastructure was restored.

### Data and partitioning

- Schema inspected: `examples` (id, group_id, kind, question, answer_index, number_of_levels, domain,
  family, LZ4 payloads, target_lz4), `groups` (group_id, source), `exemplar_vectors`, `db_info`
  (schema 4). The database has 270,573 rows; the allowlist holds the 111,596 training-split rows with
  one-hot answers.
- Queries come only from the allowlist; the 158,977 other rows (validation, test, external) are never
  read as queries.
- Protected unit: `group_id`. Groups are visited in seeded XXH64 order, one query per kind per group;
  a group is wholly fitting or validation (seeded hash, `--validation-ratio`), and **all of its rows
  leave the retrieval pool**. The production content-key rule additionally excludes identical-content
  exemplars. `groups.source` names the source dataset (68 values, e.g. `shenjunhao/OpenJevData-140k`)
  and is not a scenario family, so no coarser grouping is claimed.
- Each candidate row is decoded and built as a client request (`make_query`, then `route_task`);
  invalid rows are rejected with a reason (`undecodable_payload`, `no_one_hot_answer`,
  `ambiguous_answer`, `invalid_noul_options`, `level_count_mismatch`, …) and do not consume quota.
- Insufficient data (fewer than 50 fitting or 20 validation samples for any kind) fails; there is no
  reduced-data mode.

### Inference paths

Choice and noul run through the production `TaskRouter` with an uncalibrated engine (nearest 2-shot
order A / nearest 1-shot, with the production zero-shot fallback). Score runs S1 zero-shot, the only
path its parameters apply to; kNN-5 is untouched. The executed strategy and fallback reason are
recorded per sample, and the report gives metrics per path (and a path-specific fit when a secondary
path has at least 50 fitting samples).

**Schema decision.** The version-1 schema (one parameter set per kind) was kept. Path-specific
parameters were not introduced: with the retrieval pool at ~102k rows, no choice or noul query fell
back in the real run (0 of 2,000), so a fallback-specific fit could not be estimated, let alone
justified. The report flags a path whose validation NLL the task-level parameters worsen.

### Fit and acceptance

Per kind on the fitting partition: alpha grid 0–2 step 0.1 × golden-section on log T in [0.01, 100],
minimizing NLL of `softmax((raw − alpha × blank_prompt_logits) / T)` — the transformation the decision
engine applies. Degenerate data (one label, constant logits), non-converged or out-of-range results
fail. The calibration is accepted only if no kind's validation NLL increases; otherwise only a failed
report is written and the existing file is kept.

## Validation

### Real model (`bonsai2_2b.gguf`, sha256 `36061ad4…29fa`), defaults

3,000 queries (1,000 per kind), seed 1, validation ratio 0.2; 2,010 groups (9,153 rows) held out;
0 rejected; 1,458 s of inference on an i9-13900 (CPU only); prompts 194–2,186 tokens (median 613).

| Kind (path) | T | alpha | n val | Accuracy | NLL | Brier | ECE |
|---|---|---|---|---|---|---|---|
| choice (nearest 2-shot) | 6.274 | 0.2 | 179 | 0.497 → 0.508 | 1.877 → 1.050 | 0.793 → 0.612 | 0.344 → 0.060 |
| noul (nearest 1-shot) | 3.618 | 0.7 | 216 | 0.782 → 0.787 | 0.995 → 0.502 | 0.409 → 0.331 | 0.195 → 0.081 |
| score (zero-shot) | 100 ⚠ | 2.0 ⚠ | 218 | 0.165 → 0.188 | 3.085 → 1.554 | 1.237 → 0.786 | 0.525 → 0.061 |

Uncalibrated predictions were heavily overconfident (mean max probability 0.84 / 0.96 / 0.68 for
choice / noul / score). ⚠ Score sits on both search bounds: its zero-shot logits carry almost no
signal (accuracy near chance), so the fit flattens them to near-uniform. The supplied
`calibration.json` has the same `score_temperature` (99.99999952490862, the T bound reached by the same
search), which suggests it was produced by this fitter on the `natural` formulation. These figures are
on training-split queries with their groups held out; they are not comparable with the R4 accuracy
figures.

### Release checklist

| # | Item | Result |
|---|---|---|
| 1 | Clean Release build | `cmake --build build --config Release --clean-first`: 0 errors, no compiler warnings in project sources; 18/18 tests pass on the clean binaries. (A fresh build directory under `%TEMP%` fails in llama.cpp's Vulkan shader-generator sub-build, which cannot find a compiler there; unrelated to R6.) |
| 2 | Existing R5 inference tests | 18/18 `ctest` pass, including the golden regression (24/24 predictions, logits exact) |
| 3 | Generation command works | Real run above (exit 0); fixture run in `test_calibrate` |
| 4 | Generated JSON loads | Production loader: `calibration: enabled (…; prior correction on; model verified)`; `diagnostics --load` → `enabled` |
| 5 | Model identified | Artifact `model` = GGUF SHA-256; another model's artifact → `incompatible`, decisions uncalibrated (golden test and CLI) |
| 6 | Database read-only | SHA-256 of `examplar.db` identical before and after (`b7cd1292…`); checked by the command itself and by `test_calibrate` |
| 7 | Task-specific parameters | Six parameters fitted per kind |
| 8 | Production formulation | `layout: auto`, `candidate_scheme: letters`, `prior_correction: true` from `task_policy.h` |
| 9 | Score kNN-5 unchanged | Golden test (all calibration states) and `test_calibrate` (generated artifact) |
| 10 | Missing-exemplar fallback | `test_calibrate`: noul queries with an empty pool decided zero-shot and recorded as `no_eligible_exemplar`; generated score parameters applied to the zero-shot fallback |
| 11 | Invalid calibration never replaces a valid one | Unit tests (malformed, invalid, other model, other formulation, altered content) and `test_calibrate` (insufficient, degenerate, validation-rejected); CLI insufficient-data run left the file byte-identical |
| 12 | No R1–R4 executables | Only `pjev` and `pjev-filldb` are built |

Tests added: `test_calibration` (fit determinism, temperature and alpha recovery on synthetic data,
boundaries, NLL / Brier / ECE, stable log-sum-exp, partition atomicity and leakage, query mapping,
atomic output, model identity) and `test_calibrate` (end to end on a fixture database: leakage checked
on every evaluated prompt, paths, ground-truth alignment, determinism, loader, read-only database,
replacement and failure cases, generated parameters applied by the router).

## Open points

- The repository's `models/calibration.json` was **not** replaced; the real-run output is a separate
  file. Running `pjev calibrate` with defaults replaces it (atomically, after validation).
- Score zero-shot calibration is effectively "make it uniform". Applying it is correct for probability
  quality, but the score fallback stays weak; that is a model limitation, not a calibration one.
- Validation sets of ~200 per kind give coarse ECE estimates; raise `--max-samples` for tighter ones
  (about 0.5 s per query on CPU).

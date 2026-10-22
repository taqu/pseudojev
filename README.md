# pjev

[![CI](https://github.com/taqu/pseudojev/actions/workflows/ci.yml/badge.svg)](https://github.com/taqu/pseudojev/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**pjev** is a small, local, Jev-compatible decision engine. It answers three kinds of decision
questions about a piece of state — `noul` (true/false), `choice` (pick one option) and `score`
(pick a level on an ordered scale) — using a local GGUF language model through
[llama.cpp](https://github.com/ggerganov/llama.cpp), a local embedding model, and a bank of solved
example tasks. No network or cloud service is involved.

> **Status:** release candidate 0.9.0. Validated on English tasks, Windows x86_64.
> The API and output formats are pre-1.0 and may change.

---

## How pjev decides

Every decision is routed by task kind (policy `B_task_only@A_latency`):

| Task kind | Strategy | Inference model |
|---|---|---|
| `choice` | **Nearest 2-shot.** The two most similar solved exemplars are shown as earlier turns — second-nearest first, then the nearest — followed by the question. The candidate with the highest logit wins. | 1 evaluation |
| `noul` | **Nearest 1-shot.** The most similar solved exemplar is shown as an earlier turn, followed by the question. | 1 evaluation |
| `score` | **Similarity-weighted kNN-5.** The five most similar exemplars with the same number of levels vote with their answer level; weights are `exp((s - s_max) / 0.005)` on the estimated cosine similarity `s`. | none |

Retrieval: the task (kind, domain, family, question, state, and for `score` the level texts) is
embedded with `nomic-embed-text-v1.5` and matched against the exemplar database by Hamming distance
on binary vectors. Only exemplars of the same kind (for `score`: the same number of levels) from the
training split are eligible. Ties are broken by database row order, so decisions are deterministic.

Each strategy needs its full number of valid exemplars: 2 for `choice`, 1 for `noul`, 5 for `score`.
When fewer are found, the task is decided zero-shot instead (S1: the question alone, scored by
candidate logits; 1 evaluation). See [Fallback behaviour](#fallback-behaviour).

This policy was selected in the R4 evaluation: 56.75% overall accuracy on the held-out partition
(choice 45.13%, noul 90.62%, score 35.79%), versus 55.05% for nearest 1-shot everywhere and 44.69%
for zero-shot. These figures describe that benchmark only; accuracy on other tasks will differ.

---

## Quick start

1. Build pjev (see [Building from source](#building-from-source)) or unpack a release archive.
2. Put the four [runtime resources](#runtime-resources) next to the executable (or point to them).
3. Run:

```bash
./pjev diagnostics          # checks that every resource is found
./pjev noul --state "The sky is blue." --question "Is it daytime?"
```

---

## Runtime resources

| Resource | Default file name | Flag | `pjev.json` key |
|---|---|---|---|
| Inference model (Ternary-Bonsai-2-2B, GGUF) | `bonsai2_2b.gguf` | `--model` | `default_model` |
| Embedding model (nomic-embed-text-v1.5, GGUF) | `nomic-embed-text-v1.5.gguf` | `--embedding-model` | `embedding_model` |
| Exemplar database (SQLite) | `examplar.db` | `--exemplar-db` | `exemplar_db` |
| Exemplar allowlist (text) | `examplar-allowlist.txt` | `--exemplar-allowlist` | `exemplar_allowlist` |
| Calibration artifact (JSON, optional) | `calibration.json` | `--calibration` / `--no-calibration` | `calibration` |

They are included in the release archive.
They are also stored in this repository under `models/` with Git LFS:

```bash
git lfs install
git clone https://github.com/taqu/pseudojev.git
cd pseudojev && git lfs pull
cp models/bonsai2_2b.gguf               <install-dir>/model.gguf
cp models/nomic-embed-text-v1.5.gguf    <install-dir>/
cp models/examplar.db                   <install-dir>/
cp models/examplar-allowlist.txt        <install-dir>/
```

Check the licenses before redistributing them: the inference model is Apache-2.0
(`models/LICENSE-bonsai.txt`); nomic-embed-text-v1.5 is Apache-2.0 according to its upstream model
card; the database contains Open-Jev v1.1 data (CC0 1.0 for original content, CC BY 4.0 for
WANLI-derived records; see `doc/Open-Jev-v1.1/`) and OpenJevData-140k records (MIT; see
`doc/OpenJevData-140k/`).

**The resources belong together.** The routing policy was validated with exactly these files
(inference model sha256 `36061ad4…`, embedding fingerprint `3e24342164b3d949`). pjev refuses to start
when the embedding model does not match the database; another inference model will run but is not
validated.

**Lookup order**, per resource:

1. the command-line flag;
2. the key in `pjev.json` next to the executable (a bare file name is relative to the executable);
3. the default file name next to the executable;
4. `models/<file>` relative to the working directory (development checkout; the inference model is
   `models/bonsai2_2b.gguf`).

`pjev diagnostics` shows the resolved paths; `pjev diagnostics --load` also loads everything and
checks compatibility.

### Calibration

The calibration artifact (`models/calibration.json` in a checkout, `calibration.json` next to the
executable in a release) is read once at start-up, with no flag needed. It holds per-kind
temperatures and prior-correction alphas, applied exactly as stored by the existing calibration
implementation: prior correction (candidate logits − alpha × blank-prompt candidate logits, one blank
prompt per kind and candidate count), then temperature scaling. It applies to every decision that
scores candidate logits — `choice` nearest 2-shot, `noul` nearest 1-shot and the zero-shot fallback
of all kinds (`score` zero-shot uses `score_temperature` / `score_prior_alpha`). Weighted kNN-5 votes
for `score` are never calibrated.

| Situation | Behaviour |
|---|---|
| Default file absent | Warning; decisions are uncalibrated |
| Valid artifact for the production formulation | Calibration enabled |
| Artifact fitted for another formulation (`candidate_scheme`, `layout`) | **Error logged; not applied**; decisions are uncalibrated |
| Artifact fitted for another inference model (its `model` SHA-256 differs from the loaded GGUF, or is not a SHA-256) | **Error logged; not applied**; decisions are uncalibrated |
| Malformed or unsupported file (JSON, version, method, missing or invalid formulation / parameters, temperature ≤ 0, alpha < 0) | Configuration error; pjev does not start |
| Path set with `--calibration` or `pjev.json` but absent | Configuration error |
| `--no-calibration` | Uncalibrated (the R4 reference behaviour) |

The production formulation is `layout: auto`, `candidate_scheme: letters` (the validated R4 prompt).
**The supplied `models/calibration.json` was fitted for `candidate_scheme: natural`** and is therefore
reported as incompatible and not applied: release decisions are uncalibrated until a calibration is
generated for the model with [`pjev calibrate`](#generating-a-calibration-pjev-calibrate). The state is
shown by `/version` (`calibration`), `pjev status`, `pjev diagnostics --load` and the start-up log.
The R4 accuracy figures describe uncalibrated decisions.

An artifact's `model` is the SHA-256 of the GGUF file it was fitted for. It is checked when the
runtime loads the model; an empty `model` (artifacts from earlier releases) is not checked.

#### Generating a calibration (`pjev calibrate`)

Calibration is model-specific: temperatures and prior alphas fitted for one GGUF file say nothing
about another. Regenerate the calibration whenever the inference model changes:

```bash
pjev calibrate \
  --model models/my-model.gguf \
  --database models/examplar.db \
  --output models/calibration.json
```

It needs the inference model, the embedding model, `examplar.db` and `examplar-allowlist.txt` (the
same resources as decisions; no existing calibration file). With no options it calibrates the default
model and writes the file the runtime loads by default (`models/calibration.json` in a checkout,
`calibration.json` next to the executable in a release).

| Option | Description |
|---|---|
| `--model PATH` | Inference model to calibrate |
| `--database PATH` | Exemplar database, opened read-only (default `examplar.db`) |
| `--output PATH` | Calibration file to write (default: the runtime's calibration path) |
| `--seed N` | Seed of the query sampling and the fitting / validation partition (default 1) |
| `--max-samples N` | Maximum calibration queries **per task kind** (default 1000) |
| `--validation-ratio R` | Fraction of query groups held out for validation (default 0.2) |
| `--batch-size N` | Queries per processing batch and progress report (default 32). Inference itself is one prompt at a time, as at runtime. |
| `--threads N` | CPU threads (default: physical cores) |
| `--verbose` | One line per query (path, prompt tokens, correct candidate, argmax) |
| `--embedding-model`, `--exemplar-allowlist`, `--ctx-size`, `--n-gpu-layers`, `--main-gpu` | As for decisions; the context size defaults to the runtime's |

What it does:

1. **Queries and leakage prevention.** Calibration queries are training-split exemplars (the
   allowlist) with exactly one correct answer that maps to one candidate; validation, test and
   external rows are never used. Each candidate query is decoded and built as the request a client
   would send; rows that cannot be (undecodable payload, ambiguous answer, invalid options, …) are
   rejected with a recorded reason. Queries are drawn by protected group (`group_id`, the variants of
   one source task) in seeded order, one per task kind per group, and **every row of a selected group is
   removed from the retrieval pool**, so no query is answered with itself or a sibling variant as a
   demonstration. A group belongs wholly to fitting or to validation. Exemplars with the query's content
   are excluded as at runtime. The database records no scenario family beyond `group_id`
   (`groups.source` is the source dataset), so no coarser grouping is enforced.
2. **Logits.** Each query runs through the production decision path, uncalibrated: `choice` nearest
   2-shot (order A) and `noul` nearest 1-shot through the task router, with its zero-shot fallback
   when the pool lacks exemplars; `score` through S1 zero-shot, the LLM path its parameters apply to
   (kNN-5 votes are never calibrated). The raw candidate logits, the correct candidate index and the
   executed path (primary or fallback, with the reason) are recorded for every query.
3. **Fit.** Per task kind, on the fitting queries only: prior alpha on a grid (0–2, step 0.1) and
   temperature by golden-section search on log T (0.01–100), minimizing negative log-likelihood of
   `softmax((logits − alpha × blank-prompt logits) / T)` — exactly what the runtime applies. Results on
   a search bound are reported.
4. **Validation.** Uncalibrated and calibrated metrics on the validation queries, per task kind and
   per executed path: accuracy, NLL, Brier score, ECE (15 bins, with reliability bins), mean maximum
   probability, and mean confidence on correct and incorrect predictions. **The calibration is
   accepted only if it does not increase validation NLL for any task kind.** Accuracy, Brier or ECE
   regressions are reported as warnings.
5. **Output.** `calibration.json` (version 1, the production formulation, the model's SHA-256) is
   written to a temporary file, read back through the production loader for this model, and then
   atomically renamed over the output. A report is written next to it (`calibration.report.json`):
   model, database and allowlist SHA-256, seed, partition, counts by task kind (eligible, examined,
   rejected by reason, processed, accepted), fitted parameters, metrics and per-query logits (no task
   text).

Calibration adjusts probabilities. Prior correction also shifts candidate logits, so it can change
which candidate wins; calibration does not necessarily improve accuracy, and lower confidence alone is
not better calibration — compare the validation NLL, Brier score and ECE in the report.

Errors (exit code 1; an existing calibration file is never replaced and no partial file is left):

| Situation | Result |
|---|---|
| Fewer than 50 fitting or 20 validation samples for a task kind | `insufficient calibration data`; nothing written. There is no reduced-data mode. |
| Every fitting sample of a kind has the same correct candidate, or the logits never differ | `degenerate … data`; nothing written |
| The fit does not converge to finite parameters within bounds | `invalid … optimization result`; nothing written |
| Validation NLL increases for a task kind | `calibration rejected`; only `calibration.failed-report.json` is written |
| Database, model or inference failure | Reported; nothing written |

Resources: the same memory as a decision process (both models plus the database); one LLM evaluation
per query (prompts of a few hundred to about 2,000 tokens) plus one blank prompt per task kind and
candidate count. On a 24-core desktop CPU (i9-13900, CPU only) with the default model, the default 3,000 queries take
about 25 minutes. The database is opened read-only; `pjev calibrate` also confirms it is
byte-identical after the run.

### Database setup

`examplar.db` (schema version 4) holds 270,573 tasks with binary embedding vectors and LZ4-compressed
payloads. It also contains validation, test and external records, so pjev restricts demonstrations
and voters to the IDs in `examplar-allowlist.txt`: the 111,596 training-split records with a one-hot
answer. The database is opened read-only and never modified. SQLite keeps `-shm` / `-wal` side files
next to it, so its directory must be writable.

Rebuilding both from the datasets (only needed for a new dataset or embedding model):

```bash
cmake --build build --config Release --target pjev-filldb
./bin/pjev-filldb --input datasets/train.jsonl --output models/examplar.db      # repeat per dataset
./bin/pjev-filldb --allowlist --train datasets/train.jsonl --db models/examplar.db \
                  --output models/examplar-allowlist.txt
```

The datasets are stored as `datasets/*.zip` (Git LFS). `tools/convert_openjev140k.py` converts the
OpenJevData-140k corpus to the input format.

---

## CLI

```bash
pjev noul   --question TEXT [--state TEXT | --state-file FILE] [options]
pjev choice --question TEXT --option KEY[:DESC] --option KEY[:DESC] ... [options]
pjev score  --question TEXT --level DESC --level DESC ... [options]
```

| Option | Description |
|---|---|
| `--question TEXT` | Question to decide (required) |
| `--state TEXT` / `--state-file FILE` | The state the question is about (`-` reads stdin). A state that is a JSON object is also matched field by field against the exemplar bank. |
| `--option KEY[:DESC]` | `choice` candidate; `KEY` is returned, `DESC` is shown to the model |
| `--level DESC` | `score` level, lowest first; levels are keyed `0`, `1`, … |
| `--domain TEXT`, `--family TEXT` | Optional task domain / family, used only for exemplar retrieval |
| `--model`, `--embedding-model`, `--exemplar-db`, `--exemplar-allowlist` | Resource paths |
| `--threads N`, `--ctx-size N` | CPU threads (default: physical cores), context size (default 4096) |
| `--n-gpu-layers N`, `--main-gpu N` | Offload inference-model layers to a GPU (default 0) |
| `--json` | JSON output |
| `--direct` | Load the models in this process instead of using the background worker |
| `--idle-timeout SECS` | Worker idle timeout (default 600) |
| `--verbose` | Verbose logging |

Output:

| Command | Plain | `--json` |
|---|---|---|
| `noul` | `true` / `false` | `{"primitive":"noul","probabilities":[p_true,p_false],"result":{"p_true":…,"value":…}}` |
| `choice` | selected key | `{"primitive":"choice","probabilities":[…],"keys":[…],"result":{"index":…,"key":…}}` |
| `score` | expected level, e.g. `2.1874` | `{"primitive":"score","probabilities":[…],"keys":["0",…],"levels":[…],"result":{"expected_score":…}}` |

For `noul` and `choice`, probabilities are the softmax of the candidate logits. For `score` they are
the normalized kNN vote weights, and the expected level is their mean; the most likely level is the
kNN prediction.

Example:

```bash
pjev score --state-file essay.json --question "How many of the four requirements are met?" \
  --level "None" --level "One" --level "Two" --level "Three" --level "All four" --json
```

Flags of earlier releases that configured the decision (`--ensemble`, `--layout`, `--scheme`,
`--no-kv-reuse`, `--no-batch-questions`) are accepted and ignored with a warning. `--calibration PATH`
selects the calibration artifact again (see [Calibration](#calibration)).

---

## HTTP server

```bash
pjev serve --port 8080
```

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | `{"status":"ok","state":"ready","model":…}` once the runtime is loaded |
| `GET` | `/version` | pjev version, commit, llama.cpp revision, model, `routing_policy`, `calibration` (`enabled` / `incompatible` / `missing` / `disabled`) |
| `POST` | `/v1/systemone` | Jev-compatible decision API |

Request:

```json
{
  "state": "The order arrived damaged; photos were provided within 28 days.",
  "questions": {
    "refund": {"type": "noul", "instructions": "Is the customer eligible for a refund?",
               "criteria": {"false": "", "true": ""}, "domain": "support", "family": "policy"},
    "queue":  {"type": "choice", "instructions": "Which queue handles this?",
               "criteria": {"billing": "Billing", "returns": "Returns"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["Not urgent", "Somewhat urgent", "Very urgent"]}
  }
}
```

`domain` and `family` are optional. Questions are decided one after another. Send the body with
`Content-Type: application/json` (the HTTP library limits form-encoded bodies to 8 KiB).

Response:

```json
{
  "answers": {
    "refund":  {"type": "noul", "noul": 0.93, "duration": "412"},
    "queue":   {"type": "choice", "probabilities": {"billing": 0.12, "returns": 0.88}, "choice": "returns", "duration": "655"},
    "urgency": {"type": "score", "probabilities": {"0": 0.0, "1": 0.27, "2": 0.73}, "duration": "131"}
  },
  "model": "pseudojev",
  "usage": {},
  "request_id": "req-0"
}
```

`duration` is the decision time in milliseconds (retrieval included).

Server options: `--port`, `--host` (default `127.0.0.1`), `--model-name`, `--max-queued` (default 8),
`--request-limit` (bytes, default 1 MiB), `--read-timeout` (seconds, default 30), `--threads-batch`,
plus the resource and runtime options of the CLI. Inference is serialized; concurrent requests queue.

---

## Errors

Errors are returned as `{"error":{"code":…,"message":…},"request_id":…}` (HTTP) or printed to stderr
with exit code 1 (CLI; exit code 2 is a usage error).

| Code | HTTP | When |
|---|---|---|
| `invalid_json` | 400 | Request body is not JSON |
| `invalid_request` | 400 | Missing / invalid fields, limits exceeded, invalid candidates |
| `context_too_long` | 400 | The task does not fit the inference context (4096 tokens) or the embedding model's context (2048 tokens) |
| `unsupported_primitive` | 422 | Unknown question type |
| `no_eligible_exemplar` | 422 | Reserved; no longer returned (tasks without enough exemplars are decided zero-shot) |
| `inference_error` | 500 | Embedding, database or inference-backend failure |
| `server_busy` | 503 | More than `--max-queued` requests in flight |
| `request_too_large` / `not_found` / `method_not_allowed` | 413 / 404 / 405 | HTTP-layer errors (no `request_id`) |

### Fallback behaviour

| Task kind | Strategy | Valid exemplars required | With fewer |
|---|---|---:|---|
| `choice` | Nearest 2-shot, order A | 2 | S1 zero-shot |
| `noul` | Nearest 1-shot | 1 | S1 zero-shot |
| `score` | Weighted kNN-5 | 5 | S1 zero-shot |

A strategy is never run with fewer exemplars (no 1-shot `choice`, no kNN with k < 5), and missing
exemplars are never filled with arbitrary ones. An exemplar counts only if it is in the database and
the training allowlist, has a one-hot answer, has the task's kind (and for `score` its number of
levels), is not the query itself or a duplicate of another selected exemplar, loads and decodes, and
(for `score`) its answer maps onto the query's levels. Invalid exemplars are skipped in rank order
within the top-10 search. The zero-shot fallback is the original S1 path (same prompt, candidate
mapping and candidate-logit scoring); its result has the same response format as any other decision.

One fallback from R4 is kept: for `choice` with two valid exemplars, if the 2-shot prompt does not fit
the context, the decision uses the nearest exemplar alone (nearest 1-shot).

Fallback applies only to missing or invalid exemplars. Embedding, database and inference-backend
failures, invalid requests and context overflows keep their error codes; a failing zero-shot decision
reports through the same codes. The requested and executed strategy, the valid / required exemplar
counts and the fallback reason are kept in the decision's routing diagnostics (debug log,
`pjev benchmark latency`), not in API or CLI output.

Missing or incompatible resources stop the server, the worker or the CLI at startup with a message
naming the resource.

---

## Background worker

Decision commands start a background worker on first use; it keeps the models and the database loaded
and exits after `--idle-timeout` seconds without requests. A running worker with a different
configuration (resources, threads, context size) is replaced automatically.

```bash
pjev status [--json]   # worker state
pjev stop              # stop it and release memory
```

Use `--direct` to run in-process (each invocation loads the models, a few seconds).

---

## Configuration

`pjev.json` next to the executable (all keys optional; unknown keys are ignored):

```json
{
  "schema_version": 1,
  "default_model": "model.gguf",
  "embedding_model": "nomic-embed-text-v1.5.gguf",
  "exemplar_db": "examplar.db",
  "exemplar_allowlist": "examplar-allowlist.txt",
  "calibration": "calibration.json",
  "threads": -1,
  "context_size": 4096
}
```

Precedence: command-line flag > `pjev.json` > file next to the executable > `models/` (development).
`threads: -1` uses the number of physical cores. Keep `context_size` at 4096 — the validated value.

---

## Diagnostics

```bash
pjev diagnostics [--load] [--json]
```

Reports version, build, platform, the resolved resource paths (and whether each exists or is an
unfetched Git LFS pointer), routing policy, threads, context size and worker state. `--load` also
loads the models and the database and reports the model hash, embedding fingerprint and exemplar
counts. The report contains no prompt content.

---

## Building from source

Requirements: CMake 4.0+, a C++20 compiler (MSVC 2022, GCC 11+, Clang 14+), Git with Git LFS, and the
Vulkan SDK (llama.cpp is built with `GGML_VULKAN=ON`; the embedding model runs on the GPU when one is
available).

```bash
git clone --recursive https://github.com/taqu/pseudojev.git
cd pseudojev && git lfs pull
cmake -S . -B build
cmake --build build --config Release
```

Executables are written to `bin/`. Build options: `-DPJEV_BUILD_TOOLS=OFF` skips `pjev-filldb`;
`-DBUILD_TESTING=OFF` skips the tests. On Windows, keep the build directory outside `%TEMP%` (the
llama.cpp Vulkan shader generator fails to configure there).

## Testing

```bash
ctest --test-dir build -C Release --output-on-failure
```

- Unit tests (no model files): prompt construction, decision engine, task routing (dispatch, kNN
  voting, label mapping, fallbacks, errors), exemplar store (allowlist, search, LZ4, corrupt input),
  calibration (loading, fitting, metrics, partitioning, atomic output), API, CLI formatting, worker
  protocol, configuration.
- `test_calibrate`: `pjev calibrate` generation end to end on a fixture database with a mock model
  (partitioning and leakage, inference paths, determinism, loader and model identity, read-only
  database, atomic replacement, insufficient / degenerate data, validation rejection).
- `test_golden_regression` (label `integration`): replays 24 tasks recorded from the validated R4 run
  (`tests/data/golden_r4.jsonl`) and requires identical exemplars, prompt token counts, candidate
  logits, vote shares and predictions. It needs the four runtime resources and is reported as skipped
  without them. Run only the unit tests with `ctest … -LE integration`.

Latency on your hardware: `pjev benchmark latency [--input tests/data/golden_r4.jsonl] [--iterations N]`.

## Packaging

```bash
cd build && cpack -C Release
```

Produces `pjev-<version>-<platform>-<arch>.{zip|tar.gz}` with the `pjev` executable, `pjev.json`,
`README.md`, `LICENSE` and `THIRD-PARTY-NOTICES.md`. Models and the database are not included; see
[Runtime resources](#runtime-resources).

---

## Architecture

```
pjev noul / choice / score ──(local IPC)──► background worker ─┐
pjev serve ──(HTTP /v1/systemone)──► JevApiHandler ────────────┤
pjev … --direct ───────────────────────────────────────────────┤
                                                               ▼
                                              TaskRouter (DecisionService)
                         ┌──────────────────────┬──────────────┴───────────────┐
                    choice: 2-shot        noul: 1-shot                  score: kNN-5
                         └──────────┬───────────┘                              │
                         DecisionEngine (prompt, candidate logits)             │
                                    │                                          │
                                llama.cpp ◄── inference model                  │
                                                                               │
                  shared retrieval: Embedding (nomic) ─► ExemplarStore (SQLite, sqlite-vec, LZ4, allowlist)
```

Source layout: `src/routing` (policy, task mapping, kNN, router, runtime), `src/retrieval` (exemplar
store, allowlist, embedding text), `src/decision` (prompt strategy, decision engine), `src/calibration`
(loading, fitting, metrics, `pjev calibrate` generation), `src/server`,
`src/worker`, `src/cli`, `src/distribution`; `tools/pjevfilldb` builds the database.

---

## Known limitations

- Validated on English tasks from the Open-Jev distribution; behaviour on other languages or task
  families not represented in the exemplar bank is not measured.
- `score` uses kNN only for the level counts present in the bank (3, 4 and 5 levels); other scales
  are decided zero-shot, whose accuracy was lower in R4 (44.69% overall for zero-shot).
- Decisions run one at a time; each LLM decision evaluates its prompt from an empty KV cache.
- The first decision after start-up is slower (cold caches); see `pjev benchmark latency`.
- Binaries are unsigned; macOS Gatekeeper / Windows SmartScreen may warn.

## License

pjev is [MIT licensed](LICENSE). Bundled components (llama.cpp, cpp-httplib, nlohmann/json, spdlog and
{fmt}, SQLite, sqlite-vec, LZ4, xxHash, MurmurHash3) are listed with their licenses in
[release/THIRD-PARTY-NOTICES.md](release/THIRD-PARTY-NOTICES.md). Models and data have their own
licenses (see [Runtime resources](#runtime-resources)).

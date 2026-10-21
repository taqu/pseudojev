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
| Inference model (Ternary-Bonsai-2-2B, GGUF) | `model.gguf` | `--model` | `default_model` |
| Embedding model (nomic-embed-text-v1.5, GGUF) | `nomic-embed-text-v1.5.gguf` | `--embedding-model` | `embedding_model` |
| Exemplar database (SQLite) | `examplar.db` | `--exemplar-db` | `exemplar_db` |
| Exemplar allowlist (text) | `examplar-allowlist.txt` | `--exemplar-allowlist` | `exemplar_allowlist` |

None of them is included in the release archive. They are stored in this repository under
`models/` with Git LFS:

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

Flags of earlier releases that configured the decision (`--calibration`, `--ensemble`, `--layout`,
`--scheme`, `--no-kv-reuse`, `--no-batch-questions`) are accepted and ignored with a warning.

---

## HTTP server

```bash
pjev serve --port 8080
```

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | `{"status":"ok","state":"ready","model":…}` once the runtime is loaded |
| `GET` | `/version` | pjev version, commit, llama.cpp revision, model, `routing_policy` |
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

`domain` and `family` are optional. Questions are decided one after another.

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
| `no_eligible_exemplar` | 422 | The exemplar bank has no eligible exemplar for the task — for `score`, no training exemplar with that number of levels (the bank has 3-, 4- and 5-level scales) |
| `inference_error` | 500 | Embedding, database, payload or inference-backend failure |
| `server_busy` | 503 | More than `--max-queued` requests in flight |

Fallback behaviour: for `choice`, if fewer than two eligible exemplars exist or the 2-shot prompt does
not fit the context, the decision uses the nearest exemplar alone (nearest 1-shot), as validated in R4.
There are no other fallbacks: pjev never substitutes a different strategy or an arbitrary answer.
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
  API, CLI formatting, worker protocol, configuration.
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
store, allowlist, embedding text), `src/decision` (prompt strategy, decision engine), `src/server`,
`src/worker`, `src/cli`, `src/distribution`; `tools/pjevfilldb` builds the database.

---

## Known limitations

- Validated on English tasks from the Open-Jev distribution; behaviour on other languages or task
  families not represented in the exemplar bank is not measured.
- `score` supports the level counts present in the bank (3, 4 and 5 levels); other scales return
  `no_eligible_exemplar`.
- Decisions run one at a time; each LLM decision evaluates its prompt from an empty KV cache.
- The first decision after start-up is slower (cold caches); see `pjev benchmark latency`.
- Binaries are unsigned; macOS Gatekeeper / Windows SmartScreen may warn.

## License

pjev is [MIT licensed](LICENSE). Bundled components (llama.cpp, cpp-httplib, nlohmann/json, spdlog and
{fmt}, SQLite, sqlite-vec, LZ4, xxHash, MurmurHash3) are listed with their licenses in
[release/THIRD-PARTY-NOTICES.md](release/THIRD-PARTY-NOTICES.md). Models and data have their own
licenses (see [Runtime resources](#runtime-resources)).

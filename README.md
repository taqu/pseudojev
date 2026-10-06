# pjev

[![CI](https://github.com/taqu/pseudojev/actions/workflows/ci.yml/badge.svg)](https://github.com/taqu/pseudojev/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**pjev** is a small, local, Jev-compatible decision engine powered by [llama.cpp](https://github.com/ggerganov/llama.cpp) and a GGUF model.

It runs entirely on your CPU — no GPU, no network, no cloud required.

> **Status:** Experimental. Validated on Windows x86_64, Linux x86_64, and macOS arm64.
> API and output formats may change before 1.0.

---

## Features

- **Three decision primitives**: `noul` (true/false), `choice` (multiple-choice), `score` (quality score)
- **Jev-compatible HTTP server** — drop-in replacement for Jev API consumers
- **Direct CLI** — run decisions as shell commands, pipe results, script with `--json`
- **Background worker** — model loads once, stays resident, exits automatically after idle timeout
- **Calibrated probabilities** — temperature scaling for well-calibrated confidence
- **English** validated
- **Native binaries** for Windows, Linux, and macOS

---

## Quick Start

### Binary release

1. Download the archive for your platform from [Releases](https://github.com/taqu/pseudojev/releases)
2. Unpack and place a compatible GGUF model next to the executable (see [Model Setup](#model-setup))
3. Run:

```bash
./pjev --version
./pjev noul --state "The sky is blue." --question "Is it daytime?"
```

Expected output: `true`

### One-liner check

```bash
./pjev diagnostics
```

---

## Model Setup

pjev requires a GGUF-format language model. It is **not** bundled in the release archive.

**Default lookup order:**

1. `--model /path/to/model.gguf` (explicit flag)
2. `default_model` in `pjev.json` adjacent to the executable
3. `model.gguf` adjacent to the executable ← simplest setup
4. `models/bonsai.gguf` (development fallback)

**Recommended setup:** place your GGUF file as `model.gguf` in the same directory as `pjev`.

```
pjev-0.8.0-linux-x86_64/
├── pjev
└── model.gguf      ← place your model here
```

**Obtaining a model:** Any Qwen2.5-family or compatible instruction-tuned GGUF model works.
The project was validated with Bonsai-1.7B-Q4. Download from Hugging Face or another GGUF source.

**Model integrity:** `pjev diagnostics` reports whether the model file is accessible and shows its identity hash.

---

## CLI Reference

All decision commands default to using the background worker (see [Background Worker](#background-worker)).
Add `--direct` to run in-process without a worker.

### `pjev noul` — true/false question

```bash
pjev noul \
  --state "The patient has a fever of 39°C." \
  --question "Does this patient need immediate attention?"
```

Output: `true`

```bash
# Machine-readable
pjev noul --state "..." --question "..." --json
```

```json
{"primitive":"noul","probabilities":[0.08,0.92],"result":{"p_true":0.92,"value":true}}
```

### `pjev choice` — multiple-choice question

```bash
pjev choice \
  --state "The user prefers concise replies." \
  --question "What tone should the response use?" \
  --option formal:"Formal and detailed" \
  --option casual:"Casual and brief" \
  --option technical:"Technical and precise"
```

Output: `casual`

```bash
# With JSON output
pjev choice --question "Which direction?" --option left --option right --json
```

```json
{"keys":["left","right"],"primitive":"choice","probabilities":[0.71,0.29],"result":{"index":0,"key":"left"}}
```

### `pjev score` — quality score

```bash
pjev score \
  --state "The essay introduces the topic well but lacks supporting evidence." \
  --question "Rate the overall quality of this essay." \
  --level "Poor" \
  --level "Below average" \
  --level "Average" \
  --level "Good" \
  --level "Excellent"
```

Output: `2.1234`  (expected value across levels 0–4)

### Common options

| Option | Description |
|--------|-------------|
| `--state TEXT` | Context state (default: empty) |
| `--state-file FILE` | Read state from file (`-` for stdin) |
| `--question TEXT` | Question to evaluate (required) |
| `--model PATH` | Path to GGUF model |
| `--calibration FILE` | Calibration artifact JSON |
| `--threads N` | CPU threads (default: 8) |
| `--ctx-size N` | Context size (default: 4096) |
| `--json` | Output JSON to stdout |
| `--direct` | Bypass background worker, load model in-process |
| `--verbose` | Enable verbose model logging to stderr |

### Japanese example

```bash
pjev noul \
  --state "空は青い。" \
  --question "晴れていますか?"
```

---

## HTTP Server

```bash
pjev serve --model model.gguf --port 8080
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Server health and model identity |
| `GET`  | `/version` | pjev version and build info |
| `POST` | `/api/v1/systemone/` | Jev-compatible decision API |

### Minimal request example

```bash
curl -s http://localhost:8080/health

curl -s -X POST http://localhost:8080/api/v1/systemone/ \
  -H "Content-Type: application/json" \
  -d '{
    "state": "The sky is blue.",
    "questions": {
      "q1": {
        "type": "noul",
        "instructions": "Is it daytime?",
        "criteria": { "false": "", "true": "" }
      }
    }
  }'
```

### Server options

```
pjev serve --help
```

Key options: `--model`, `--port`, `--host`, `--threads`, `--ctx-size`, `--calibration`, `--max-queued`

---

## Background Worker

Decision commands (`noul`, `choice`, `score`) automatically start a **background worker** the first time they run. The worker keeps the model loaded in memory so subsequent CLI calls are fast.

```
first call:  pjev choice ...   → starts worker, loads model (~seconds), runs decision
next calls:  pjev choice ...   → reuses worker (~milliseconds overhead)
after idle:  worker exits automatically, model memory released
```

### Worker commands

```bash
pjev status          # show worker state
pjev status --json   # machine-readable status
pjev stop            # cleanly stop the worker
```

### Bypass the worker

```bash
pjev choice ... --direct   # load model in-process, no worker
```

Use `--direct` for scripting, testing, or when you want to control model lifetime precisely.

### Idle timeout

The worker exits after 5 minutes of inactivity by default. Subsequent commands restart it automatically.

---

## Diagnostics

```bash
pjev diagnostics          # human-readable install health check
pjev diagnostics --json   # JSON output (useful for bug reports)
```

Sample output:

```
pjev version:    0.8.0 (commit: abc1234)
llama.cpp:       88c4bc60b
platform:        linux-x86_64
cpu logical cores: 8
model path:      /opt/pjev/model.gguf (found)
threads:         8
context size:    4096
worker:          not running
```

---

## Configuration

### `pjev.json` (optional)

Place `pjev.json` adjacent to the executable to configure defaults:

```json
{
  "schema_version": 1,
  "default_model": "model.gguf",
  "threads": 8,
  "context_size": 4096
}
```

### Precedence

`--flag` on command line > `pjev.json` > adjacent `model.gguf` > development fallback

---

## Building from Source

**Requirements:**

- CMake 3.14+
- C++17 compiler: MSVC 2022, GCC 11+, or Clang 14+
- Git (for submodules)

```bash
git clone --recursive https://github.com/taqu/pseudojev.git
cd pseudojev
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

The `pjev` executable is placed in `bin/`.

### Running tests

Unit tests do not require a model:

```bash
ctest --test-dir build -C Release --output-on-failure
```

All 14 test suites cover: prompt strategy, decision engine, calibration, server API, CLI formatting, IPC protocol, and more.

### Packaging (local release archive)

```bash
cd build
cpack -C Release
```

Produces `pjev-<version>-<platform>-<arch>.{zip|tar.gz}`.

---

## Architecture

```
pjev noul / choice / score
          │
          ▼ (local IPC — named pipe or Unix socket)
  background worker process
          │
          ▼
    DecisionEngine
          │
          ▼
  PromptStrategy + Calibration
          │
          ▼
      llama.cpp
          │
          ▼
       GGUF model

pjev serve  →  HTTP  →  DecisionEngine  (same engine, separate process)
```

CLI and HTTP server share the same `DecisionEngine` — no decision logic is duplicated.

---

## Troubleshooting

**Model not found**

```
action: model not found at 'model.gguf'
        use --model PATH or place model.gguf next to the executable
```

Place your GGUF file as `model.gguf` next to `pjev`, or pass `--model /path/to/model.gguf`.

**Worker appears stale**

```bash
pjev stop
pjev status   # should show: not running
pjev noul ... # auto-restarts worker
```

**Port already in use**

```bash
pjev serve --port 8081   # use a different port
```

**Slow first CLI invocation**

This is expected — the first invocation loads the model. Subsequent calls reuse the background worker and are much faster. See `pjev status` to check if the worker is running.

**Getting a full diagnostic report for a bug report**

```bash
pjev diagnostics --json > pjev-diag.json
```

Attach `pjev-diag.json` to the GitHub issue. It does not contain any prompt content or user data.

---

## Performance

Model load time and decision latency depend on the model size and CPU.

For Bonsai-1.7B-Q4 on a modern CPU:

- **Model load (first CLI call):** ~1–5 seconds
- **Decision latency (warm worker):** ~50–500ms depending on state/question length
- **IPC overhead:** negligible (<5ms)

Use `pjev benchmark latency --model model.gguf` for reproducible measurements on your hardware.

---

## Known Limitations

- **CPU-only.** GPU inference is not supported in current releases.
- **Every direct invocation loads the model** — use the background worker (`--direct` bypasses it).
- **Background worker is per-user and single-instance per configuration.** Changing `--model` or other core settings restarts the worker.
- **Inference is serialized.** Concurrent requests queue behind a single model context.
- **Binaries are unsigned.** macOS may require Gatekeeper bypass; Windows may show SmartScreen warnings.
- **API is pre-1.0** and subject to change.

---

## License

pjev is [MIT licensed](LICENSE).

**Third-party components:**

- [llama.cpp](https://github.com/ggerganov/llama.cpp) — MIT license
- [nlohmann/json](https://github.com/nlohmann/json) — MIT license
- [cpp-httplib](https://github.com/yhirose/cpp-httplib) — MIT license
- [spdlog](https://github.com/gabime/spdlog) — MIT license

**Model license:** The GGUF model you use has its own license terms. Consult the model's upstream repository before redistribution.

# Phase 0 — experimental decision core

A single C++ program (`phase0.cpp`) that runs Bonsai 1.7B through llama.cpp and turns
the logits of a few single-token candidates into `noul` / `choice` / `score` decisions.
This is throwaway experiment code. See [`REPORT.md`](REPORT.md) for the findings.

## Build

Requirements: cmake, a C++17 compiler, OpenMP. No Go or Python needed.

```bash
cd phase0
make llama     # configures and builds ../llama.cpp/build (static, CPU only)
make           # builds ./phase0
./phase0 --selftest   # model-free checks of softmax / expected score
```

The model is expected at `../models/Bonsai-1.7B.gguf`. Use the PrismML llama.cpp fork
(the submodule), because it provides the `Q1_0` type.

## Run

```bash
# smoke tests (English + Japanese) with expectation checks and a 3x determinism check
./phase0 -m ../models/Bonsai-1.7B.gguf -t 16 --check --repeat 3 cases/smoke.jsonl

# JevBench rows can be passed in unchanged
./phase0 -m ../models/Bonsai-1.7B.gguf -t 16 ../jevbench/datasets/public/original.jsonl

# full debug output for one case
echo '{"type":"choice","state":"The customer was billed twice.","question":"Which queue?",
       "choices":{"billing":"Invoices, charges, refunds","technical":"Product failures","sales":"Pre-purchase questions"}}' \
  | tr -d '\n' | ./phase0 -m ../models/Bonsai-1.7B.gguf -v -
```

The program writes one JSON result per case to stdout. Debug output and the summary go to stderr.
The exit code is non-zero on errors, failed `--check` expectations, or non-deterministic repeats.

Options:

| option | meaning |
|---|---|
| `-v` | print the prompt, candidate → token map, raw logits, probabilities, and timings |
| `--check` | evaluate each case's `check` field (`selected`, or score `group`/`rank` ordering) |
| `--repeat N` | re-run each case N times and require bit-identical candidate logits |
| `--per-type N` | keep only the first N cases of each decision type |
| `--no-gen` | skip the constrained one-token generation check |
| `--format` | `nothink` (default, Qwen3 chat with an empty `<think>` block), `chatml`, `raw` |
| `--scheme` | `natural` (default) or `letters`; sets the internal candidate tokens, see below |
| `--layout` | `auto` (default), `state-first`, `state-last`; sets the prompt section order |

## Input

One JSON object per line (or a JSON array, or a single object). Two shapes are accepted:

```jsonc
{"type":"noul",  "state":"...", "question":"..."}
{"type":"choice","state":"...", "question":"...", "choices":{"key":"description", ...}}
{"type":"score", "state":"...", "question":"...", "levels":["lowest", ..., "highest"]}
```

Raw JevBench rows (`question.type`, `question.criteria`, `labels`, `expected`) also work.
When `expected` is present, the output includes `correct` and the summary reports accuracy.

Limit: at most **10** candidates (choices or levels) per decision.

## How a decision is made

1. **Internal candidates.** User labels are never output tokens. Each candidate gets an internal token:
   - `natural` scheme: noul uses `No`/`Yes`, score uses `0`..`N-1`, and choice uses `A`..`J`.
   - `letters` scheme: every type uses `A`..`J`.

   At startup, every internal text is tokenized and must map to exactly one distinct token ID.
   The map is printed to stderr. Before each decision, `prompt + candidate` is also re-tokenized
   to confirm that the candidate does not merge with the end of the prompt.
2. **Prompt.** The prompt contains the state, the question, and `X: description` for each candidate,
   followed by "Reply with exactly one of …". It is wrapped in the Qwen3 chat template, and the
   assistant turn is prefilled with an empty think block.
3. **Logits.** One `llama_decode` call evaluates the prompt, with logits requested only for the
   last position. The logits of the candidate token IDs are read from that position.
4. **Restricted softmax.** A max-subtracted softmax is taken over only the candidate logits.
   Results with NaN or infinite values, an empty candidate set, duplicate IDs, or a sum ≠ 1
   are rejected.
5. **Results.**
   - noul: `value = p_true`.
   - choice: the argmax.
   - score: `score = Σ index·p` and `selected_level = argmax`.
6. **Validation (not used for the result).** A `logit_bias` sampler sets every non-candidate token
   to −∞, and a greedy sampler then generates exactly one token. `gen_matches` reports whether it
   equals the argmax.

## Reading the output

```jsonc
{"id":"choice-en","type":"choice",
 "probabilities":{"billing":0.996,"technical":0.0004,"sales":0.0035},
 "selected":"billing",
 "raw_logits":{"A":23.16,"B":15.34,"C":17.50},
 "candidate_mass":0.999,       // share of the FULL-vocab softmax on the candidates (diagnostic only)
 "free_argmax":"A",            // unconstrained top token (diagnostic only)
 "generated":"A","gen_matches":true,
 "n_prompt_tokens":81,"t_prompt_ms":252.7,"t_decision_ms":0.01,"t_gen_ms":3.8,"t_total_ms":265.0}
```

If `candidate_mass` is far below 1, the model wanted to say something other than a candidate
at that position, which usually points to a prompt or template problem.

## Files

- `cases/smoke.jsonl`: noul, choice, and score smoke cases in English and Japanese, with expectations
- `results/`: outputs of the runs cited in `REPORT.md`

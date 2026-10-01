# Phase 0 report — constrained-logit decisions with Bonsai 1.7B

Date: 2026-10-01. Model: `Bonsai-1.7B.gguf` (qwen3 1.7B, `Q1_0`, 237 MiB), via the PrismML llama.cpp fork
at `88c4bc60b`, built as static CPU-only libraries. Hardware: 32-thread x86 under WSL2; runs used `-t 16`.

## Implementation

**Candidate tokens.** Semantic labels are never output tokens. Each candidate gets an internal token.
The default `natural` scheme uses `No`/`Yes` (noul), `0`..`9` (score), and `A`..`J` (choice), so at most
10 candidates are supported. At startup each internal text is tokenized
(`add_special=false, parse_special=false`) and must yield exactly one token, with no two texts sharing
an ID. The map is logged:
`A→32 … J→41`, `0→15 … 9→24`, `No→2753`, `Yes→9454`. Before each decision, `prompt + candidate` is
re-tokenized and must equal `prompt tokens + [candidate id]`, which guards against merges at the boundary.

**Logits.** The prompt is wrapped in the Qwen3 chat template with an empty `<think>\n\n</think>\n\n`
prefilled, then evaluated with one `llama_decode`. The batch requests logits only for the last position.
The candidate IDs are read from `llama_get_logits_ith(ctx, -1)`. The KV cache is cleared before each case.

**Restriction and probabilities.** The program takes a max-subtracted softmax over only the candidate
logits, in double precision. It validates that the input is non-empty, all values are finite,
there are no duplicate IDs, each `p ∈ [0,1]`, and `|Σp − 1| < 1e-9`. noul exposes `p_true`.
choice returns the argmax. score returns `Σ i·pᵢ` together with the argmax level.

**Constrained one-token generation.** A sampler chain of `llama_sampler_init_logit_bias` (−∞ on every
non-candidate token, ~151k entries) followed by `llama_sampler_init_greedy` samples exactly one token
at the same position. It is used only to confirm that the result equals the argmax.

## Results

### Mechanics (exit criteria)

| criterion | result |
|---|---|
| all three primitives work | yes; 0 errors over 924 decisions on the 4 configs × 231 public JevBench rows |
| candidates verified single-token | yes, at startup and per prompt at the boundary |
| `0 ≤ p ≤ 1`, `Σp ≈ 1`, no NaN or inf | yes, enforced on every decision; `--selftest` covers the numeric edge cases |
| constrained 1-token generation = argmax | 72/72 (JevBench original) and 10/10 (smoke) |
| English and Japanese | both run end to end; tokenization and extraction are unaffected by UTF-8 input |
| reproducibility | candidate logits are bit-identical over 3 repeats, and identical at 4, 16, and 32 threads |

### Smoke tests (`cases/smoke.jsonl`, default config): 9/10 checks pass

| case | result |
|---|---|
| noul positive (en) "Please refund the duplicate charge." | p_true **0.876** ✓ |
| noul negative (en) "Thanks, everything is working now." | p_true **0.001** ✓ |
| noul positive (ja) | p_true **0.489** ✗ (borderline) |
| noul negative (ja) | p_true **0.121** ✓ |
| choice (en) billed twice → billing | p = **0.996** ✓ |
| choice (ja) API 500 error → technical | p = **0.994** ✓ |
| score frustration (en), low vs high | **1.34 < 2.40** ✓ |
| score frustration (ja), low vs high | **1.92 < 1.93** ✓ (barely; nearly undifferentiated) |

### JevBench public splits (231 rows; accuracy = argmax matches `expected`)

The decision types are independent, so the per-type numbers below come straight from the 4-config grid
in `results/all.*`.

| config | choice (n=139) | noul (n=74) | score (n=18) acc / MAE |
|---|---|---|---|
| state-first, letters | 0.554 | 0.622 | 0.111 / 1.21 |
| state-first, natural | **0.554** | **0.649** | 0.167 / 1.32 |
| state-last, letters | 0.496 | 0.595 | 0.278 / 0.77 |
| state-last, natural | 0.496 | 0.622 | **0.444 / 0.78** |
| *reference baseline* | random ≈ 0.23 | majority 0.527 | majority 0.333 |

The default (`auto` layout, `natural` scheme) gives choice 0.554, noul 0.649, and score 0.444 (MAE 0.78).

Results by split (state-first, natural):

| split | choice | noul |
|---|---|---|
| easy | **36/36** | **11/12** |
| original | 17/36 | 13/24 |
| hard | 0.36 | 0.63 |

The mechanism produces correct, confident decisions when the semantic question is easy. Accuracy
drops on rubric-heavy items.

Default-config run on the `original` split (`results/jevbench-original.jsonl`, with expected,
predicted, probabilities, and latency per row): choice 17/36, noul 13/24, score 7/12 (MAE 0.75).

### Timing and memory

| measure | value |
|---|---|
| model load (file → context ready) | ~440 ms |
| process RSS | ~940 MiB (237 MiB weights + 304 MiB compute buffer at n_ctx 4096) |
| prompt evaluation | ~2.5–3.4 ms per prompt token; a typical 60–140-token prompt takes 140–420 ms |
| long prompts (hard split, up to 3.2k tokens) | up to 11 s |
| decision (logit read + softmax) | ~0.01 ms |
| constrained one-token generation | ~2–4 ms (mostly building the 151k-entry bias list) |
| mean total per case | 0.23 s (smoke), 0.31 s (original), 1.7 s (all public, long-tailed) |

None of this has been optimized; prompt evaluation is the whole cost.

## Problems found

- **Recency and position bias (main quality issue).** With the state first and the options last,
  score collapsed onto the last level: 0/12 → "D", including "The icon is misaligned. Every function
  works." → 97% "irreversible data loss". Choice favoured the last letter (E: 12, D: 10, A: 5 of 36).
  Reversing option order flips rubric answers ("cosmetic" 19% → 5%), but not easy knowledge answers
  (Paris 97% / 90%). In free-text generation the same model answers "Cosmetic issue." correctly, so the
  loss happens in binding letters to meanings.
  Putting the state *last* fixes most of the score collapse (MAE 1.21 → 0.78) but costs about 6 points on
  choice. Hence the per-type `auto` layout.
- **Polarity prior in noul.** The polarity prior depends on the candidates:
  - with letter candidates, the model predicts "true" for 53–61 of 74 rows;
  - with `No`/`Yes`, it predicts "true" for only 11–13 of 74;
  - 35 rows are actually true.

  The probabilities are therefore poorly calibrated for noul. This should respond well to the later
  calibration phase, for example a content-free prior correction on the candidate logits.
- **Whitespace-sensitive tokens.** In a raw prompt ending `"Answer: "`, the trailing space merges with
  the answer (`" A"` is a single token, 362), so the boundary check rejects it. Raw mode therefore ends with
  `"Answer:"` and scores space-prefixed tokens. The Qwen3 tokenizer **splits digits from the
  space**: `" 0"` is 2 tokens, so space-prefixed digit candidates are impossible and raw mode supports
  only letters. Chat formats put the answer straight after `\n`, so bare tokens work.
- **Thinking mode.** With the plain ChatML assistant prefix the model wants to emit `<think>`, and
  candidate mass falls to 1e-12 – 4e-7 (smoke cases). The empty think block is required.
- **BOS/EOS.** `add_bos=0` for this model, so no BOS is added. The loader warns that token 128247 `</s>` is
  not typed as control and overrides it; this has no effect on the decision path. `n_ctx_train` reads as
  0 from the GGUF, which produces a harmless warning.
- **Special-token parsing.** The whole prompt is tokenized with `parse_special=true`, so a user state
  containing `<|im_end|>` would be interpreted as a control token. This is acceptable for Phase 0, but
  Phase 1 must tokenize user text separately with special-token parsing disabled.
- **Logit access, logit bias, and numerics.** No issues. The −∞ bias plus greedy sampling always matched
  the argmax. Candidate logits are around 13–24, well within float range. Median candidate mass is 0.99;
  about 5% of rows fall below 0.86, and the minimum is 0.20, where the model wanted to start with "To".
- **Determinism.** No issues. With a single sequence, a cleared KV cache, and greedy sampling on the
  CPU backend, logits are bit-identical across repeats and thread counts.

## Recommendation

**Proceed to Phase 1.**

The constrained-logit decision primitive is technically sound:

- every candidate is a verified single token;
- logit extraction, the restricted softmax, and constrained one-token decoding agree exactly and deterministically;
- multilingual input passes through unchanged;
- one decision costs one prompt evaluation plus about 0.01 ms.

Where the semantic question is easy, the distributions are sharp and correct (JevBench easy: choice
36/36, noul 11/12). Overall the model is clearly above chance on choice (0.55 vs 0.23).

The real risk is **decision quality, not mechanism**:

- noul is only modestly above the majority baseline and carries a strong candidate-dependent polarity prior;
- score is order-sensitive;
- rubric-heavy (hard) items are weak.

These are the problems that the prompt-design and calibration phases exist to address. Because the output is a
full probability vector rather than generated text, prior correction and calibration can be applied
directly. Phase 1 should keep `--layout`/`--scheme` style switches, so that quality work can proceed
against the JevBench harness without touching the inference path.

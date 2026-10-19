# R1 — Retrieved 1-Shot Inference Evaluation

Corpus: **external:validation-heldout.jsonl**  
Run date: 2026-10-16T22:04:34

## Configuration

| Setting | Value |
|---|---|
| input_path | C:/Users/SAKAI_~1/AppData/Local/Temp/claude/D--MyDocuments-Projects-go-pseudojev/758f89a0-7d36-4a5e-bbef-67cbed2a16ea/scratchpad/validation-heldout.jsonl |
| db_path | models/examplar.db |
| train_path | datasets/train.jsonl |
| inference_model | models/bonsai2_2b.gguf |
| embedding_model | models/nomic-embed-text-v1.5.gguf |
| prompt_format_version | r1-demo-turn-v1 |
| scheme | letters |
| layout | auto |
| ensemble | none (single order) |
| calibration | none (temperature 1.0, no prior correction) |
| n_ctx | 4096 |
| top_k | 1 |
| kind_filter | all |
| limit | -1 |
| seed | null |
| leakage_policy | demonstrations restricted to IDs whose split is 'train' in train_path; the query's own ID is rejected; query text never includes the target |

## Coverage

- Evaluated queries: 5811
- Paired (S1 ok and R1 1-shot ok): **5467** (94.08%)
- S1 status: ok=5811
- Retrieval status: ok=5811
- R1 status: gate_rejected=344, ok=5467

Unpaired queries are excluded from every paired metric below; no zero-shot fallback is counted as a 1-shot result.

## Headline (paired)

| Metric | Value |
|---|---|
| Paired examples | 5467 |
| S1 accuracy | 46.22% (2527) |
| R1 accuracy | 57.38% (3137) |
| Difference | +11.16 pp |
| 95% CI (paired Wald) | [+9.75, +12.57] pp |
| Rescue / Harm / Net | 1110 / 500 / 610 |
| McNemar χ² (cc) p | 0.000000 |
| Exact binomial p | 0.000000 |

Check: (rescue − harm) / n = +11.16 pp = R1 acc − S1 acc.

## Accuracy by group

| Group | n | S1 acc | R1 acc | Diff (pp) | 95% CI (pp) | Rescue | Harm | Net | McNemar p |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| overall | 5467 | 46.22% (2527) | 57.38% (3137) | +11.16 | [+9.75, +12.57] | 1110 | 500 | 610 | 0.0000 |
| kind:choice | 3313 | 43.68% (1447) | 45.52% (1508) | +1.84 | [+0.17, +3.51] | 430 | 369 | 61 | 0.0338 |
| kind:noul | 1598 | 63.45% (1014) | 90.61% (1448) | +27.16 | [+24.39, +29.93] | 531 | 97 | 434 | 0.0000 |
| kind:score | 556 | 11.87% (66) | 32.55% (181) | +20.68 | [+16.24, +25.13] | 149 | 34 | 115 | 0.0000 |
| score_levels:3 | 36 | 66.67% (24) | 91.67% (33) | +25.00 | [+10.85, +39.15] | 9 | 0 | 9 | 0.0077 |
| score_levels:5 | 520 | 8.08% (42) | 28.46% (148) | +20.38 | [+15.73, +25.04] | 140 | 34 | 106 | 0.0000 |
| choice_options:3 | 2375 | 47.71% (1133) | 49.81% (1183) | +2.11 | [+0.11, +4.10] | 318 | 268 | 50 | 0.0430 |
| choice_options:4 | 10 | 90.00% (9) | 80.00% (8) | -10.00 | [-28.59, +8.59] | 0 | 1 | -1 | 1.0000 |
| choice_options:6 | 463 | 47.73% (221) | 47.52% (220) | -0.22 | [-4.25, +3.82] | 45 | 46 | -1 | 1.0000 |
| choice_options:7 | 225 | 18.22% (41) | 24.44% (55) | +6.22 | [-0.70, +13.14] | 39 | 25 | 14 | 0.1042 |
| choice_options:8 | 240 | 17.92% (43) | 17.50% (42) | -0.42 | [-6.58, +5.75] | 28 | 29 | -1 | 1.0000 |

## Rescue / harm transitions

### Overall

| | R1 correct | R1 incorrect |
|---|---:|---:|
| S1 correct | 2027 (both correct) | 500 (harm) |
| S1 incorrect | 1110 (rescue) | 1830 (both incorrect) |

### choice

| | R1 correct | R1 incorrect |
|---|---:|---:|
| S1 correct | 1078 (both correct) | 369 (harm) |
| S1 incorrect | 430 (rescue) | 1436 (both incorrect) |

### noul

| | R1 correct | R1 incorrect |
|---|---:|---:|
| S1 correct | 917 (both correct) | 97 (harm) |
| S1 incorrect | 531 (rescue) | 53 (both incorrect) |

### score

| | R1 correct | R1 incorrect |
|---|---:|---:|
| S1 correct | 32 (both correct) | 34 (harm) |
| S1 incorrect | 149 (rescue) | 341 (both incorrect) |

### score, 3 levels

| | R1 correct | R1 incorrect |
|---|---:|---:|
| S1 correct | 24 (both correct) | 0 (harm) |
| S1 incorrect | 9 (rescue) | 3 (both incorrect) |

### score, 5 levels

| | R1 correct | R1 incorrect |
|---|---:|---:|
| S1 correct | 8 (both correct) | 34 (harm) |
| S1 incorrect | 140 (rescue) | 338 (both incorrect) |

## Similarity versus outcome

The index stores sign-quantized (bit) vectors, so sqlite-vec returns a **Hamming distance** over 768 bits, not a float cosine distance. Estimated cosine = cos(π·h/768) (sign-quantization angle estimate). Bins are observed quantiles of the Top-1 Hamming distance (closest first); equal distances share a bin. They are descriptive only — no gate.

### All kinds

| Bin | Hamming | Est. cosine | n | S1 acc | R1 acc | Rescue | Harm | Net change (pp) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| q1 | 38–63 | 0.967–0.988 | 1118 | 64.94% | 76.57% | 202 | 72 | +11.63 |
| q2 | 64–69 | 0.960–0.966 | 1125 | 41.87% | 52.62% | 229 | 108 | +10.76 |
| q3 | 70–77 | 0.951–0.959 | 1132 | 48.23% | 60.78% | 251 | 109 | +12.54 |
| q4 | 78–87 | 0.937–0.950 | 1000 | 43.60% | 55.20% | 215 | 99 | +11.60 |
| q5 | 88–108 | 0.904–0.936 | 1092 | 31.87% | 41.12% | 213 | 112 | +9.25 |

### choice

| Bin | Hamming | Est. cosine | n | S1 acc | R1 acc | Rescue | Harm | Net change (pp) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| q1 | 38–65 | 0.965–0.988 | 952 | 41.81% | 44.22% | 146 | 123 | +2.42 |
| q2 | 66–70 | 0.959–0.964 | 374 | 52.41% | 53.48% | 45 | 41 | +1.07 |
| q3 | 71–79 | 0.948–0.958 | 720 | 51.67% | 54.58% | 83 | 62 | +2.92 |
| q4 | 80–89 | 0.934–0.947 | 616 | 43.67% | 47.89% | 89 | 63 | +4.22 |
| q5 | 90–108 | 0.904–0.933 | 651 | 32.57% | 30.57% | 67 | 80 | -2.00 |

### noul

| Bin | Hamming | Est. cosine | n | S1 acc | R1 acc | Rescue | Harm | Net change (pp) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| q1 | 42–59 | 0.971–0.985 | 358 | 75.98% | 97.21% | 80 | 4 | +21.23 |
| q2 | 60–65 | 0.965–0.970 | 325 | 74.15% | 97.23% | 80 | 5 | +23.08 |
| q3 | 66–72 | 0.957–0.964 | 313 | 66.45% | 93.93% | 101 | 15 | +27.48 |
| q4 | 73–82 | 0.944–0.956 | 291 | 56.01% | 84.19% | 125 | 43 | +28.18 |
| q5 | 83–107 | 0.906–0.943 | 311 | 41.80% | 78.78% | 145 | 30 | +36.98 |

### score

| Bin | Hamming | Est. cosine | n | S1 acc | R1 acc | Rescue | Harm | Net change (pp) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| q1 | 60–67 | 0.963–0.970 | 111 | 8.11% | 34.23% | 36 | 7 | +26.13 |
| q2 | 68–72 | 0.957–0.962 | 117 | 10.26% | 26.50% | 30 | 11 | +16.24 |
| q3 | 73–79 | 0.948–0.956 | 117 | 15.38% | 43.59% | 41 | 8 | +28.21 |
| q4 | 80–89 | 0.934–0.947 | 102 | 14.71% | 36.27% | 24 | 2 | +21.57 |
| q5 | 90–107 | 0.906–0.933 | 109 | 11.01% | 22.02% | 18 | 6 | +11.01 |

## Retrieval relationships

| Kind | n | Same group | Same domain | Same family | Same score option texts |
|---|---:|---:|---:|---:|---:|
| all | 5467 | 0.00% (0/5467) | 32.19% (1760/5467) | 100.00% (5467/5467) | 100.00% (556/556) |
| choice | 3313 | 0.00% (0/3313) | 21.73% (720/3313) | 100.00% (3313/3313) | n/a |
| noul | 1598 | 0.00% (0/1598) | 32.54% (520/1598) | 100.00% (1598/1598) | n/a |
| score | 556 | 0.00% (0/556) | 93.53% (520/556) | 100.00% (556/556) | 100.00% (556/556) |

Outcome by domain / family relationship:

| Group | n | S1 acc | R1 acc | Diff (pp) | 95% CI (pp) | Rescue | Harm | Net | McNemar p |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| same_domain | 1760 | 21.59% (380) | 38.41% (676) | +16.82 | [+13.94, +19.69] | 506 | 210 | 296 | 0.0000 |
| different_domain | 3707 | 57.92% (2147) | 66.39% (2461) | +8.47 | [+6.91, +10.03] | 604 | 290 | 314 | 0.0000 |
| same_family | 5467 | 46.22% (2527) | 57.38% (3137) | +11.16 | [+9.75, +12.57] | 1110 | 500 | 610 | 0.0000 |
| different_family | 0 | 0.00% (0) | 0.00% (0) | +0.00 | [+0.00, +0.00] | 0 | 0 | 0 | 1.0000 |

## Retrieved answer-label bias

Label agreement compares candidate labels (A, B, ...). Semantic agreement compares the option text of the prediction with the demonstration's answer text; for noul and score the label and the semantic value coincide by construction (fixed false/true order, ordered levels), for choice they do not. Label agreement alone is not evidence of reasoning.

| Kind | n | Demo label = GT label | S1 = demo label | R1 = demo label | Moved toward demo label | of which rescue / harm / other | Moved away | R1 = demo answer text | S1 = demo answer text |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| all | 5467 | 47.80% (2613/5467) | 38.45% (2102/5467) | 60.29% (3296/5467) | 1549 | 794 / 279 / 476 | 355 | 57.62% (3150/5467) | 39.58% (2164/5467) |
| choice | 3313 | 33.53% (1111/3313) | 31.87% (1056/3313) | 40.90% (1355/3313) | 584 | 174 / 164 / 246 | 285 | 36.49% (1209/3313) | 33.75% (1118/3313) |
| noul | 1598 | 83.48% (1334/1598) | 59.20% (946/1598) | 89.99% (1438/1598) | 560 | 476 / 84 / 0 | 68 | 89.99% (1438/1598) | 59.20% (946/1598) |
| score | 556 | 30.22% (168/556) | 17.99% (100/556) | 90.47% (503/556) | 405 | 144 / 31 / 230 | 2 | 90.47% (503/556) | 17.99% (100/556) |

### Label distributions (paired queries)

| Kind | Label | Demo answers | S1 predictions | R1 predictions | Ground truth |
|---|---|---:|---:|---:|---:|
| choice | A | 950 | 869 | 879 | 928 |
| choice | B | 964 | 1579 | 1263 | 892 |
| choice | C | 1185 | 647 | 856 | 995 |
| choice | D | 106 | 49 | 97 | 123 |
| choice | E | 10 | 30 | 35 | 122 |
| choice | F | 90 | 68 | 123 | 148 |
| choice | G | 8 | 45 | 56 | 70 |
| choice | H | 0 | 26 | 4 | 35 |
| noul | A | 1242 | 804 | 1368 | 1262 |
| noul | B | 356 | 794 | 230 | 336 |
| score | A | 32 | 18 | 56 | 28 |
| score | B | 98 | 538 | 111 | 55 |
| score | C | 85 | 0 | 71 | 116 |
| score | D | 264 | 0 | 241 | 215 |
| score | E | 77 | 0 | 77 | 142 |

## Margins

| Kind | S1 mean winner margin | R1 mean winner margin | S1 mean signed margin | R1 mean signed margin |
|---|---:|---:|---:|---:|
| all | 2.008 | 4.313 | -0.358 | 1.432 |
| choice | 2.785 | 3.360 | -0.296 | -0.286 |
| noul | 0.663 | 6.925 | 0.271 | 6.248 |
| score | 1.248 | 2.488 | -2.539 | -2.176 |

## Prompt tokens

Maximum context (n_ctx): 4096. Query tokens = S1 prompt tokens; exemplar tokens = R1 − S1 prompt tokens (demonstration turn).

| Kind | S1 mean | R1 mean | Extra mean | Extra median | Extra p95 | R1 max |
|---|---:|---:|---:|---:|---:|---:|
| all | 325.2 | 659.5 | 334.3 | 244 | 697 | 1592 |
| choice | 256.2 | 524.0 | 267.8 | 144 | 682 | 1440 |
| noul | 358.9 | 722.2 | 363.3 | 279 | 649 | 1385 |
| score | 639.8 | 1287.1 | 647.3 | 697 | 776 | 1592 |

## Latency

S1 and R1 run sequentially per query in one process; both models are loaded once and one untimed warm-up embedding and decision precede the timed loop. Every decision evaluates its full prompt from an empty KV cache (no prefix reuse between queries or between S1 and R1). The R1 total includes query embedding, vector search, exemplar read, LZ4 decompression and the 1-shot decision; it is the per-request cost of R1 on its own, not the combined experiment time.

| Condition | Stage | Mean ms | Median ms | P95 ms |
|---|---|---:|---:|---:|
| s1 | prompt_construction | 1.14 | 0.82 | 2.47 |
| s1 | prompt_evaluation | 197.66 | 143.63 | 323.62 |
| s1 | answer_scoring | 0.00 | 0.00 | 0.01 |
| s1 | total | 198.81 | 144.49 | 325.74 |
| r1 | query_embedding | 4.69 | 4.34 | 6.08 |
| r1 | vector_search | 99.88 | 55.75 | 334.99 |
| r1 | exemplar_retrieval | 0.01 | 0.01 | 0.01 |
| r1 | lz4_decompression | 0.03 | 0.02 | 0.07 |
| r1 | prompt_construction | 1.67 | 1.24 | 3.05 |
| r1 | prompt_evaluation | 295.45 | 179.04 | 526.64 |
| r1 | answer_scoring | 0.00 | 0.00 | 0.01 |
| r1 | decision_total | 297.13 | 180.34 | 529.48 |
| r1 | total | 401.74 | 249.24 | 835.96 |

- R1 / S1 latency multiplier (mean): **2.02×**
- R1 / S1 latency multiplier (median): 1.73×
- Total evaluation wall time: 3525.1 s (S1 decisions 1191.7 s, R1 incl. retrieval 2307.7 s)

## Context-limit failures

None.

## Main observations

- R1 − S1 = +11.16 pp over 5467 paired queries (95% CI [+9.75, +12.57]); the interval excludes zero.
- 1110 rescues vs 500 harms (29.45% (1610/5467) of paired queries change correctness).
- S1 parity: S1 re-run after R1 on the first 50 queries — 0 mismatches.
- Exemplar DB holds 158898 records outside the training split; they are never used as demonstrations.
- 79 training records have no one-hot answer (soft target) and are not eligible exemplars.
- R2 gate at estimated cosine >= 0.903989: 344 queries kept S1 without a 1-shot pass, so the R1 columns of this report cover activated queries only.


# R1 — Retrieved 1-Shot Inference Evaluation

Corpus: **open-jev-validation**  
Run date: 2026-10-16T19:44:33

## Configuration

| Setting | Value |
|---|---|
| input_path | datasets/validation.jsonl |
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

- Evaluated queries: 11674
- Paired (S1 ok and R1 1-shot ok): **11674** (100.00%)
- S1 status: ok=11674
- Retrieval status: ok=11674
- R1 status: ok=11674

Unpaired queries are excluded from every paired metric below; no zero-shot fallback is counted as a 1-shot result.

## Headline (paired)

| Metric | Value |
|---|---|
| Paired examples | 11674 |
| S1 accuracy | 41.86% (4887) |
| R1 accuracy | 52.72% (6155) |
| Difference | +10.86 pp |
| 95% CI (paired Wald) | [+9.87, +11.86] pp |
| Rescue / Harm / Net | 2459 / 1191 / 1268 |
| McNemar χ² (cc) p | 0.000000 |
| Exact binomial p | 0.000000 |

Check: (rescue − harm) / n = +10.86 pp = R1 acc − S1 acc.

## Accuracy by group

| Group | n | S1 acc | R1 acc | Diff (pp) | 95% CI (pp) | Rescue | Harm | Net | McNemar p |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| overall | 11674 | 41.86% (4887) | 52.72% (6155) | +10.86 | [+9.87, +11.86] | 2459 | 1191 | 1268 | 0.0000 |
| kind:choice | 7511 | 40.23% (3022) | 41.78% (3138) | +1.54 | [+0.43, +2.66] | 970 | 854 | 116 | 0.0071 |
| kind:noul | 2921 | 58.58% (1711) | 88.94% (2598) | +30.37 | [+28.13, +32.61] | 1135 | 248 | 887 | 0.0000 |
| kind:score | 1242 | 12.40% (154) | 33.74% (419) | +21.34 | [+18.23, +24.44] | 354 | 89 | 265 | 0.0000 |
| score_levels:3 | 82 | 64.63% (53) | 90.24% (74) | +25.61 | [+15.02, +36.20] | 23 | 2 | 21 | 0.0001 |
| score_levels:5 | 1160 | 8.71% (101) | 29.74% (345) | +21.03 | [+17.80, +24.27] | 331 | 87 | 244 | 0.0000 |
| choice_options:3 | 4830 | 47.66% (2302) | 48.63% (2349) | +0.97 | [-0.39, +2.34] | 588 | 541 | 47 | 0.1710 |
| choice_options:4 | 25 | 88.00% (22) | 68.00% (17) | -20.00 | [-35.68, -4.32] | 0 | 5 | -5 | 0.0736 |
| choice_options:6 | 1071 | 39.31% (421) | 41.18% (441) | +1.87 | [-1.09, +4.83] | 141 | 121 | 20 | 0.2405 |
| choice_options:7 | 809 | 18.05% (146) | 21.51% (174) | +3.46 | [-0.12, +7.05] | 124 | 96 | 28 | 0.0687 |
| choice_options:8 | 776 | 16.88% (131) | 20.23% (157) | +3.35 | [-0.28, +6.99] | 117 | 91 | 26 | 0.0830 |

## Rescue / harm transitions

### Overall

| | R1 correct | R1 incorrect |
|---|---:|---:|
| S1 correct | 3696 (both correct) | 1191 (harm) |
| S1 incorrect | 2459 (rescue) | 4328 (both incorrect) |

### choice

| | R1 correct | R1 incorrect |
|---|---:|---:|
| S1 correct | 2168 (both correct) | 854 (harm) |
| S1 incorrect | 970 (rescue) | 3519 (both incorrect) |

### noul

| | R1 correct | R1 incorrect |
|---|---:|---:|
| S1 correct | 1463 (both correct) | 248 (harm) |
| S1 incorrect | 1135 (rescue) | 75 (both incorrect) |

### score

| | R1 correct | R1 incorrect |
|---|---:|---:|
| S1 correct | 65 (both correct) | 89 (harm) |
| S1 incorrect | 354 (rescue) | 734 (both incorrect) |

### score, 3 levels

| | R1 correct | R1 incorrect |
|---|---:|---:|
| S1 correct | 51 (both correct) | 2 (harm) |
| S1 incorrect | 23 (rescue) | 6 (both incorrect) |

### score, 5 levels

| | R1 correct | R1 incorrect |
|---|---:|---:|
| S1 correct | 14 (both correct) | 87 (harm) |
| S1 incorrect | 331 (rescue) | 728 (both incorrect) |

## Similarity versus outcome

The index stores sign-quantized (bit) vectors, so sqlite-vec returns a **Hamming distance** over 768 bits, not a float cosine distance. Estimated cosine = cos(π·h/768) (sign-quantization angle estimate). Bins are observed quantiles of the Top-1 Hamming distance (closest first); equal distances share a bin. They are descriptive only — no gate.

### All kinds

| Bin | Hamming | Est. cosine | n | S1 acc | R1 acc | Rescue | Harm | Net change (pp) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| q1 | 33–65 | 0.965–0.991 | 2850 | 53.96% | 64.95% | 544 | 231 | +10.98 |
| q2 | 66–73 | 0.956–0.964 | 2083 | 50.50% | 61.74% | 423 | 189 | +11.23 |
| q3 | 74–81 | 0.946–0.955 | 2114 | 41.34% | 54.12% | 494 | 224 | +12.77 |
| q4 | 82–95 | 0.925–0.944 | 2461 | 37.63% | 45.23% | 467 | 280 | +7.60 |
| q5 | 96–127 | 0.868–0.924 | 2166 | 22.95% | 35.13% | 531 | 267 | +12.19 |

### choice

| Bin | Hamming | Est. cosine | n | S1 acc | R1 acc | Rescue | Harm | Net change (pp) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| q1 | 33–65 | 0.965–0.991 | 1562 | 45.20% | 47.12% | 226 | 196 | +1.92 |
| q2 | 66–75 | 0.953–0.964 | 1609 | 48.79% | 50.40% | 199 | 173 | +1.62 |
| q3 | 76–83 | 0.943–0.952 | 1402 | 44.01% | 46.65% | 173 | 136 | +2.64 |
| q4 | 84–97 | 0.922–0.942 | 1760 | 35.97% | 37.90% | 241 | 207 | +1.93 |
| q5 | 98–127 | 0.868–0.921 | 1178 | 23.85% | 22.92% | 131 | 142 | -0.93 |

### noul

| Bin | Hamming | Est. cosine | n | S1 acc | R1 acc | Rescue | Harm | Net change (pp) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| q1 | 42–60 | 0.970–0.985 | 611 | 76.43% | 97.22% | 134 | 7 | +20.79 |
| q2 | 61–67 | 0.963–0.969 | 616 | 73.54% | 96.75% | 159 | 16 | +23.21 |
| q3 | 68–75 | 0.953–0.962 | 591 | 58.71% | 90.52% | 237 | 49 | +31.81 |
| q4 | 76–87 | 0.937–0.952 | 533 | 46.72% | 80.11% | 269 | 91 | +33.40 |
| q5 | 88–111 | 0.899–0.936 | 570 | 34.21% | 78.25% | 336 | 85 | +44.04 |

### score

| Bin | Hamming | Est. cosine | n | S1 acc | R1 acc | Rescue | Harm | Net change (pp) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| q1 | 53–67 | 0.963–0.977 | 265 | 12.08% | 32.08% | 76 | 23 | +20.00 |
| q2 | 68–74 | 0.955–0.962 | 254 | 12.99% | 33.07% | 72 | 21 | +20.08 |
| q3 | 75–84 | 0.942–0.953 | 230 | 16.52% | 41.30% | 69 | 12 | +24.78 |
| q4 | 85–94 | 0.927–0.940 | 251 | 13.55% | 34.26% | 71 | 19 | +20.72 |
| q5 | 95–107 | 0.906–0.925 | 242 | 7.02% | 28.51% | 66 | 14 | +21.49 |

## Retrieval relationships

| Kind | n | Same group | Same domain | Same family | Same score option texts |
|---|---:|---:|---:|---:|---:|
| all | 11674 | 0.00% (0/11674) | 39.75% (4640/11674) | 100.00% (11674/11674) | 100.00% (1242/1242) |
| choice | 7511 | 0.00% (0/7511) | 30.89% (2320/7511) | 100.00% (7511/7511) | n/a |
| noul | 2921 | 0.00% (0/2921) | 39.71% (1160/2921) | 100.00% (2921/2921) | n/a |
| score | 1242 | 0.00% (0/1242) | 93.40% (1160/1242) | 100.00% (1242/1242) | 100.00% (1242/1242) |

Outcome by domain / family relationship:

| Group | n | S1 acc | R1 acc | Diff (pp) | 95% CI (pp) | Rescue | Harm | Net | McNemar p |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| same_domain | 4640 | 19.76% (917) | 36.75% (1705) | +16.98 | [+15.16, +18.80] | 1389 | 601 | 788 | 0.0000 |
| different_domain | 7034 | 56.44% (3970) | 63.26% (4450) | +6.82 | [+5.70, +7.95] | 1070 | 590 | 480 | 0.0000 |
| same_family | 11674 | 41.86% (4887) | 52.72% (6155) | +10.86 | [+9.87, +11.86] | 2459 | 1191 | 1268 | 0.0000 |
| different_family | 0 | 0.00% (0) | 0.00% (0) | +0.00 | [+0.00, +0.00] | 0 | 0 | 0 | 1.0000 |

## Retrieved answer-label bias

Label agreement compares candidate labels (A, B, ...). Semantic agreement compares the option text of the prediction with the demonstration's answer text; for noul and score the label and the semantic value coincide by construction (fixed false/true order, ordered levels), for choice they do not. Label agreement alone is not evidence of reasoning.

| Kind | n | Demo label = GT label | S1 = demo label | R1 = demo label | Moved toward demo label | of which rescue / harm / other | Moved away | R1 = demo answer text | S1 = demo answer text |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| all | 11674 | 43.39% (5065/11674) | 32.06% (3743/11674) | 57.72% (6738/11674) | 3677 | 1764 / 667 / 1246 | 682 | 53.72% (6271/11674) | 33.30% (3888/11674) |
| choice | 7511 | 29.89% (2245/7511) | 28.01% (2104/7511) | 38.74% (2910/7511) | 1382 | 371 / 353 / 658 | 576 | 32.53% (2443/7511) | 29.94% (2249/7511) |
| noul | 2921 | 83.12% (2428/2921) | 51.63% (1508/2921) | 91.85% (2683/2921) | 1279 | 1050 / 229 / 0 | 104 | 91.85% (2683/2921) | 51.63% (1508/2921) |
| score | 1242 | 31.56% (392/1242) | 10.55% (131/1242) | 92.19% (1145/1242) | 1016 | 343 / 85 / 588 | 2 | 92.19% (1145/1242) | 10.55% (131/1242) |

### Label distributions (paired queries)

| Kind | Label | Demo answers | S1 predictions | R1 predictions | Ground truth |
|---|---|---:|---:|---:|---:|
| choice | A | 1959 | 2135 | 1985 | 1985 |
| choice | B | 1824 | 3613 | 2564 | 2001 |
| choice | C | 2732 | 1295 | 1852 | 2061 |
| choice | D | 206 | 98 | 290 | 358 |
| choice | E | 341 | 59 | 295 | 373 |
| choice | F | 430 | 121 | 295 | 412 |
| choice | G | 19 | 101 | 209 | 225 |
| choice | H | 0 | 89 | 21 | 96 |
| noul | A | 2379 | 1274 | 2545 | 2292 |
| noul | B | 542 | 1647 | 376 | 629 |
| score | A | 50 | 38 | 85 | 59 |
| score | B | 124 | 1204 | 151 | 123 |
| score | C | 229 | 0 | 218 | 265 |
| score | D | 440 | 0 | 402 | 474 |
| score | E | 399 | 0 | 386 | 321 |

## Margins

| Kind | S1 mean winner margin | R1 mean winner margin | S1 mean signed margin | R1 mean signed margin |
|---|---:|---:|---:|---:|
| all | 1.933 | 3.934 | -0.565 | 0.894 |
| choice | 2.475 | 3.085 | -0.497 | -0.628 |
| noul | 0.745 | 6.810 | 0.167 | 5.955 |
| score | 1.451 | 2.302 | -2.691 | -1.807 |

## Prompt tokens

Maximum context (n_ctx): 4096. Query tokens = S1 prompt tokens; exemplar tokens = R1 − S1 prompt tokens (demonstration turn).

| Kind | S1 mean | R1 mean | Extra mean | Extra median | Extra p95 | R1 max |
|---|---:|---:|---:|---:|---:|---:|
| all | 360.7 | 709.6 | 348.9 | 260 | 682 | 1661 |
| choice | 303.7 | 596.0 | 292.3 | 147 | 668 | 1567 |
| noul | 385.1 | 763.9 | 378.8 | 286 | 637 | 1522 |
| score | 648.1 | 1269.1 | 621.0 | 634 | 734 | 1661 |

## Latency

S1 and R1 run sequentially per query in one process; both models are loaded once and one untimed warm-up embedding and decision precede the timed loop. Every decision evaluates its full prompt from an empty KV cache (no prefix reuse between queries or between S1 and R1). The R1 total includes query embedding, vector search, exemplar read, LZ4 decompression and the 1-shot decision; it is the per-request cost of R1 on its own, not the combined experiment time.

| Condition | Stage | Mean ms | Median ms | P95 ms |
|---|---|---:|---:|---:|
| s1 | prompt_construction | 1.30 | 0.97 | 2.77 |
| s1 | prompt_evaluation | 224.20 | 154.81 | 467.54 |
| s1 | answer_scoring | 0.00 | 0.00 | 0.01 |
| s1 | total | 225.51 | 155.77 | 470.40 |
| r1 | query_embedding | 4.85 | 4.70 | 6.84 |
| r1 | vector_search | 125.94 | 65.46 | 342.94 |
| r1 | exemplar_retrieval | 0.01 | 0.01 | 0.01 |
| r1 | lz4_decompression | 0.03 | 0.02 | 0.07 |
| r1 | prompt_construction | 1.72 | 1.35 | 3.04 |
| r1 | prompt_evaluation | 339.00 | 277.96 | 726.79 |
| r1 | answer_scoring | 0.00 | 0.00 | 0.01 |
| r1 | decision_total | 340.74 | 279.48 | 729.05 |
| r1 | total | 471.57 | 338.88 | 950.97 |

- R1 / S1 latency multiplier (mean): **2.09×**
- R1 / S1 latency multiplier (median): 2.18×
- Total evaluation wall time: 8184.1 s (S1 decisions 2632.6 s, R1 incl. retrieval 5505.1 s)

## Context-limit failures

None.

## Main observations

- R1 − S1 = +10.86 pp over 11674 paired queries (95% CI [+9.87, +11.86]); the interval excludes zero.
- 2459 rescues vs 1191 harms (31.27% (3650/11674) of paired queries change correctness).
- S1 parity: S1 re-run after R1 on the first 50 queries — 0 mismatches.
- Exemplar DB holds 158898 records outside the training split; they are never used as demonstrations.
- 79 training records have no one-hot answer (soft target) and are not eligible exemplars.


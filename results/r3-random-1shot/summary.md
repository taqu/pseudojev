# R3 — Random 1-shot vs Nearest 1-shot

Central question: does nearest-neighbour selection add accuracy over an arbitrary eligible training demonstration, enough to pay for its retrieval cost?

## Configuration

| Setting | Value |
|---|---|
| Inference model | models/bonsai2_2b.gguf (Ternary-Bonsai-2-2B, sha256 `36061ad406df8a4903a6db005be350c50fabfaf717f9ef0acf4b76adcd9929fa`) |
| Embedding model | models/nomic-embed-text-v1.5.gguf (fingerprint `3e24342164b3d949`; R1 retrieval and similarity analysis only) |
| Exemplar DB | models/examplar.db (schema 4, read-only) |
| Training allowlist | datasets/train.jsonl |
| Queries | datasets/validation.jsonl (sha256 `ce5d18132b4967ec5516a07b41700babeebef43b5d3dcc63ea62f01fcb4767cc`), held-out partition |
| Prompt format | r1-demo-turn-v1 — S1 user turn for the exemplar, assistant turn = exemplar answer label, then the unchanged S1 query prompt |
| Decision | letters scheme, auto layout, single order, no calibration; candidate-logit argmax, correctness by candidate index |
| n_ctx | 4096 |
| Reference S1 / R1 | results/r1-2b/predictions.jsonl (sha256 `336c0cd757f48245…`, unchanged) |
| R3 selection | XXH64("r3-random-v1|<seed>|<query_id>") mod pool_size, over the pool sorted by exemplar ID with the query itself removed |
| Seeds (fixed before any R3 result) | 42, 123, 456, 789, 2026, 7, 99, 314, 1000, 31337 |
| Run | 2026-10-19T16:41:12, wall 17914 s |
| Bootstrap | 2000 cluster replicates over group_id, seed 20261019 |

All identity keys of the R1 configuration (model hash, embedding fingerprint, DB, prompt format, scheme, layout, ensemble, calibration, n_ctx, top_k, leakage policy) were checked equal to the reference before the run.

## Data integrity

- Held-out partition recomputed with the R2 implementation (`r2::apply_partition`) and compared with the saved assignments in `results/r2/predictions.jsonl`: 5811 evaluation queries, 0 disagreements, 0 groups straddling tuning/evaluation, 0 queries labelled train.
- Evaluation families: `contracts/governing_law_venue` `ir/access_schedule` `rubric_judge/trace_consistency` `support/damaged_order`; remaining queries split by group_id.
- Alignment: 5811 queries present in the reference and in every seed; missing 0, unexpected 0, duplicates 0, kind/ground-truth disagreements 0.
- Analysed (S1, R1 and all 10 R3 seeds ok): **5811**. Reference failures: 0; R3 failures per seed: 7=0 42=0 99=0 123=0 314=0 456=0 789=0 1000=0 2026=0 31337=0. No query is excluded silently: failed queries would be listed here and dropped from every strategy alike.
- Eligible pool = R1's candidate population: 111596 of 270573 vector-index rows (158898 outside the training split, 79 training rows without a one-hot answer); partitioned by R1's KNN filter (kind, and level count for score): choice=96268 noul=7822 score/3=546 score/5=6960. The query itself is removed before drawing. Queries without an eligible exemplar: 0.
- R3-L control `results/r3-length-matched-1shot`: same reference (sha256 checked), identical identity keys, aligned on the same queries (missing 0, duplicates 0); queries dropped because an R3-L seed failed: 0.
- Leakage: every selected exemplar was re-checked at load time (in the allowlist, one-hot, same kind/levels, not the query).
- Reference reproducibility: S1 and nearest R1 re-run in the R3 process on the first 100 held-out queries — S1 logit/token mismatches 0, nearest exemplar mismatches 0, R1 logit mismatches 0. Similarity probe reproduced the KNN Hamming distance on 100/100 nearest exemplars.
- Context: R1's policy (no length filter, no truncation); context-limit failures are counted as R3 failures above.

## Overall results (held-out, n=5811)

| Strategy | Accuracy |
|---|---:|
| S1 — zero-shot | 44.69% |
| R1 — nearest 1-shot | 55.05% |
| R3 — random 1-shot (mean of 10 seeds) | 51.79% |

| Effect | Difference | 95% cluster bootstrap |
|---|---:|---|
| General 1-shot effect (R3 − S1) | +7.10 pp | [+5.70, +8.43] pp |
| Nearest-selection effect (R1 − R3) | +3.26 pp | [+1.50, +4.86] pp |
| Total (R1 − S1) | +10.36 pp | [+8.42, +12.03] pp |

Share of the R1 − S1 improvement attributable to nearest selection, (R1 − R3) / (R1 − S1): **31.4%** (95% cluster bootstrap [16.8%, 43.4%]).

The bootstrap resamples evaluation groups (1598 clusters); each sampled query keeps its S1, R1 and all seed outcomes, so repeated predictions of one query are never treated as independent.

## Seed variance

| Seed | R3 accuracy | R3 − S1 | R1 − R3 | R1 − R3 95% CI |
|---:|---:|---:|---:|---|
| 42 | 51.82% | +7.12 pp | +3.24 pp | [+1.36, +4.98] pp |
| 123 | 52.14% | +7.45 pp | +2.91 pp | [+0.95, +4.62] pp |
| 456 | 51.57% | +6.88 pp | +3.48 pp | [+1.57, +5.14] pp |
| 789 | 51.99% | +7.30 pp | +3.06 pp | [+1.08, +4.87] pp |
| 2026 | 51.28% | +6.59 pp | +3.77 pp | [+1.96, +5.40] pp |
| 7 | 52.30% | +7.61 pp | +2.75 pp | [+0.79, +4.54] pp |
| 99 | 51.26% | +6.57 pp | +3.79 pp | [+1.86, +5.49] pp |
| 314 | 52.18% | +7.49 pp | +2.87 pp | [+0.91, +4.61] pp |
| 1000 | 51.85% | +7.16 pp | +3.20 pp | [+1.24, +5.00] pp |
| 31337 | 51.56% | +6.87 pp | +3.49 pp | [+1.65, +5.20] pp |

R3 across seeds: mean 51.79%, SD 0.37 pp, min 51.26%, max 52.30%. R1 beats R3 on 10/10 seeds (CI above zero on 10), R3 beats R1 on 0/10 (CI below zero on 0).

## Task-level results

| Kind | n | S1 | R1 | R3 mean (SD) | R3 min–max | R3 − S1 [95% CI] | R1 − R3 [95% CI] | Selection share |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| choice | 3656 | 41.49% | 42.92% | 43.20% (0.51) | 42.37% – 43.82% | +1.71 pp [+0.52, +2.88] pp | -0.28 pp [-1.42, +0.88] pp | -20% |
| noul | 1599 | 63.41% | 90.62% | 79.01% (0.28) | 78.42% – 79.24% | +15.59 pp [+11.29, +21.63] pp | +11.61 pp [+8.61, +13.56] pp | 43% |
| score | 556 | 11.87% | 32.55% | 30.05% (1.69) | 27.70% – 32.37% | +18.18 pp [+15.36, +20.92] pp | +2.50 pp [-1.19, +6.25] pp | 12% |
| all | 5811 | 44.69% | 55.05% | 51.79% (0.37) | 51.26% – 52.30% | +7.10 pp [+5.70, +8.43] pp | +3.26 pp [+1.50, +4.86] pp | 31% |

Per-seed R1 − R3 by kind:

| Kind | 42 | 123 | 456 | 789 | 2026 | 7 | 99 | 314 | 1000 | 31337 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| choice | -0.27 pp | -0.79 pp | -0.14 pp | -0.68 pp | +0.36 pp | -0.79 pp | +0.14 pp | -0.90 pp | -0.30 pp | +0.55 pp |
| noul | +11.51 pp | +12.20 pp | +11.44 pp | +11.44 pp | +11.38 pp | +11.44 pp | +11.76 pp | +12.01 pp | +11.57 pp | +11.38 pp |
| score | +2.52 pp | +0.54 pp | +4.32 pp | +3.60 pp | +4.32 pp | +1.08 pp | +4.86 pp | +1.44 pp | +2.16 pp | +0.18 pp |
| all | +3.24 pp | +2.91 pp | +3.48 pp | +3.06 pp | +3.77 pp | +2.75 pp | +3.79 pp | +2.87 pp | +3.20 pp | +3.49 pp |

Selection share is (R1 − R3) / (R1 − S1); shown only where R1 − S1 exceeds 0.5 pp.

## Rescue and harm (relative to S1)

| Kind | Strategy | Rescue | Harm | Net | Net / n |
|---|---|---:|---:|---:|---:|
| choice | R1 | 458 | 406 | 52 | +1.42 pp |
| choice | R3 mean | 441.2 | 378.8 | 62.4 | +1.71 pp |
| noul | R1 | 532 | 97 | 435 | +27.20 pp |
| noul | R3 mean | 509.8 | 260.5 | 249.3 | +15.59 pp |
| score | R1 | 149 | 34 | 115 | +20.68 pp |
| score | R3 mean | 140.3 | 39.2 | 101.1 | +18.18 pp |
| all | R1 | 1139 | 537 | 602 | +10.36 pp |
| all | R3 mean | 1091.3 | 678.5 | 412.8 | +7.10 pp |

R3 per seed (all kinds):

| Seed | Rescue | Harm | Net |
|---:|---:|---:|---:|
| 42 | 1099 | 685 | 414 |
| 123 | 1116 | 683 | 433 |
| 456 | 1075 | 675 | 400 |
| 789 | 1085 | 661 | 424 |
| 2026 | 1076 | 693 | 383 |
| 7 | 1128 | 686 | 442 |
| 99 | 1075 | 693 | 382 |
| 314 | 1087 | 652 | 435 |
| 1000 | 1092 | 676 | 416 |
| 31337 | 1080 | 681 | 399 |

Direct R1 vs R3 (mean over seeds):

| Kind | R1 correct / R3 incorrect | R1 incorrect / R3 correct | Both correct | Both incorrect |
|---|---:|---:|---:|---:|
| choice | 416.5 | 426.9 | 1152.5 | 1660.1 |
| noul | 211.9 | 26.2 | 1237.1 | 123.8 |
| score | 107.4 | 93.5 | 73.6 | 281.5 |
| all | 735.8 | 546.6 | 2463.2 | 2065.4 |

Nearest vs random selection changes rescues by +47.7 and harms by -141.5 (R1 minus R3 mean, all kinds).

## Length-matched control (R3-L)

Random demonstrations differ from nearest ones in length (see Cost analysis), so a secondary control draws uniformly from the 100 eligible exemplars whose demonstration length is closest to the nearest exemplar's (same pools, same leakage rules, no similarity used; 5 seeds: 42, 123, 456, 789, 2026). R3 remains the primary baseline.

| Kind | n | R1 | R3 mean | R3-L mean (SD) | R1 − R3-L [95% CI] | R3-L − R3 [95% CI] | Demo tokens R1 / R3 / R3-L |
|---|---:|---:|---:|---:|---:|---:|---|
| choice | 3656 | 42.92% | 43.20% | 42.01% (0.54) | +0.91 pp [-0.30, +2.02] pp | -1.19 pp [-1.88, -0.52] pp | 295 / 208 / 295 |
| noul | 1599 | 90.62% | 79.01% | 81.56% (0.39) | +9.06 pp [+6.89, +10.58] pp | +2.56 pp [+1.61, +3.23] pp | 363 / 527 / 362 |
| score | 556 | 32.55% | 30.05% | 30.14% (2.04) | +2.41 pp [-1.56, +6.19] pp | +0.09 pp [-1.85, +2.12] pp | 647 / 626 / 647 |
| all | 5811 | 55.05% | 51.79% | 51.76% (0.44) | +3.29 pp [+1.87, +4.56] pp | -0.04 pp [-0.65, +0.58] pp | 348 / 336 / 347 |

R3-L per seed (all kinds): 42 = 51.13%, 123 = 51.56%, 456 = 52.31%, 789 = 51.94%, 2026 = 51.85%. R3-L − S1 = +7.07 pp [+5.73, +8.36] pp. Draws identical to the nearest exemplar: 191 / 29055. Mean similarity of R3-L draws: 0.8589.

## Demonstration label effects

Share of queries whose demonstration answer label equals the query's ground-truth label, and how often the prediction copies the demonstration label. For noul and score the label also fixes the semantic value.

| Kind | Nearest demo = GT | Random demo = GT | S1 = nearest demo | R1 = nearest demo | S1 = random demo | R3 = random demo |
|---|---:|---:|---:|---:|---:|---:|
| choice | 31.78% | 25.35% | 31.84% | 42.04% | 28.60% | 30.25% |
| noul | 83.49% | 65.10% | 59.16% | 89.99% | 49.84% | 77.94% |
| score | 30.22% | 29.48% | 17.99% | 90.47% | 13.06% | 90.02% |
| all | 45.86% | 36.68% | 38.03% | 59.87% | 32.96% | 49.09% |

## Similarity analysis (exploratory)

Estimated cosine cos(π·h/768) from the Hamming distance between the query's binary embedding and the exemplar's stored vector (same quantisation as R1's KNN). Similarity was computed after selection and never influenced it.

| Kind | Nearest mean | Nearest median | Random mean | Random median | Random p90 |
|---|---:|---:|---:|---:|---:|
| choice | 0.9446 | 0.9508 | 0.7940 | 0.8199 | 0.9040 |
| noul | 0.9571 | 0.9616 | 0.7514 | 0.7327 | 0.8535 |
| score | 0.9489 | 0.9533 | 0.8471 | 0.8470 | 0.9125 |
| all | 0.9485 | 0.9545 | 0.7873 | 0.8008 | 0.8987 |

Outcome by similarity bin (bins = sextiles of the pooled nearest + random similarities; random counts are query×seed draws):

| Bin | Est. cosine | Nearest n | R1 acc | R1 rescue / harm | Random draws | R3 acc | R3 rescue / harm |
|---|---|---:|---:|---|---:|---:|---|
| 1 | min – 0.692 | 0 | – | 0 / 0 | 10080 | 55.61% | 1548 / 1123 |
| 2 | 0.692 – 0.738 | 0 | – | 0 / 0 | 10877 | 54.97% | 1932 / 1400 |
| 3 | 0.738 – 0.820 | 0 | – | 0 / 0 | 10954 | 53.45% | 2407 / 1344 |
| 4 | 0.820 – 0.860 | 0 | – | 0 / 0 | 10399 | 47.45% | 2321 / 1150 |
| 5 | 0.860 – 0.902 | 336 | 17.56% | 28 / 36 | 10516 | 47.87% | 1649 / 1108 |
| 6 | 0.902 – max | 5475 | 57.35% | 1111 / 501 | 5284 | 50.93% | 1056 / 660 |

Accuracy across bins mixes query difficulty with similarity (harder queries tend to have less similar neighbours). A within-query check removes that: across the 10 random draws of the same query, the correlation between draw similarity and correctness is **-0.0017**.

## Cost analysis

| Strategy | Accuracy | Mean latency | Median | P95 | Selection (mean) | Decision (mean) | Mean prompt tokens | Mean demo tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S1 | 44.69% | 211.5 ms | 154.4 | 335.4 | – | 211.5 | 343.4 | 0 |
| R1 | 55.05% | 440.0 ms | 296.4 | 854.8 | 120.7 | 319.3 | 691.1 | 347.7 |
| R3 | 51.79% | 304.6 ms | 322.2 | 520.8 | 0.108 | 304.5 | 679.4 | 336.0 |
| R3-L (control) | 51.76% | 305.1 ms | 185.7 | 516.6 | 0.126 | 305.0 | 690.8 | 347.4 |

All values are **measured**, per query, on the held-out set. S1 and R1 come from the reference R1 run (both models loaded once, untimed warm-up, sequential, full prompt from an empty KV cache); R3 comes from this run under the same conditions. R1 selection = query embedding + vector search + row fetch + LZ4; R3 selection = hash draw + row fetch + LZ4 (no embedding, no search). Decision = prompt construction + evaluation + scoring. Hardware: the same local machine and GPU for all runs; the two runs were on different days.

Same-session check (first 100 held-out queries, S1 and nearest R1 re-run inside the R3 process):

| Strategy | Selection mean | Decision mean | Total mean |
|---|---:|---:|---:|
| S1 | – | 354.3 ms | 354.3 ms |
| R1 | 143.7 ms | 508.2 ms | 651.8 ms |
| R3 | 0.118 ms | 424.8 ms | 424.9 ms |

Demonstration length (extra prompt tokens over S1) by kind:

| Kind | R1 mean | R1 median | R3 mean | R3 median | R3 / R1 mean |
|---|---:|---:|---:|---:|---:|
| choice | 295.4 | 147 | 208.3 | 141 | 0.71 |
| noul | 363.2 | 279 | 527.1 | 546 | 1.45 |
| score | 647.3 | 697 | 626.4 | 635 | 0.97 |
| all | 347.7 | 252 | 336.0 | 160 | 0.97 |

Plot: `accuracy_latency.svg` (accuracy vs mean latency; small points are individual R3 seeds).

## Conclusion

**Classification: Evidence favoring Nearest 1-shot.**

1. **Does Random 1-shot significantly outperform Zero-shot?** R3 − S1 = +7.10 pp [+5.70, +8.43] pp — yes; the interval excludes zero; every seed: R3 − S1 between +6.57 pp and +7.61 pp.
2. **Does Nearest 1-shot outperform Random 1-shot?** R1 − R3 = +3.26 pp [+1.50, +4.86] pp; R1 is ahead on 10/10 seeds. By kind: choice -0.28 pp [-1.42, +0.88] pp; noul +11.61 pp [+8.61, +13.56] pp; score +2.50 pp [-1.19, +6.25] pp. In noul the nearest demonstration carries the query's ground-truth label in 83.49% of queries vs 65.10% for a random one, and the model copies the demonstration label in 89.99% (R1) / 77.94% (R3) of predictions, so the selection effect there coincides largely with label agreement. Against the length-matched control, R1 − R3-L = +3.29 pp [+1.87, +4.56] pp (R3-L − R3 = -0.04 pp [-0.65, +0.58] pp): the nearest advantage survives when demonstration length is matched, so it is not a length artefact.
3. **How much of the R1 improvement is due to selection?** 31.4% of R1 − S1 (16.8% – 43.4%); the remaining 68.6% is obtained with an arbitrary eligible demonstration.
4. **Is the nearest-neighbour latency justified?** R1 costs +135.4 ms per query more than R3 (120.7 ms retrieval vs 0.108 ms random selection, plus decision-time differences from demonstration length) for +32.6 correct answers per 1,000 queries.
5. **Which strategy for deployment?** Measured trade-off: 55.05% at 440 ms (R1) vs 51.79% at 305 ms (R3) vs 44.69% at 211 ms (S1). Prefer **nearest 1-shot (R1)** wherever its retrieval cost fits the latency budget: it is the most accurate strategy and its advantage over random selection is statistically supported. The selection benefit is concentrated in noul. For choice and score nearest is not detectably better than random, so routing those kinds to random selection (no embedding, no search) is a candidate cost saving — to be evaluated on the tuning partition, not chosen here from held-out results. Where retrieval cannot be afforded, random 1-shot keeps 69% of R1's gain over zero-shot at negligible selection cost and is clearly preferable to zero-shot. This report changes no production default.

## Reproduction

```sh
cmake --build build --config Release --target pjev-r3 test_r3
./tests/bin/test_r3.exe
./bin/pjev-r3.exe --mode run --reference results/r1-2b/predictions.jsonl --split-assignments results/r2/predictions.jsonl --output results/r3-random-1shot
./bin/pjev-r3.exe --mode run --selection length-matched --output results/r3-length-matched-1shot
./bin/pjev-r3.exe --mode aggregate --output results/r3-random-1shot --control results/r3-length-matched-1shot
```

Outputs: `seed-<seed>/predictions.jsonl` (one record per query: selected exemplar, prediction, logits, tokens, latency, status), `selections.jsonl` (every (query, seed) draw, written before inference), `reference_check.jsonl`, `config.json`, `per_seed.csv`, `similarity.csv`, `latency.csv`, `aggregate.json`, `accuracy_latency.svg`, this file.

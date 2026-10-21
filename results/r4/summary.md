# R4 — Nearest 2-shot, kNN classification and task-aware routing

All numbers below are measured on the R2 held-out evaluation partition (n=5811) under a configuration that was frozen on the tuning partition (n=5863) before the held-out run. R3 had already reported held-out S1 / random / nearest results, so R4's design is motivated by known held-out outcomes: the confirmatory tests below are pre-registered in the frozen file, but they are not a completely independent confirmation of the R3 pattern.

## A. Reproducibility

| Item | Value |
|---|---|
| Frozen configuration | `frozen_policy.json`, sha256 `22556532d51cc3a7ccb85156b57c2dbdfadd7979ab3cf3ef42140ec22300c327` (created 2026-10-20T18:33:32) |
| Inference model | models/bonsai2_2b.gguf (sha256 `36061ad406df8a4903a6db005be350c50fabfaf717f9ef0acf4b76adcd9929fa`) |
| Embedding model | models/nomic-embed-text-v1.5.gguf (fingerprint `3e24342164b3d949`) |
| Exemplar DB | models/examplar.db (schema 4, read-only) |
| Queries | datasets/validation.jsonl (sha256 `ce5d18132b4967ec5516a07b41700babeebef43b5d3dcc63ea62f01fcb4767cc`) |
| Partition | R2 split, recomputed and checked against `results/r2/predictions.jsonl` (sha256 `d1bb738a2e661417…`): tuning 5863, evaluation 5811, disagreements 0, group overlap 0 |
| Reference | results/r1-2b/predictions.jsonl (sha256 `336c0cd757f48245…`, unchanged) |
| Decision | letters, auto layout, single order, no calibration, candidate-logit argmax; n_ctx 4096 |
| Prompt | r1-demo-turn-v1; 2-shot = two demonstration turns then the S1 query |
| Random 1-shot | seed 42 (R3's first seed), XXH64 draw from R1's eligible pool |
| Bootstrap | 2000 cluster replicates over group_id, seed 20261020 |
| Runs | tuning 2026-10-20T15:06:41 (12305 s), evaluation 2026-10-20T18:53:44 (13406 s) |

**Integrity checks (evaluation run).** S1 reproduced the reference logits and token counts on 5811/5811 queries; R1's nearest exemplar on 5811/5811 and its logits on 5811/5811 (the 1-shot decision now runs through the generalised few-shot prompt builder); random 1-shot reproduced R3 seed 42 (exemplar and logits) on 5811/5811. Retrieval failures 0; neighbour lists shorter than five 0; top-K search ranking a different exemplar first than R1's top-1 search 96; content duplicates skipped 0; neighbours whose content equals the query 0; queries rejected as training records 0. Every exemplar is re-checked at load (training allowlist, one-hot, same kind and level count, not the query).

Condition status (evaluation run): r1 {ok 5811} random_1shot {ok 5811} s1 {ok 5811} two_shot_a {ok 5811} two_shot_b {ok 5811}. Context-limit failures by strategy and kind: none. Nothing is excluded: a failed decision counts as incorrect.

## Overview (held-out)

| Strategy | Accuracy | choice | noul | score | Mean ms | P50 | P95 | P99 | LLM evals | Embeddings | Prompt tokens | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S1 zero-shot | 44.69% | 41.49% | 63.41% | 11.87% | 263.5 | 188.8 | 443.1 | 713.2 | 1.00 | 0.00 | 343 | 0 |
| Random 1-shot (seed 42) | 51.82% | 43.19% | 79.11% | 30.04% | 380.4 | 401.4 | 682.0 | 767.2 | 1.00 | 0.00 | 679 | 0 |
| R1 nearest 1-shot | 55.05% | 42.92% | 90.62% | 32.55% | 519.9 | 314.7 | 1048.5 | 1133.9 | 1.00 | 1.00 | 691 | 0 |
| Nearest 2-shot (order A, frozen) | 55.31% | 45.13% | 90.68% | 20.50% | 620.3 | 474.3 | 1201.6 | 1394.8 | 1.00 | 1.00 | 1037 | 0 |
| Nearest 2-shot (order B, not selected) | 55.33% | 45.10% | 90.49% | 21.40% | 618.9 | 474.6 | 1195.6 | 1393.3 | 1.00 | 1.00 | 1037 | 0 |
| kNN-1 uniform | 69.74% (noul+score) | – | 83.49% | 30.22% | 97.6 | 100.5 | 144.9 | 151.3 | 0.00 | 1.00 | 0 | 0 |
| kNN-3 weighted (tau 0.005) (frozen) | 72.06% (noul+score) | – | 85.87% | 32.37% | 83.5 | 80.3 | 104.7 | 111.1 | 0.00 | 1.00 | 0 | 0 |
| kNN-5 weighted (tau 0.005) (frozen) | 73.87% (noul+score) | – | 87.12% | 35.79% | 83.6 | 80.3 | 104.7 | 111.1 | 0.00 | 1.00 | 0 | 0 |
| A_static_nearest_1shot | 55.05% | 42.92% | 90.62% | 32.55% | 519.9 | 314.7 | 1048.5 | 1133.9 | 1.00 | 1.00 | 691 | 0 |
| B_task_only@A_latency | 56.75% | 45.13% | 90.62% | 35.79% | 495.0 | 283.9 | 1166.8 | 1319.4 | 0.90 | 1.00 | 754 | 0 |
| B_task_only@random_latency | 55.55% | 43.19% | 90.68% | 35.79% | 376.1 | 394.6 | 945.6 | 1051.6 | 0.90 | 0.37 | 612 | 0 |
| B_task_only@zero_shot_latency | 54.57% | 43.19% | 87.12% | 35.79% | 221.3 | 191.9 | 467.1 | 664.3 | 0.63 | 0.37 | 314 | 0 |
| B_task_only@A_accuracy | 55.53% | 43.19% | 90.62% | 35.79% | 341.9 | 281.9 | 758.9 | 846.6 | 0.90 | 0.37 | 513 | 0 |
| C_adaptive@A_latency | 56.58% | 45.13% | 89.99% | 35.79% | 493.4 | 279.9 | 1173.1 | 1337.9 | 0.82 | 1.00 | 785 | 0 |
| C_adaptive@random_latency | 55.41% | 43.13% | 90.31% | 35.79% | 355.9 | 181.3 | 991.5 | 1137.0 | 0.99 | 0.37 | 499 | 0 |
| C_adaptive@zero_shot_latency | 54.57% | 43.19% | 87.12% | 35.79% | 221.3 | 191.9 | 467.1 | 664.3 | 0.63 | 0.37 | 314 | 0 |
| C_adaptive@A_accuracy | 55.43% | 43.19% | 90.24% | 35.79% | 261.8 | 196.5 | 688.7 | 781.6 | 0.71 | 0.37 | 384 | 0 |

kNN rows cover noul and score only (choice is unsupported; see C). Latencies are composed from per-query measured stages along each strategy's execution plan (see Latency). Failed = decisions that did not complete (counted as incorrect); excluded queries: 0.

Rescue / harm relative to S1 (rescue = S1 wrong and strategy right; net = rescue − harm):

| Strategy | choice | noul | score | all |
|---|---|---|---|---|
| Random 1-shot (seed 42) | 442 / 380 (+62) | 513 / 262 (+251) | 144 / 43 (+101) | 1099 / 685 (+414) |
| R1 nearest 1-shot | 458 / 406 (+52) | 532 / 97 (+435) | 149 / 34 (+115) | 1139 / 537 (+602) |
| Nearest 2-shot (order A, frozen) | 524 / 391 (+133) | 541 / 105 (+436) | 85 / 37 (+48) | 1150 / 533 (+617) |
| Nearest 2-shot (order B, not selected) | 526 / 394 (+132) | 540 / 107 (+433) | 93 / 40 (+53) | 1159 / 541 (+618) |
| kNN-1 uniform | – | 487 / 166 (+321) | 153 / 51 (+102) | – |
| kNN-3 weighted (tau 0.005) (frozen) | – | 519 / 160 (+359) | 162 / 48 (+114) | – |
| kNN-5 weighted (tau 0.005) (frozen) | – | 534 / 155 (+379) | 183 / 50 (+133) | – |
| A_static_nearest_1shot | 458 / 406 (+52) | 532 / 97 (+435) | 149 / 34 (+115) | 1139 / 537 (+602) |
| B_task_only@A_latency | 524 / 391 (+133) | 532 / 97 (+435) | 183 / 50 (+133) | 1239 / 538 (+701) |
| B_task_only@random_latency | 442 / 380 (+62) | 541 / 105 (+436) | 183 / 50 (+133) | 1166 / 535 (+631) |
| B_task_only@zero_shot_latency | 442 / 380 (+62) | 534 / 155 (+379) | 183 / 50 (+133) | 1159 / 585 (+574) |
| B_task_only@A_accuracy | 442 / 380 (+62) | 532 / 97 (+435) | 183 / 50 (+133) | 1157 / 527 (+630) |
| C_adaptive@A_latency | 524 / 391 (+133) | 542 / 117 (+425) | 183 / 50 (+133) | 1249 / 558 (+691) |
| C_adaptive@random_latency | 230 / 170 (+60) | 543 / 113 (+430) | 183 / 50 (+133) | 956 / 333 (+623) |
| C_adaptive@zero_shot_latency | 442 / 380 (+62) | 534 / 155 (+379) | 183 / 50 (+133) | 1159 / 585 (+574) |
| C_adaptive@A_accuracy | 442 / 380 (+62) | 534 / 105 (+429) | 183 / 50 (+133) | 1159 / 535 (+624) |

## Confirmatory comparisons (pre-registered)

| # | Comparison | Scope | n | Acc a | Acc b | Diff | 95% CI | Supported |
|---|---|---|---:|---:|---:|---:|---|---|
| 1 | nearest_2shot − nearest_1shot | all | 5811 | 55.31% | 55.05% | +0.26 pp | [-0.59, +1.11] pp | no (CI includes 0) |
| 2 | nearest_2shot − zero_shot | all | 5811 | 55.31% | 44.69% | +10.62 pp | [+8.90, +12.24] pp | yes (a better) |
| 3 | knn_1 − nearest_1shot | noul | 1599 | 83.49% | 90.62% | -7.13 pp | [-8.99, -5.55] pp | yes (b better) |
| 4 | knn_3 − knn_1 | noul | 1599 | 85.87% | 83.49% | +2.38 pp | [+1.12, +3.82] pp | yes (a better) |
| 5 | knn_5 − knn_1 | noul | 1599 | 87.12% | 83.49% | +3.63 pp | [+2.19, +5.63] pp | yes (a better) |
| 6 | B_task_only@A_latency − A_static_nearest_1shot | all | 5811 | 56.75% | 55.05% | +1.70 pp | [+0.85, +2.53] pp | yes (a better) |
| 7 | C_adaptive@A_latency − A_static_nearest_1shot | all | 5811 | 56.58% | 55.05% | +1.53 pp | [+0.66, +2.41] pp | yes (a better) |
| 8 | C_adaptive@A_latency − B_task_only@A_latency | all | 5811 | 56.58% | 56.75% | -0.17 pp | [-0.41, +0.08] pp | no (CI includes 0) |

"Supported" means the 95% cluster-bootstrap interval excludes zero; an interval that includes zero is reported as inconclusive, never as equivalence.

## B. Nearest 2-shot

Order A was selected on the tuning partition (second-nearest, nearest, query; tuning correct A 2895 vs B 2883 of 5863).

| Kind | n | S1 | R1 1-shot | 2-shot (selected) | 2-shot (other order) | 2-shot − 1-shot [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| choice | 3656 | 41.49% | 42.92% | 45.13% | 45.10% | +2.22 pp [+1.08, +3.34] pp |
| noul | 1599 | 63.41% | 90.62% | 90.68% | 90.49% | +0.06 pp [-0.83, +1.10] pp |
| score | 556 | 11.87% | 32.55% | 20.50% | 21.40% | -12.05 pp [-15.68, -8.48] pp |
| all | 5811 | 44.69% | 55.05% | 55.31% | 55.33% | +0.26 pp [-0.59, +1.11] pp |

Order effect (exploratory, selected − other): -0.02 pp [-0.76, +0.69] pp.

Cost: 2-shot evaluates 1037 prompt tokens per query vs 691 for 1-shot (+346, +50%) and takes 620.3 ms vs 519.9 ms per query (+100.4 ms; its top-10 search is cheaper than R1's top-1 search, see Latency). Accuracy change: +0.26 pp [-0.59, +1.11] pp = +2.6 correct answers per 1,000 queries. The difference to 1-shot is not statistically supported; the extra prompt cost is therefore not justified by this evidence. Versus zero-shot: +10.62 pp [+8.90, +12.24] pp.

## C. kNN classification (no inference LLM)

Label mapping by semantic identity: noul maps the exemplar's answer text (no/yes) to the query option with the same text; score maps the exemplar's answer level index to the same level (level counts match by R1's filter). **choice is unsupported**: options are per-task item identifiers, so a letter or position carries no shared meaning. The query's ground truth is never read before scoring.

Frozen voting (selected on tuning): kNN-3 = kNN-3 weighted (tau 0.005), kNN-5 = kNN-5 weighted (tau 0.005). Weights w = exp((s − s_max)/τ) on estimated cosine; ties → label of the best-ranked neighbour.

| Kind | n | S1 | R1 1-shot | kNN-1 | kNN-3 | kNN-5 | kNN-3 uniform / weighted | kNN-5 uniform / weighted |
|---|---:|---:|---:|---:|---:|---:|---|---|
| noul | 1599 | 63.41% | 90.62% | 83.49% | 85.87% | 87.12% | 85.43% / 85.87% | 85.99% / 87.12% |
| score | 556 | 11.87% | 32.55% | 30.22% | 32.37% | 35.79% | 32.37% / 32.37% | 36.69% / 35.79% |
| noul+score | 2155 | 50.12% | 75.64% | 69.74% | 72.06% | 73.87% | 71.74% / 72.06% | 73.27% / 73.87% |

Diagnostics (held-out):

| Kind | Mean nearest sim | kNN-1 = R1 prediction | R1 = nearest demo label | kNN-1 ✓ R1 ✗ | R1 ✓ kNN-1 ✗ | Top-3 consistency | Top-3 unanimous | Top-5 consistency | Top-5 unanimous | kNN-1 rescue / harm vs S1 | R1 rescue / harm vs S1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| noul | 0.9571 | 89.99% | 89.99% | 23 | 137 | 90.60% | 71.79% | 86.60% | 53.10% | 487 / 166 | 532 / 97 |
| score | 0.9489 | 90.47% | 90.47% | 9 | 22 | 59.47% | 13.31% | 51.98% | 2.70% | 153 / 51 | 149 / 34 |

Consistency = share of the top-k mapped labels that agree with the majority label (mean over queries).

**Label transfer vs LLM decision (noul).** R1 predicts the nearest demonstration's label on 1439 queries (91.17% correct, which is exactly kNN-1's accuracy on them) and departs from it on 160 queries, where R1 is correct 137 times and the demonstration label (= kNN-1) 23 times. The LLM's contribution beyond label transfer is therefore confined to those 160 departures: net +114 correct answers for R1 relative to kNN-1.

Accuracy by nearest-neighbour similarity (exploratory; quintiles of the held-out noul+score similarities):

| Bin | Similarity | n | S1 | R1 | kNN-1 | kNN-5 (frozen) |
|---|---|---:|---:|---:|---:|---:|
| 1 | min – 0.942 | 461 | 29.72% | 58.79% | 49.02% | 58.79% |
| 2 | 0.942 – 0.955 | 351 | 44.16% | 71.23% | 61.54% | 70.37% |
| 3 | 0.955 – 0.963 | 513 | 44.83% | 70.57% | 66.67% | 70.76% |
| 4 | 0.963 – 0.969 | 373 | 56.03% | 81.50% | 77.48% | 75.60% |
| 5 | 0.969 – max | 457 | 76.37% | 96.94% | 94.09% | 93.87% |

choice (diagnostic only): the nearest exemplar's answer text occurs among the query's options for 2636 of 3656 queries (72.10%), and is the correct option for 1132 of them (30.96% of all choice queries). Text identity is therefore not a usable kNN label mapping for choice; this diagnostic was not used by any policy.

kNN cost on the same noul+score queries: kNN-1 97.6 ms, kNN-3 83.5 ms, kNN-5 83.6 ms (no LLM evaluation) vs R1 584.0 ms; see Latency for the stages.

## D. Task-aware routing

Finalists were chosen on the tuning partition from 18 task-only and 775 adaptive per-kind rules (all non-dominated combinations), using the tuning partition's measured latencies. Tuning frontier: `pareto_frontier_tuning.csv`; every rule: `routing/tuning_policies.csv`.

| Policy | choice | noul | score | Tuning acc / ms | Held-out acc | Mean ms | P50 | P95 | P99 | LLM | Emb | Tokens | Failed |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_static_nearest_1shot (confirmatory) | nearest_1shot | nearest_1shot | nearest_1shot | 50.42% / 488 | 55.05% | 519.9 | 314.7 | 1048.5 | 1133.9 | 1.00 | 1.00 | 691 | 0 |
| B_task_only@A_latency (confirmatory) | nearest_2shot_a | nearest_1shot | knn_5 | 51.20% / 464 | 56.75% | 495.0 | 283.9 | 1166.8 | 1319.4 | 0.90 | 1.00 | 754 | 0 |
| B_task_only@random_latency | random_1shot | nearest_2shot_a | knn_5 | 50.79% / 329 | 55.55% | 376.1 | 394.6 | 945.6 | 1051.6 | 0.90 | 0.37 | 612 | 0 |
| B_task_only@zero_shot_latency | random_1shot | knn_5 | knn_5 | 50.08% / 215 | 54.57% | 221.3 | 191.9 | 467.1 | 664.3 | 0.63 | 0.37 | 314 | 0 |
| B_task_only@A_accuracy | random_1shot | nearest_1shot | knn_5 | 50.64% / 296 | 55.53% | 341.9 | 281.9 | 758.9 | 846.6 | 0.90 | 0.37 | 513 | 0 |
| C_adaptive@A_latency (confirmatory) | nearest_2shot_a | retrieve; if nearest sim >= 0.968011 -> knn_1, else nearest_2shot_a | knn_5 | 51.30% / 481 | 56.58% | 493.4 | 279.9 | 1173.1 | 1337.9 | 0.82 | 1.00 | 785 | 0 |
| C_adaptive@random_latency | S1; if S1 margin >= 1.07269 keep S1, else random_1shot | retrieve; if nearest sim >= 0.898674 and top-5 labels unanimous -> knn_5, else nearest_2shot_a | knn_5 | 51.07% / 318 | 55.41% | 355.9 | 181.3 | 991.5 | 1137.0 | 0.99 | 0.37 | 499 | 0 |
| C_adaptive@zero_shot_latency | random_1shot | knn_5 | knn_5 | 50.08% / 215 | 54.57% | 221.3 | 191.9 | 467.1 | 664.3 | 0.63 | 0.37 | 314 | 0 |
| C_adaptive@A_accuracy | random_1shot | retrieve; if nearest sim >= 0.898674 and top-3 labels unanimous -> knn_3, else nearest_1shot | knn_5 | 50.45% / 238 | 55.43% | 261.8 | 196.5 | 688.7 | 781.6 | 0.71 | 0.37 | 384 | 0 |

Routing frequency (held-out; share of each kind's queries answered by each terminal action):

- **A_static_nearest_1shot**: choice {nearest_1shot 100%} noul {nearest_1shot 100%} score {nearest_1shot 100%}
- **B_task_only@A_latency**: choice {nearest_2shot_a 100%} noul {nearest_1shot 100%} score {knn_5 100%}
- **B_task_only@random_latency**: choice {random_1shot 100%} noul {nearest_2shot_a 100%} score {knn_5 100%}
- **B_task_only@zero_shot_latency**: choice {random_1shot 100%} noul {knn_5 100%} score {knn_5 100%}
- **B_task_only@A_accuracy**: choice {random_1shot 100%} noul {nearest_1shot 100%} score {knn_5 100%}
- **C_adaptive@A_latency**: choice {nearest_2shot_a 100%} noul {knn_1 32%, nearest_2shot_a 68%} score {knn_5 100%}
- **C_adaptive@random_latency**: choice {random_1shot 37%, zero_shot 63%} noul {knn_5 53%, nearest_2shot_a 47%} score {knn_5 100%}
- **C_adaptive@zero_shot_latency**: choice {random_1shot 100%} noul {knn_5 100%} score {knn_5 100%}
- **C_adaptive@A_accuracy**: choice {random_1shot 100%} noul {knn_3 72%, nearest_1shot 28%} score {knn_5 100%}

Held-out accuracy/latency points (`pareto_frontier.csv`, `accuracy_latency.svg`); non-dominated: B_task_only@A_latency (56.75%, 495 ms); B_task_only@random_latency (55.55%, 376 ms); B_task_only@zero_shot_latency (54.57%, 221 ms); B_task_only@A_accuracy (55.53%, 342 ms); C_adaptive@A_latency (56.58%, 493 ms); C_adaptive@A_accuracy (55.43%, 262 ms);

Which kinds benefit from retrieval (held-out, exploratory per-kind differences):

| Kind | R1 − S1 | random − S1 | R1 − random | kNN-1 − S1 |
|---|---|---|---|---|
| choice | +1.42 pp [-0.11, +2.99] pp | +1.70 pp [+0.16, +3.22] pp | -0.27 pp [-1.75, +1.27] pp | unsupported |
| noul | +27.20 pp [+24.12, +31.18] pp | +15.70 pp [+11.23, +22.28] pp | +11.51 pp [+8.53, +13.43] pp | +20.08 pp [+16.81, +24.00] pp |
| score | +20.68 pp [+16.30, +24.78] pp | +18.17 pp [+13.88, +22.36] pp | +2.52 pp [-2.16, +7.20] pp | +18.35 pp [+13.49, +22.98] pp |

**Cost of adaptive decisions.** Every policy is costed along its execution plan with the work needed to decide included: a kNN or similarity gate always pays the query embedding and the vector search, even when it then answers from S1; an S1-margin gate always pays the S1 evaluation and, below the threshold, the full fallback action as well (two LLM evaluations). Mean LLM and embedding evaluations per query are in the table above.

**End-to-end check.** The finalists were executed live, plan by plan, on the first 300 held-out queries (`policy_timing.jsonl`):

| Policy | n | Live mean ms | Composed mean ms | Live / composed | Same prediction |
|---|---:|---:|---:|---:|---:|
| A_static_nearest_1shot | 300 | 680.1 | 926.7 | 0.73 | 100.00% |
| B_task_only@A_latency | 300 | 619.2 | 819.8 | 0.76 | 100.00% |
| B_task_only@random_latency | 300 | 380.0 | 528.4 | 0.72 | 100.00% |
| B_task_only@zero_shot_latency | 300 | 212.1 | 289.0 | 0.73 | 100.00% |
| B_task_only@A_accuracy | 300 | 337.5 | 474.3 | 0.71 | 100.00% |
| C_adaptive@A_latency | 300 | 657.1 | 873.9 | 0.75 | 100.00% |
| C_adaptive@random_latency | 300 | 490.8 | 674.7 | 0.73 | 100.00% |
| C_adaptive@zero_shot_latency | 300 | 215.6 | 289.0 | 0.75 | 100.00% |
| C_adaptive@A_accuracy | 300 | 298.5 | 416.6 | 0.72 | 100.00% |

## Latency

Measured per query in one process (models loaded once, untimed warm-up of every stage, sequential, every LLM decision from an empty KV cache; all actions of a query back to back). Means in ms over the held-out queries the action applies to:

| Action | Embedding | Search | Row fetch | LZ4 | Random draw | Prompt construction | LLM inference | Candidate scoring | Vote | Decision wall | Total mean | P50 | P95 | P99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| zero_shot | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.875 | 262.6 | 0.004 | 0.000 | 263.5 | 263.5 | 188.8 | 443.1 | 713.2 |
| random_1shot | 0.000 | 0.000 | 0.080 | 0.042 | 0.003 | 2.1 | 378.2 | 0.004 | 0.000 | 380.3 | 380.4 | 401.4 | 682.0 | 767.2 |
| nearest_1shot | 4.6 | 120.7 | 0.009 | 0.026 | 0.000 | 2.1 | 392.5 | 0.004 | 0.000 | 394.5 | 519.9 | 314.7 | 1048.5 | 1133.9 |
| nearest_2shot | 4.6 | 94.5 | 0.015 | 0.044 | 0.000 | 2.9 | 518.2 | 0.004 | 0.000 | 521.1 | 620.3 | 474.3 | 1201.6 | 1394.8 |
| nearest_2shot_other | 4.6 | 94.5 | 0.015 | 0.044 | 0.000 | 2.9 | 516.9 | 0.004 | 0.000 | 519.8 | 618.9 | 474.6 | 1195.6 | 1393.3 |
| knn_1 | 5.2 | 92.4 | 0.009 | 0.031 | 0.000 | 0.000 | 0.000 | 0.000 | 0.002 | 0.000 | 97.6 | 100.5 | 144.9 | 151.3 |
| knn_3 | 5.2 | 78.3 | 0.020 | 0.073 | 0.000 | 0.000 | 0.000 | 0.000 | 0.002 | 0.000 | 83.5 | 80.3 | 104.7 | 111.1 |
| knn_5 | 5.2 | 78.3 | 0.030 | 0.111 | 0.000 | 0.000 | 0.000 | 0.000 | 0.002 | 0.000 | 83.6 | 80.3 | 104.7 | 111.1 |

Decision wall = prompt construction + LLM inference + scoring + call overhead. nearest_1shot uses R1's own top-1 search, which starts the eligibility-filtered KNN at k=1 and doubles until an eligible row appears; the top-10 search used by 2-shot and kNN-3/5 starts at k=10 and needs fewer rounds, which is why it is cheaper. That is a property of R1's retrieval loop, not of the strategies, and R1's mechanism was not changed. Offline evaluation reuses each query's single measurement of every stage; a deployed policy performs only the stages of its plan, which the end-to-end check above executes for real. These are local measurements on one machine and say nothing about production latency. The tuning run shared the machine with occasional compilation of the analysis tool; the evaluation run had no concurrent work.

## E. Deployment implications

**Recommendation: Adopt a simple task-only routing policy.** (B and C both qualify; C is not supported as better than B, so the simpler policy is preferred.)

Pre-registered decision rule (frozen file): a policy qualifies when (i) its accuracy difference to A has a CI lower bound ≥ −1.0 pp and its mean latency is at least 10% below A's (519.9 ms), or (ii) the CI lower bound is above 0 at no more than A's latency. B (B_task_only@A_latency): +1.70 pp [+0.85, +2.53] pp, 495.0 ms → qualifies. C (C_adaptive@A_latency): +1.53 pp [+0.66, +2.41] pp, 493.4 ms → qualifies. C − B: -0.17 pp [-0.41, +0.08] pp. kNN-1 − R1 on noul: -7.13 pp [-8.99, -5.55] pp.

No production default was changed; R4 adds an N-demonstration entry point (`decide_with_demonstrations`) used only by this tool.

### Confirmed observations

- nearest_2shot − nearest_1shot (all): +0.26 pp [-0.59, +1.11] pp — inconclusive (CI includes 0).
- nearest_2shot − zero_shot (all): +10.62 pp [+8.90, +12.24] pp — statistically supported.
- knn_1 − nearest_1shot (noul): -7.13 pp [-8.99, -5.55] pp — statistically supported.
- knn_3 − knn_1 (noul): +2.38 pp [+1.12, +3.82] pp — statistically supported.
- knn_5 − knn_1 (noul): +3.63 pp [+2.19, +5.63] pp — statistically supported.
- B_task_only@A_latency − A_static_nearest_1shot (all): +1.70 pp [+0.85, +2.53] pp — statistically supported.
- C_adaptive@A_latency − A_static_nearest_1shot (all): +1.53 pp [+0.66, +2.41] pp — statistically supported.
- C_adaptive@A_latency − B_task_only@A_latency (all): -0.17 pp [-0.41, +0.08] pp — inconclusive (CI includes 0).

### Exploratory findings

- Per-kind effects, the non-selected 2-shot order, voting variants, similarity bins and the non-confirmatory finalists (all in `pairwise_comparisons.csv`, type = exploratory) were not pre-registered as tests.
- Order effect (selected − other): -0.02 pp [-0.76, +0.69] pp.
- B_task_only@random_latency − A: +0.50 pp [-0.54, +1.59] pp at 376.1 ms.
- B_task_only@zero_shot_latency − A: -0.48 pp [-1.63, +0.74] pp at 221.3 ms.
- B_task_only@A_accuracy − A: +0.48 pp [-0.58, +1.53] pp at 341.9 ms.
- C_adaptive@random_latency − A: +0.36 pp [-0.65, +1.40] pp at 355.9 ms.
- C_adaptive@zero_shot_latency − A: -0.48 pp [-1.63, +0.74] pp at 221.3 ms.
- C_adaptive@A_accuracy − A: +0.38 pp [-0.68, +1.47] pp at 261.8 ms.

### Unresolved questions

- The internal tuning/evaluation split is the only held-out evidence; R3's held-out results were known when R4 was designed.
- Whether kNN's label transfer generalises to task families not represented in the exemplar bank (both partitions share the bank).
- choice has no kNN mapping; whether a semantic candidate representation could support one was not studied.
- Latency is local and single-machine; R1's top-1 retrieval loop cost depends on how many eligible rows sit near the query.

## Commands

```sh
cmake --build build --config Release --target pjev-r4 test_r4
./tests/bin/test_r4.exe
./bin/pjev-r4.exe --mode run --partition tuning
./bin/pjev-r4.exe --mode tune
./bin/pjev-r4.exe --mode run --partition evaluation
./bin/pjev-r4.exe --mode policy-timing --limit 300
./bin/pjev-r4.exe --mode evaluate
```

Files: `frozen_policy.json` / `.sha256`, `tuning/` and `evaluation/` (per-query measurements: `queries.jsonl`, `run_config.json`), `nearest-2shot/`, `knn/`, `routing/` (`predictions.jsonl`, `metrics.json`; routing also `tuning_policies.csv`, `selected_policies.json`), `latency.csv`, `pairwise_comparisons.csv`, `task_metrics.csv`, `rescue_harm.csv`, `pareto_frontier.csv`, `pareto_frontier_tuning.csv`, `policy_timing.jsonl`, `accuracy_latency.svg`, `task_accuracy.svg`.

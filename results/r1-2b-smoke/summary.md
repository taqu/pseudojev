# R1 — Retrieved 1-Shot Inference Evaluation

> **PRELIMINARY** — this run used a subset (--limit / --kind / --seed). Do not treat these numbers as the full validation result.

Corpus: **open-jev-validation**  
Run date: 2026-10-16T19:42:26

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
| limit | 12 |
| seed | null |
| leakage_policy | demonstrations restricted to IDs whose split is 'train' in train_path; the query's own ID is rejected; query text never includes the target |

## Coverage

- Evaluated queries: 12
- Paired (S1 ok and R1 1-shot ok): **12** (100.00%)
- S1 status: ok=12
- Retrieval status: ok=12
- R1 status: ok=12

Unpaired queries are excluded from every paired metric below; no zero-shot fallback is counted as a 1-shot result.

## Headline (paired)

| Metric | Value |
|---|---|
| Paired examples | 12 |
| S1 accuracy | 33.33% (4) |
| R1 accuracy | 33.33% (4) |
| Difference | +0.00 pp |
| 95% CI (paired Wald) | [-23.10, +23.10] pp |
| Rescue / Harm / Net | 1 / 1 / 0 |
| McNemar χ² (cc) p | 1.000000 |
| Exact binomial p | 1.000000 |

Check: (rescue − harm) / n = +0.00 pp = R1 acc − S1 acc.

## Accuracy by group

| Group | n | S1 acc | R1 acc | Diff (pp) | 95% CI (pp) | Rescue | Harm | Net | McNemar p |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| overall | 12 | 33.33% (4) | 33.33% (4) | +0.00 | [-23.10, +23.10] | 1 | 1 | 0 | 1.0000 |
| kind:choice | 6 | 16.67% (1) | 0.00% (0) | -16.67 | [-46.49, +13.15] | 0 | 1 | -1 | 1.0000 |
| kind:noul | 3 | 100.00% (3) | 100.00% (3) | +0.00 | [+0.00, +0.00] | 0 | 0 | 0 | 1.0000 |
| kind:score | 3 | 0.00% (0) | 33.33% (1) | +33.33 | [-20.01, +86.68] | 1 | 0 | 1 | 1.0000 |
| score_levels:5 | 3 | 0.00% (0) | 33.33% (1) | +33.33 | [-20.01, +86.68] | 1 | 0 | 1 | 1.0000 |
| choice_options:7 | 4 | 25.00% (1) | 0.00% (0) | -25.00 | [-67.44, +17.44] | 0 | 1 | -1 | 1.0000 |
| choice_options:8 | 2 | 0.00% (0) | 0.00% (0) | +0.00 | [+0.00, +0.00] | 0 | 0 | 0 | 1.0000 |

## Rescue / harm transitions

### Overall

| | R1 correct | R1 incorrect |
|---|---:|---:|
| S1 correct | 3 (both correct) | 1 (harm) |
| S1 incorrect | 1 (rescue) | 7 (both incorrect) |

### choice

| | R1 correct | R1 incorrect |
|---|---:|---:|
| S1 correct | 0 (both correct) | 1 (harm) |
| S1 incorrect | 0 (rescue) | 5 (both incorrect) |

### noul

| | R1 correct | R1 incorrect |
|---|---:|---:|
| S1 correct | 3 (both correct) | 0 (harm) |
| S1 incorrect | 0 (rescue) | 0 (both incorrect) |

### score

| | R1 correct | R1 incorrect |
|---|---:|---:|
| S1 correct | 0 (both correct) | 0 (harm) |
| S1 incorrect | 1 (rescue) | 2 (both incorrect) |

### score, 5 levels

| | R1 correct | R1 incorrect |
|---|---:|---:|
| S1 correct | 0 (both correct) | 0 (harm) |
| S1 incorrect | 1 (rescue) | 2 (both incorrect) |

## Similarity versus outcome

The index stores sign-quantized (bit) vectors, so sqlite-vec returns a **Hamming distance** over 768 bits, not a float cosine distance. Estimated cosine = cos(π·h/768) (sign-quantization angle estimate). Bins are observed quantiles of the Top-1 Hamming distance (closest first); equal distances share a bin. They are descriptive only — no gate.

### All kinds

| Bin | Hamming | Est. cosine | n | S1 acc | R1 acc | Rescue | Harm | Net change (pp) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| q1 | 78–79 | 0.948–0.950 | 3 | 0.00% | 33.33% | 1 | 0 | +33.33 |
| q2 | 84–84 | 0.942–0.942 | 4 | 25.00% | 0.00% | 0 | 1 | -25.00 |
| q3 | 85–85 | 0.940–0.940 | 2 | 0.00% | 0.00% | 0 | 0 | +0.00 |
| q4 | 92–92 | 0.930–0.930 | 2 | 100.00% | 100.00% | 0 | 0 | +0.00 |
| q5 | 93–93 | 0.929–0.929 | 1 | 100.00% | 100.00% | 0 | 0 | +0.00 |

### choice

| Bin | Hamming | Est. cosine | n | S1 acc | R1 acc | Rescue | Harm | Net change (pp) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| q1 | 84–84 | 0.942–0.942 | 4 | 25.00% | 0.00% | 0 | 1 | -25.00 |
| q2 | 85–85 | 0.940–0.940 | 2 | 0.00% | 0.00% | 0 | 0 | +0.00 |

### noul

| Bin | Hamming | Est. cosine | n | S1 acc | R1 acc | Rescue | Harm | Net change (pp) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| q1 | 92–92 | 0.930–0.930 | 2 | 100.00% | 100.00% | 0 | 0 | +0.00 |
| q2 | 93–93 | 0.929–0.929 | 1 | 100.00% | 100.00% | 0 | 0 | +0.00 |

### score

| Bin | Hamming | Est. cosine | n | S1 acc | R1 acc | Rescue | Harm | Net change (pp) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| q1 | 78–78 | 0.950–0.950 | 1 | 0.00% | 0.00% | 0 | 0 | +0.00 |
| q2 | 79–79 | 0.948–0.948 | 2 | 0.00% | 50.00% | 1 | 0 | +50.00 |

## Retrieval relationships

| Kind | n | Same group | Same domain | Same family | Same score option texts |
|---|---:|---:|---:|---:|---:|
| all | 12 | 0.00% (0/12) | 100.00% (12/12) | 100.00% (12/12) | 100.00% (3/3) |
| choice | 6 | 0.00% (0/6) | 100.00% (6/6) | 100.00% (6/6) | n/a |
| noul | 3 | 0.00% (0/3) | 100.00% (3/3) | 100.00% (3/3) | n/a |
| score | 3 | 0.00% (0/3) | 100.00% (3/3) | 100.00% (3/3) | 100.00% (3/3) |

Outcome by domain / family relationship:

| Group | n | S1 acc | R1 acc | Diff (pp) | 95% CI (pp) | Rescue | Harm | Net | McNemar p |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| same_domain | 12 | 33.33% (4) | 33.33% (4) | +0.00 | [-23.10, +23.10] | 1 | 1 | 0 | 1.0000 |
| different_domain | 0 | 0.00% (0) | 0.00% (0) | +0.00 | [+0.00, +0.00] | 0 | 0 | 0 | 1.0000 |
| same_family | 12 | 33.33% (4) | 33.33% (4) | +0.00 | [-23.10, +23.10] | 1 | 1 | 0 | 1.0000 |
| different_family | 0 | 0.00% (0) | 0.00% (0) | +0.00 | [+0.00, +0.00] | 0 | 0 | 0 | 1.0000 |

## Retrieved answer-label bias

Label agreement compares candidate labels (A, B, ...). Semantic agreement compares the option text of the prediction with the demonstration's answer text; for noul and score the label and the semantic value coincide by construction (fixed false/true order, ordered levels), for choice they do not. Label agreement alone is not evidence of reasoning.

| Kind | n | Demo label = GT label | S1 = demo label | R1 = demo label | Moved toward demo label | of which rescue / harm / other | Moved away | R1 = demo answer text | S1 = demo answer text |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| all | 12 | 8.33% (1/12) | 0.00% (0/12) | 33.33% (4/12) | 4 | 1 / 0 / 3 | 0 | 25.00% (3/12) | 0.00% (0/12) |
| choice | 6 | 0.00% (0/6) | 0.00% (0/6) | 16.67% (1/6) | 1 | 0 / 0 / 1 | 0 | 0.00% (0/6) | 0.00% (0/6) |
| noul | 3 | 0.00% (0/3) | 0.00% (0/3) | 0.00% (0/3) | 0 | 0 / 0 / 0 | 0 | 0.00% (0/3) | 0.00% (0/3) |
| score | 3 | 33.33% (1/3) | 0.00% (0/3) | 100.00% (3/3) | 3 | 1 / 0 / 2 | 0 | 100.00% (3/3) | 0.00% (0/3) |

### Label distributions (paired queries)

| Kind | Label | Demo answers | S1 predictions | R1 predictions | Ground truth |
|---|---|---:|---:|---:|---:|
| choice | A | 0 | 3 | 1 | 0 |
| choice | B | 0 | 1 | 2 | 0 |
| choice | C | 0 | 0 | 0 | 1 |
| choice | D | 6 | 0 | 1 | 0 |
| choice | E | 0 | 0 | 0 | 2 |
| choice | F | 0 | 0 | 0 | 1 |
| choice | G | 0 | 1 | 2 | 2 |
| choice | H | 0 | 1 | 0 | 0 |
| noul | A | 0 | 3 | 3 | 3 |
| noul | B | 3 | 0 | 0 | 0 |
| score | B | 0 | 3 | 0 | 0 |
| score | C | 3 | 0 | 3 | 1 |
| score | D | 0 | 0 | 0 | 2 |

## Margins

| Kind | S1 mean winner margin | R1 mean winner margin | S1 mean signed margin | R1 mean signed margin |
|---|---:|---:|---:|---:|
| all | 0.753 | 3.272 | -1.887 | -0.793 |
| choice | 0.739 | 1.881 | -2.471 | -2.914 |
| noul | 0.701 | 5.829 | 0.701 | 5.829 |
| score | 0.834 | 3.499 | -3.307 | -3.174 |

## Prompt tokens

Maximum context (n_ctx): 4096. Query tokens = S1 prompt tokens; exemplar tokens = R1 − S1 prompt tokens (demonstration turn).

| Kind | S1 mean | R1 mean | Extra mean | Extra median | Extra p95 | R1 max |
|---|---:|---:|---:|---:|---:|---:|
| all | 660.9 | 1321.2 | 660.3 | 661 | 673 | 1358 |
| choice | 664.3 | 1337.3 | 673.0 | 673 | 673 | 1354 |
| noul | 614.7 | 1263.7 | 649.0 | 649 | 649 | 1275 |
| score | 700.3 | 1346.7 | 646.3 | 647 | 647 | 1358 |

## Latency

S1 and R1 run sequentially per query in one process; both models are loaded once and one untimed warm-up embedding and decision precede the timed loop. Every decision evaluates its full prompt from an empty KV cache (no prefix reuse between queries or between S1 and R1). The R1 total includes query embedding, vector search, exemplar read, LZ4 decompression and the 1-shot decision; it is the per-request cost of R1 on its own, not the combined experiment time.

| Condition | Stage | Mean ms | Median ms | P95 ms |
|---|---|---:|---:|---:|
| s1 | prompt_construction | 1.90 | 1.79 | 2.41 |
| s1 | prompt_evaluation | 317.34 | 309.28 | 360.95 |
| s1 | answer_scoring | 0.01 | 0.01 | 0.01 |
| s1 | total | 319.26 | 311.30 | 362.67 |
| r1 | query_embedding | 6.66 | 5.57 | 12.73 |
| r1 | vector_search | 115.46 | 116.62 | 153.52 |
| r1 | exemplar_retrieval | 0.01 | 0.01 | 0.01 |
| r1 | lz4_decompression | 0.05 | 0.06 | 0.07 |
| r1 | prompt_construction | 2.87 | 2.73 | 3.43 |
| r1 | prompt_evaluation | 508.01 | 508.12 | 527.68 |
| r1 | answer_scoring | 0.00 | 0.00 | 0.01 |
| r1 | decision_total | 510.89 | 510.88 | 530.68 |
| r1 | total | 633.07 | 640.05 | 683.48 |

- R1 / S1 latency multiplier (mean): **1.98×**
- R1 / S1 latency multiplier (median): 2.06×
- Total evaluation wall time: 78.7 s (S1 decisions 3.8 s, R1 incl. retrieval 7.6 s)

## Context-limit failures

None.

## Main observations

- R1 − S1 = +0.00 pp over 12 paired queries (95% CI [-23.10, +23.10]); the interval includes zero — not conclusive.
- 1 rescues vs 1 harms (16.67% (2/12) of paired queries change correctness).
- S1 parity: S1 re-run after R1 on the first 12 queries — 0 mismatches.
- Exemplar DB holds 158898 records outside the training split; they are never used as demonstrations.
- 79 training records have no one-hot answer (soft target) and are not eligible exemplars.


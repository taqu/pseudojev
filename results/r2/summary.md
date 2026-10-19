# R2 — similarity-gated retrieved 1-shot (offline replay)

## Configuration

| Setting | Value |
|---|---|
| R1 input_path | datasets/validation.jsonl |
| R1 inference_model | models/bonsai2_2b.gguf |
| R1 embedding_model | models/nomic-embed-text-v1.5.gguf |
| R1 db_path | models/examplar.db |
| R1 train_path | datasets/train.jsonl |
| R1 prompt_format_version | r1-demo-turn-v1 |
| R1 n_ctx | 4096 |
| R1 run_timestamp | 2026-10-16T19:44:33 |
| R1 inference model sha256 | 36061ad406df8a4903a6db005be350c50fabfaf717f9ef0acf4b76adcd9929fa |
| Predictions | results/r1-2b/predictions.jsonl |
| Similarity | estimated cosine cos(pi·h/768) from the Top-1 Hamming distance h |
| Gate | similarity >= threshold (equality activates) |

## Data separation

No official calibration split exists in this repository (validation records are all labelled `validation`; `calibration/original.jsonl` is unrelated), so the R1 validation run is split into an **internal**, deterministic, group-disjoint tuning / evaluation partition:

- Unit: scenario family when present (whole families, alternating by sorted name), otherwise group_id (xxHash64 parity). Counterfactual variants share a group and never straddle.
- Tuning families: `browser_tools/idempotent_retry` `games/racing_overtake` `rag/policy_exception` `shell_history/checksum_verify` 
- Evaluation families: `contracts/governing_law_venue` `ir/access_schedule` `rubric_judge/trace_consistency` `support/damaged_order` 
- Groups in both partitions: 0; query records labelled train: 0
- Queries that are training records: 0
- Paired records: 11674 (tuning 5863, evaluation 5811). Excluded: none

## Threshold selection (tuning only)

Candidates: never activate (all S1) plus every distinct Top-1 similarity observed on the tuning partition (gate: similarity >= threshold). Objective: maximise paired R2 accuracy on tuning. Ties: lower activation count wins (never-activate counts as zero). Fixed before any held-out result was computed; held-out data never enters selection.

| Tuning n | Candidates | Selected | Tuning S1 | Tuning R1 | Tuning R2 | Tuning activation |
|---:|---:|---|---:|---:|---:|---:|
| 5863 | 84 | 0.90398929312344334 (Hamming <= 108.0) | 39.06% | 50.42% | 50.50% | 99.4% |

Tuning accuracy is optimistically biased (the threshold was fitted to it). Only the held-out rows below support an effectiveness claim.

## Held-out result at the frozen threshold

### Global threshold (primary)

| Kind | n | S1 | R1 | R2 | R2−S1 | R2−R1 | Activated | Rescue kept | Rescue lost | Harm kept | Harm prevented |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 5811 | 44.69% | 55.05% | 55.19% | +10.50 pp | +0.14 pp | 5467 (94.1%) | 1110 | 29 | 500 | 37 |
| choice | 3656 | 41.49% | 42.92% | 43.16% | +1.67 pp | +0.25 pp | 3313 (90.6%) | 430 | 28 | 369 | 37 |
| noul | 1599 | 63.41% | 90.62% | 90.56% | +27.14 pp | -0.06 pp | 1598 (99.9%) | 531 | 1 | 97 | 0 |
| score | 556 | 11.87% | 32.55% | 32.55% | +20.68 pp | +0.00 pp | 556 (100.0%) | 149 | 0 | 34 | 0 |

Paired uncertainty (held-out, 5811 queries, 1598 group clusters, 2000 cluster-bootstrap replicates):

| Comparison | Difference | Cluster bootstrap 95% | Wald / McNemar |
|---|---:|---|---|
| R2 − S1 | +10.50 pp | [+8.84 pp, +12.08 pp] | Wald 95% [+9.17, +11.82] pp, McNemar p=4.97e-52 (exact 2.69e-53) |
| R2 − R1 | +0.14 pp | [-0.13 pp, +0.40 pp] | Wald 95% [-0.14, +0.41] pp, McNemar p=0.389 (exact 0.389) |
| R1 − S1 | +10.36 pp | [+8.60 pp, +11.97 pp] | Wald 95% [+9.00, +11.71] pp, McNemar p=8.61e-49 (exact 7.53e-50) |

Gate on held-out: activated 5467 (94.1%), rejected 344; mean similarity activated 0.9521, rejected 0.8902. Queries without an eligible exemplar are excluded from the paired set (counted above) and would keep S1.

### Strategy comparison (held-out)

| Strategy | n | Accuracy | Activation |
|---|---:|---:|---:|
| all_s1 | 5811 | 44.69% | 0.0% |
| all_eligible_r1 | 5811 | 55.05% | 100.0% |
| global | 5811 | 55.19% | 94.1% |
| per_kind | 5811 | 55.22% | 91.7% |

Per-kind thresholds (secondary diagnostic, selected on tuning per kind): choice=0.9175 (tuning n=3855) noul=0.9040 (tuning n=1322) score=0.9109 (tuning n=686) 

### Per-kind thresholds (diagnostic)

| Kind | n | S1 | R1 | R2 | R2−S1 | R2−R1 | Activated | Rescue kept | Rescue lost | Harm kept | Harm prevented |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 5811 | 44.69% | 55.05% | 55.22% | +10.53 pp | +0.17 pp | 5329 (91.7%) | 1097 | 42 | 485 | 52 |
| choice | 3656 | 41.49% | 42.92% | 43.22% | +1.72 pp | +0.30 pp | 3176 (86.9%) | 417 | 41 | 354 | 52 |
| noul | 1599 | 63.41% | 90.62% | 90.56% | +27.14 pp | -0.06 pp | 1598 (99.9%) | 531 | 1 | 97 | 0 |
| score | 556 | 11.87% | 32.55% | 32.55% | +20.68 pp | +0.00 pp | 555 (99.8%) | 149 | 0 | 34 | 0 |

Score by level count (held-out, frozen global threshold):

| Levels | n | S1 | R1 | R2 | Activation |
|---:|---:|---:|---:|---:|---:|
| 3 | 36 | 66.67% | 91.67% | 91.67% | 100.0% |
| 5 | 520 | 8.08% | 28.46% | 28.46% | 100.0% |

## Threshold ranges

- Tuning: 4 of 83 distinct thresholds beat both S1 and R1 (thresholds 0.8987 – 0.9040).
- Held-out (exploratory only; not used for selection): 16 of 81 distinct thresholds beat both S1 and R1 (thresholds 0.8819 – 0.9175).

Full curves (accuracy, activation, rescue/harm, estimated latency vs threshold) are in `threshold_sweep.csv` (tuning) and `heldout_sweep_exploratory.csv`; plot columns threshold against r2_accuracy / activation_rate / rescue,harm / est_r2_latency_ms, and activation_rate against r2_accuracy for the trade-off.

## Similarity as a signal (held-out evaluation partition)

R1 rescue/harm by Top-1 similarity quintile (highest first). A useful gate needs net gain to fall with similarity.

| Quintile | Similarity range | n | Rescue | Harm | Net | R1−S1 |
|---|---|---:|---:|---:|---:|---:|
| Q1 | 0.9659 – 0.9879 | 1162 | 213 | 81 | 132 | +11.36 pp |
| Q2 | 0.9593 – 0.9659 | 1162 | 242 | 108 | 134 | +11.53 pp |
| Q3 | 0.9495 – 0.9593 | 1162 | 262 | 109 | 153 | +13.17 pp |
| Q4 | 0.9300 – 0.9495 | 1162 | 229 | 115 | 114 | +9.81 pp |
| Q5 | 0.8741 – 0.9300 | 1163 | 193 | 124 | 69 | +5.93 pp |

## Cost (estimated from R1 measurements, held-out evaluation partition)

| Model | Mean ms / query |
|---|---:|
| S1 only | 211.5 |
| Deployed R1 (retrieval + 1-shot, no S1 pass) | 440.0 |
| R2 (S1 + retrieval + activation × 1-shot) | 622.8 |
| Retrieval overhead (embed + search + fetch + LZ4) | 120.7 |

R2/S1 = 2.94×, R2/deployed R1 = 1.42×. These are predictions from per-query timings of the R1 harness (warm models, sequential, S1 pass counted once); the online gated run measures them directly.

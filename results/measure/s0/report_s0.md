# score: S0

## score / easy

- baseline: `results/measure/s0/s0_easy.json`

_No labeled score items in this split._

| metric | S0 |
| --- | ---: |
| labeled | 0 |
| eval_ms | 0 |
| prompt tokens | 0 |

Large error: |predicted - ground truth| >= 2. Calibrated metrics use T_score = n/a; raw metrics use T = 1. Brier is summed over levels.

### Verification

All stored aggregates match an independent recomputation from the items.

## score / original

- baseline: `results/measure/s0/s0_original.json`

| metric | S0 |
| --- | ---: |
| labeled | 12 |
| accuracy | 0.7500 |
| MAE | 0.4167 |
| expected-score MAE | 0.9108 |
| expected-score MAE (raw, T=1) | 0.3766 |
| QWK | 0.7327 |
| NLL | 1.2027 |
| Brier | 0.6523 |
| ECE | 0.4902 |
| NLL (raw, T=1) | 0.4308 |
| Brier (raw, T=1) | 0.2370 |
| ECE (raw, T=1) | 0.0923 |
| mean signed margin | 1.9534 |
| median signed margin | 1.9390 |
| p10 signed margin | -0.0761 |
| min signed margin | -0.5435 |
| mean expected-argmax distance | 1.1447 |
| mean expected-argmax distance (raw) | 0.4135 |
| error distance 0 | 9 |
| error distance 1 | 1 |
| error distance 2 | 2 |
| large error rate | 0.1667 |
| eval_ms | 5908 |
| prompt tokens | 1514 |
| evaluations per item | 1.00 |

Large error: |predicted - ground truth| >= 2. Calibrated metrics use T_score = 11.9603; raw metrics use T = 1. Brier is summed over levels.

**Candidates (S0)** — answer anchor `"<|im_end|>\n<|im_start|>assistant\n\n"`

| levels | labels | token ids | single token | separate after anchor | content-free prior logits |
| ---: | --- | --- | --- | --- | --- |
| 4 | 0 1 2 3 | 15 16 17 18 | yes | yes | -1.563 -3.179 -3.272 -5.240 |

### Verification

All stored aggregates match an independent recomputation from the items.

## score / hard

- baseline: `results/measure/s0/s0_hard.json`

| metric | S0 |
| --- | ---: |
| labeled | 6 |
| accuracy | 0.0000 |
| MAE | 1.8333 |
| expected-score MAE | 0.6081 |
| expected-score MAE (raw, T=1) | 1.0790 |
| QWK | -0.2105 |
| NLL | 1.4476 |
| Brier | 0.7703 |
| ECE | 0.2729 |
| NLL (raw, T=1) | 2.2618 |
| Brier (raw, T=1) | 1.1933 |
| ECE (raw, T=1) | 0.6156 |
| mean signed margin | -1.7618 |
| median signed margin | -1.6863 |
| p10 signed margin | -2.6227 |
| min signed margin | -3.4033 |
| mean expected-argmax distance | 1.5089 |
| mean expected-argmax distance (raw) | 0.7544 |
| error distance 0 | 0 |
| error distance 1 | 2 |
| error distance 2 | 3 |
| error distance 3 | 1 |
| large error rate | 0.6667 |
| eval_ms | 61579 |
| prompt tokens | 10078 |
| evaluations per item | 1.00 |

Large error: |predicted - ground truth| >= 2. Calibrated metrics use T_score = 11.9603; raw metrics use T = 1. Brier is summed over levels.

**Candidates (S0)** — answer anchor `"<|im_end|>\n<|im_start|>assistant\n\n"`

| levels | labels | token ids | single token | separate after anchor | content-free prior logits |
| ---: | --- | --- | --- | --- | --- |
| 4 | 0 1 2 3 | 15 16 17 18 | yes | yes | -1.563 -3.179 -3.272 -5.240 |
| 5 | 0 1 2 3 4 | 15 16 17 18 19 | yes | yes | -0.093 -2.148 -2.577 -4.448 -2.605 |

### Verification

All stored aggregates match an independent recomputation from the items.

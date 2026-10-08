# score: S0 digits vs S1 letters (raw, T = 1)

## score / easy

- baseline: `results/measure/s1/s0raw_easy.json`
- S1 letters: `results/measure/s1/s1raw_easy.json`

_No labeled score items in this split._

| metric | S0 digits | S1 letters | Δ S1 letters |
| --- | ---: | ---: | ---: |
| labeled | 0 | 0 | +0 |
| eval_ms | 0 | 0 | +0 |
| prompt tokens | 0 | 0 | +0 |

▲ better than baseline, ▼ worse.

Large error: |predicted - ground truth| >= 2. No calibration artifact was applied (T = 1), so calibrated and raw metrics coincide. Brier is summed over levels.

### S1 letters vs S0 digits

- matched items: 0; margin gain mean n/a, median n/a
- margin improved / degraded / unchanged: 0 / 0 / 0
- S0 digits correct → S1 letters wrong: 0
- S0 digits wrong → S1 letters correct: 0
- quality metrics better: 0, worse: 0
- latency: 0 ms vs 0 ms
- expected-score |error| change (reported): mean n/a, median n/a; improved / degraded / unchanged: 0 / 0 / 0 (negative change = better)
- expected-score |error| change (raw): mean n/a, median n/a; improved / degraded / unchanged: 0 / 0 / 0 (negative change = better)

### Verification

All stored aggregates match an independent recomputation from the items.

## score / original

- baseline: `results/measure/s1/s0raw_original.json`
- S1 letters: `results/measure/s1/s1raw_original.json`

| metric | S0 digits | S1 letters | Δ S1 letters |
| --- | ---: | ---: | ---: |
| labeled | 12 | 12 | +0 |
| accuracy | 0.7500 | 0.7500 | +0.0000 |
| MAE | 0.4167 | 0.3333 | -0.0833 ▲ |
| expected-score MAE | 0.3766 | 0.3607 | -0.0159 ▲ |
| expected-score MAE (raw, T=1) | 0.3766 | 0.3607 | -0.0159 ▲ |
| QWK | 0.7327 | 0.8333 | +0.1007 ▲ |
| NLL | 0.4308 | 0.7618 | +0.3310 ▼ |
| Brier | 0.2370 | 0.4321 | +0.1951 ▼ |
| ECE | 0.0923 | 0.2730 | +0.1806 ▼ |
| NLL (raw, T=1) | 0.4308 | 0.7618 | +0.3310 ▼ |
| Brier (raw, T=1) | 0.2370 | 0.4321 | +0.1951 ▼ |
| ECE (raw, T=1) | 0.0923 | 0.2730 | +0.1806 ▼ |
| mean signed margin | 1.9534 | 1.1254 | -0.8280 ▼ |
| median signed margin | 1.9390 | 0.7400 | -1.1990 ▼ |
| p10 signed margin | -0.0761 | -1.7207 | -1.6447 ▼ |
| min signed margin | -0.5435 | -2.2664 | -1.7229 ▼ |
| mean expected-argmax distance | 0.4135 | 0.3785 | -0.0349 ▲ |
| mean expected-argmax distance (raw) | 0.4135 | 0.3785 | -0.0349 ▲ |
| error distance 0 | 9 | 9 | +0 |
| error distance 1 | 1 | 2 | +1 |
| error distance 2 | 2 | 1 | -1 |
| large error rate | 0.1667 | 0.0833 | -0.0833 ▲ |
| eval_ms | 8582 | 8749 | +167 ▼ |
| prompt tokens | 1514 | 1466 | -48 ▲ |
| evaluations per item | 1.00 | 1.00 | +0.0000 |

▲ better than baseline, ▼ worse.

Large error: |predicted - ground truth| >= 2. No calibration artifact was applied (T = 1), so calibrated and raw metrics coincide. Brier is summed over levels.

**Candidates** — answer anchor `"<|im_end|>\n<|im_start|>assistant\n\n"`

| run | levels | labels | token ids | single | unique | separate after anchor | content-free prior logits | token forms (first label) |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| S0 digits | 4 | 0 1 2 3 | 15 16 17 18 | yes | yes | yes | -1.563 -3.179 -3.272 -5.240 | 0: '0'=[15] ' 0'=[220, 15] |
| S1 letters | 4 | A B C D | 32 33 34 35 | yes | yes | yes | 2.170 -0.443 0.402 -1.067 | A: 'A'=[32] ' A'=[362] |

**Selection frequency vs ground truth** (counts by semantic level; label in backticks)

| level | ground truth | S0 digits selected | S1 letters selected |
| --- | ---: | ---: | ---: |
| 0 | 4 | 7 (`0`) | 5 (`A`) |
| 1 | 2 | 1 (`1`) | 0 (`B`) |
| 2 | 4 | 2 (`2`) | 3 (`C`) |
| 3 | 2 | 2 (`3`) | 4 (`D`) |

### S1 letters vs S0 digits

- matched items: 12; margin gain mean -0.8280, median -0.7388
- margin improved / degraded / unchanged: 4 / 8 / 0
- S0 digits correct → S1 letters wrong: 1 (original-ordinal-02-0)
- S0 digits wrong → S1 letters correct: 1 (original-ordinal-03-0)
- quality metrics better: 4, worse: 10 (worse: nll, brier, ece, raw.nll, raw.brier, raw.ece, mean_signed_margin, median_signed_margin, p10_signed_margin, min_signed_margin)
- latency: 8749 ms vs 8582 ms (×1.02)
- expected-score |error| change (reported): mean -0.0159, median 0.0271; improved / degraded / unchanged: 5 / 7 / 0 (negative change = better)
- expected-score |error| change (raw): mean -0.0159, median 0.0271; improved / degraded / unchanged: 5 / 7 / 0 (negative change = better)

**Largest margin losses (top 5)**

| source_id | S0 digits | S1 letters | gain |
| --- | ---: | ---: | ---: |
| `original-ordinal-02-0` | 2.9643 | -0.7628 | -3.7270 |
| `original-ordinal-05-0` | 2.4362 | 0.4933 | -1.9429 |
| `original-ordinal-02-1` | -0.0229 | -1.8272 | -1.8043 |
| `original-ordinal-03-1` | -0.5435 | -2.2664 | -1.7229 |
| `original-ordinal-04-0` | 5.2135 | 3.5049 | -1.7086 |

**Largest margin gains (top 5)**

| source_id | S0 digits | S1 letters | gain |
| --- | ---: | ---: | ---: |
| `original-ordinal-01-1` | 1.4418 | 2.7061 | +1.2643 |
| `original-ordinal-06-0` | 0.2032 | 0.9868 | +0.7836 |
| `original-ordinal-03-0` | -0.0820 | 0.3348 | +0.4168 |
| `original-ordinal-01-0` | 3.0417 | 3.2285 | +0.1868 |

### Verification

All stored aggregates match an independent recomputation from the items.

## score / hard

- baseline: `results/measure/s1/s0raw_hard.json`
- S1 letters: `results/measure/s1/s1raw_hard.json`

| metric | S0 digits | S1 letters | Δ S1 letters |
| --- | ---: | ---: | ---: |
| labeled | 6 | 6 | +0 |
| accuracy | 0.0000 | 0.3333 | +0.3333 ▲ |
| MAE | 1.8333 | 1.1667 | -0.6667 ▲ |
| expected-score MAE | 1.0790 | 0.6271 | -0.4519 ▲ |
| expected-score MAE (raw, T=1) | 1.0790 | 0.6271 | -0.4519 ▲ |
| QWK | -0.2105 | 0.0930 | +0.3035 ▲ |
| NLL | 2.2618 | 1.8047 | -0.4571 ▲ |
| Brier | 1.1933 | 0.9204 | -0.2730 ▲ |
| ECE | 0.6156 | 0.2205 | -0.3951 ▲ |
| NLL (raw, T=1) | 2.2618 | 1.8047 | -0.4571 ▲ |
| Brier (raw, T=1) | 1.1933 | 0.9204 | -0.2730 ▲ |
| ECE (raw, T=1) | 0.6156 | 0.2205 | -0.3951 ▲ |
| mean signed margin | -1.7618 | -0.9737 | +0.7881 ▲ |
| median signed margin | -1.6863 | -1.2014 | +0.4849 ▲ |
| p10 signed margin | -2.6227 | -2.0949 | +0.5278 ▲ |
| min signed margin | -3.4033 | -2.1633 | +1.2400 ▲ |
| mean expected-argmax distance | 0.7544 | 0.9426 | +0.1882 ▼ |
| mean expected-argmax distance (raw) | 0.7544 | 0.9426 | +0.1882 ▼ |
| error distance 0 | 0 | 2 | +2 |
| error distance 1 | 2 | 1 | -1 |
| error distance 2 | 3 | 3 | +0 |
| error distance 3 | 1 | 0 | -1 |
| large error rate | 0.6667 | 0.5000 | -0.1667 ▲ |
| eval_ms | 70151 | 67924 | -2227 ▲ |
| prompt tokens | 10078 | 10053 | -25 ▲ |
| evaluations per item | 1.00 | 1.00 | +0.0000 |

▲ better than baseline, ▼ worse.

Large error: |predicted - ground truth| >= 2. No calibration artifact was applied (T = 1), so calibrated and raw metrics coincide. Brier is summed over levels.

**Candidates** — answer anchor `"<|im_end|>\n<|im_start|>assistant\n\n"`

| run | levels | labels | token ids | single | unique | separate after anchor | content-free prior logits | token forms (first label) |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| S0 digits | 4 | 0 1 2 3 | 15 16 17 18 | yes | yes | yes | -1.563 -3.179 -3.272 -5.240 | 0: '0'=[15] ' 0'=[220, 15] |
| S0 digits | 5 | 0 1 2 3 4 | 15 16 17 18 19 | yes | yes | yes | -0.093 -2.148 -2.577 -4.448 -2.605 | 0: '0'=[15] ' 0'=[220, 15] |
| S1 letters | 4 | A B C D | 32 33 34 35 | yes | yes | yes | 2.170 -0.443 0.402 -1.067 | A: 'A'=[32] ' A'=[362] |
| S1 letters | 5 | A B C D E | 32 33 34 35 36 | yes | yes | yes | 2.174 -0.407 0.518 -1.395 -4.040 | A: 'A'=[32] ' A'=[362] |

**Selection frequency vs ground truth** (counts by semantic level; label in backticks)

| level | ground truth | S0 digits selected | S1 letters selected |
| --- | ---: | ---: | ---: |
| 0 | 1 | 2 (`0`) | 3 (`A`) |
| 1 | 3 | 0 (`1`) | 0 (`B`) |
| 2 | 2 | 0 (`2`) | 1 (`C`) |
| 3 | 0 | 4 (`3`) | 2 (`D`) |

### S1 letters vs S0 digits

- matched items: 6; margin gain mean 0.7881, median 0.3854
- margin improved / degraded / unchanged: 4 / 2 / 0
- S0 digits correct → S1 letters wrong: 0
- S0 digits wrong → S1 letters correct: 2 (hard-sol-a-adversarial-09, hard-sol-a-trap-05)
- quality metrics better: 15, worse: 0
- latency: 67924 ms vs 70151 ms (×0.97)
- expected-score |error| change (reported): mean -0.4519, median -0.4804; improved / degraded / unchanged: 5 / 1 / 0 (negative change = better)
- expected-score |error| change (raw): mean -0.4519, median -0.4804; improved / degraded / unchanged: 5 / 1 / 0 (negative change = better)

**Largest margin losses (top 5)**

| source_id | S0 digits | S1 letters | gain |
| --- | ---: | ---: | ---: |
| `hard-opus-a-temporal_numeric-12` | -1.5940 | -2.0266 | -0.4326 |
| `hard-opus-c-long_policy-05` | -1.7587 | -2.1633 | -0.4046 |

**Largest margin gains (top 5)**

| source_id | S0 digits | S1 letters | gain |
| --- | ---: | ---: | ---: |
| `hard-opus-b-multi_hop-05` | -3.4033 | -0.8700 | +2.5333 |
| `hard-sol-a-trap-05` | -1.6139 | 0.6477 | +2.2615 |
| `hard-sol-a-adversarial-09` | -0.3586 | 0.1029 | +0.4615 |
| `hard-opus-c-long_policy-08` | -1.8422 | -1.5329 | +0.3093 |

### Verification

All stored aggregates match an independent recomputation from the items.

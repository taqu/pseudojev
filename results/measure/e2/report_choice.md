# choice: baseline vs E2, E2+KV

## choice / easy

- baseline: `results/measure/e2/base_easy.json`
- E2: `results/measure/e2/e2_easy.json`
- E2+KV: `results/measure/e2/e2kv_easy.json`

| metric | baseline | E2 | Δ E2 | E2+KV | Δ E2+KV |
| --- | ---: | ---: | ---: | ---: | ---: |
| labeled | 36 | 36 | +0 | 36 | +0 |
| accuracy | 1.0000 | 1.0000 | +0.0000 | 1.0000 | +0.0000 |
| NLL | 0.2163 | 0.1861 | -0.0302 ▲ | 0.1868 | -0.0295 ▲ |
| Brier | 0.0631 | 0.0470 | -0.0160 ▲ | 0.0475 | -0.0156 ▲ |
| ECE | 0.1876 | 0.1636 | -0.0241 ▲ | 0.1639 | -0.0238 ▲ |
| NLL (raw, T=1) | 0.0386 | 0.0261 | -0.0125 ▲ | 0.0270 | -0.0116 ▲ |
| Brier (raw, T=1) | 0.0126 | 0.0069 | -0.0057 ▲ | 0.0077 | -0.0049 ▲ |
| ECE (raw, T=1) | 0.0339 | 0.0231 | -0.0108 ▲ | 0.0237 | -0.0102 ▲ |
| mean winner margin | 4.7261 | 5.5771 | +0.8510 ▲ | 5.5703 | +0.8442 ▲ |
| median winner margin | 4.9637 | 5.9405 | +0.9769 ▲ | 5.9238 | +0.9602 ▲ |
| p10 winner margin | 3.1016 | 3.6874 | +0.5857 ▲ | 3.6918 | +0.5901 ▲ |
| min winner margin | 0.2654 | 0.8402 | +0.5747 ▲ | 0.7296 | +0.4642 ▲ |
| rotation disagreement rate | n/a | 0.0278 |  | 0.0278 |  |
| mean distinct winners | n/a | 1.0278 |  | 1.0556 |  |
| mean semantic variance | n/a | 1.7980 |  | 1.7966 |  |
| max semantic variance | n/a | 4.7439 |  | 4.7439 |  |
| evaluations per item | n/a | 4.75 |  | 4.75 |  |
| prompt tokens | 3639 | 17467 | +13828 ▼ | 17467 | +13828 ▼ |
| eval_ms | 19429 | 93389 | +73960 ▼ | 61077 | +41648 ▼ |

▲ better than baseline, ▼ worse.

### E2 vs baseline

- matched items: 36; margin gain mean 0.8510, median 1.1030
- margin improved / degraded / unchanged: 26 / 10 / 0
- baseline correct → E2 wrong: 0
- baseline wrong → E2 correct: 0
- quality metrics better: 10, worse: 0
- latency: 93389 ms vs 19429 ms (×4.81)

**Largest margin losses (top 5)**

| source_id | baseline | E2 | gain |
| --- | ---: | ---: | ---: |
| `easy-intent-06` | 7.4391 | 5.9712 | -1.4679 |
| `easy-extraction-06` | 2.1854 | 0.8402 | -1.3452 |
| `easy-intent-00` | 6.8779 | 5.8761 | -1.0018 |
| `easy-extraction-04` | 7.3615 | 6.4092 | -0.9523 |
| `easy-intent-05` | 5.8300 | 5.4559 | -0.3741 |

**Largest margin gains (top 5)**

| source_id | baseline | E2 | gain |
| --- | ---: | ---: | ---: |
| `easy-extraction-01` | 2.8662 | 5.3865 | +2.5203 |
| `easy-tool_selection-09` | 5.0976 | 7.6113 | +2.5137 |
| `easy-tool_selection-03` | 4.6953 | 7.0247 | +2.3294 |
| `easy-extraction-05` | 0.2654 | 2.4719 | +2.2064 |
| `easy-tool_selection-04` | 4.1955 | 6.2526 | +2.0571 |

### E2+KV vs baseline

- matched items: 36; margin gain mean 0.8442, median 1.0562
- margin improved / degraded / unchanged: 26 / 10 / 0
- baseline correct → E2+KV wrong: 0
- baseline wrong → E2+KV correct: 0
- quality metrics better: 10, worse: 0
- latency: 61077 ms vs 19429 ms (×3.14)

**Largest margin losses (top 5)**

| source_id | baseline | E2+KV | gain |
| --- | ---: | ---: | ---: |
| `easy-intent-06` | 7.4391 | 5.9712 | -1.4679 |
| `easy-extraction-06` | 2.1854 | 0.7296 | -1.4558 |
| `easy-intent-00` | 6.8779 | 5.8761 | -1.0018 |
| `easy-extraction-04` | 7.3615 | 6.4299 | -0.9316 |
| `easy-intent-05` | 5.8300 | 5.4559 | -0.3741 |

**Largest margin gains (top 5)**

| source_id | baseline | E2+KV | gain |
| --- | ---: | ---: | ---: |
| `easy-extraction-01` | 2.8662 | 5.4179 | +2.5516 |
| `easy-tool_selection-09` | 5.0976 | 7.6113 | +2.5137 |
| `easy-tool_selection-03` | 4.6953 | 7.0247 | +2.3294 |
| `easy-extraction-05` | 0.2654 | 2.5198 | +2.2544 |
| `easy-tool_selection-04` | 4.1955 | 6.2526 | +2.0571 |

### Equivalence: E2 vs E2+KV

- common items: 36
- prediction mismatches: 0
- max |Δ logit|: 0.234
- max |Δ prob|: 0.00963

### Verification

All stored aggregates match an independent recomputation from the items.

## choice / original

- baseline: `results/measure/e2/base_original.json`
- E2: `results/measure/e2/e2_original.json`
- E2+KV: `results/measure/e2/e2kv_original.json`

| metric | baseline | E2 | Δ E2 | E2+KV | Δ E2+KV |
| --- | ---: | ---: | ---: | ---: | ---: |
| labeled | 36 | 36 | +0 | 36 | +0 |
| accuracy | 0.6944 | 0.6944 | +0.0000 | 0.6944 | +0.0000 |
| NLL | 0.9760 | 0.8543 | -0.1217 ▲ | 0.8544 | -0.1216 ▲ |
| Brier | 0.4717 | 0.4295 | -0.0422 ▲ | 0.4298 | -0.0420 ▲ |
| ECE | 0.1961 | 0.2526 | +0.0565 ▼ | 0.2670 | +0.0709 ▼ |
| NLL (raw, T=1) | 1.0101 | 0.7634 | -0.2467 ▲ | 0.7645 | -0.2456 ▲ |
| Brier (raw, T=1) | 0.4285 | 0.4128 | -0.0158 ▲ | 0.4137 | -0.0148 ▲ |
| ECE (raw, T=1) | 0.1622 | 0.1266 | -0.0356 ▲ | 0.1272 | -0.0351 ▲ |
| mean winner margin | 0.7565 | 1.2658 | +0.5093 ▲ | 1.2635 | +0.5070 ▲ |
| median winner margin | 1.3468 | 1.6528 | +0.3060 ▲ | 1.6528 | +0.3060 ▲ |
| p10 winner margin | -2.5172 | -2.2038 | +0.3134 ▲ | -2.1933 | +0.3239 ▲ |
| min winner margin | -5.8441 | -2.9869 | +2.8572 ▲ | -2.9869 | +2.8572 ▲ |
| rotation disagreement rate | n/a | 0.4167 |  | 0.4444 |  |
| mean distinct winners | n/a | 1.6111 |  | 1.6389 |  |
| mean semantic variance | n/a | 1.4034 |  | 1.3977 |  |
| max semantic variance | n/a | 5.2703 |  | 4.8558 |  |
| evaluations per item | n/a | 5.00 |  | 5.00 |  |
| prompt tokens | 3991 | 20377 | +16386 ▼ | 20377 | +16386 ▼ |
| eval_ms | 20747 | 105738 | +84991 ▼ | 64205 | +43458 ▼ |

▲ better than baseline, ▼ worse.

### E2 vs baseline

- matched items: 36; margin gain mean 0.5093, median 0.5578
- margin improved / degraded / unchanged: 28 / 8 / 0
- baseline correct → E2 wrong: 3 (original-intent-02-0, original-routing-06-0, original-routing-06-1)
- baseline wrong → E2 correct: 3 (original-intent-03-0, original-extraction-04-1, original-routing-02-0)
- quality metrics better: 9, worse: 1 (worse: ece)
- latency: 105738 ms vs 20747 ms (×5.10)

**Largest margin losses (top 5)**

| source_id | baseline | E2 | gain |
| --- | ---: | ---: | ---: |
| `original-routing-06-0` | 1.9617 | -0.8952 | -2.8569 |
| `original-routing-06-1` | 1.0543 | -0.9174 | -1.9717 |
| `original-intent-06-1` | 3.7734 | 2.2029 | -1.5705 |
| `original-intent-06-0` | 2.1862 | 0.6219 | -1.5643 |
| `original-intent-02-0` | 1.2491 | -0.3144 | -1.5635 |

**Largest margin gains (top 5)**

| source_id | baseline | E2 | gain |
| --- | ---: | ---: | ---: |
| `original-routing-03-0` | 1.6487 | 4.6754 | +3.0266 |
| `original-routing-04-0` | -5.8441 | -2.9869 | +2.8572 |
| `original-routing-03-1` | 2.6470 | 5.3743 | +2.7273 |
| `original-routing-02-0` | -0.9188 | 1.6808 | +2.5996 |
| `original-routing-04-1` | -4.7962 | -2.3066 | +2.4896 |

### E2+KV vs baseline

- matched items: 36; margin gain mean 0.5070, median 0.5364
- margin improved / degraded / unchanged: 28 / 8 / 0
- baseline correct → E2+KV wrong: 3 (original-intent-02-0, original-routing-06-0, original-routing-06-1)
- baseline wrong → E2+KV correct: 3 (original-intent-03-0, original-extraction-04-1, original-routing-02-0)
- quality metrics better: 9, worse: 1 (worse: ece)
- latency: 64205 ms vs 20747 ms (×3.09)

**Largest margin losses (top 5)**

| source_id | baseline | E2+KV | gain |
| --- | ---: | ---: | ---: |
| `original-routing-06-0` | 1.9617 | -0.8952 | -2.8569 |
| `original-routing-06-1` | 1.0543 | -0.9174 | -1.9717 |
| `original-intent-02-0` | 1.2491 | -0.3843 | -1.6334 |
| `original-intent-06-1` | 3.7734 | 2.2129 | -1.5605 |
| `original-intent-06-0` | 2.1862 | 0.6893 | -1.4969 |

**Largest margin gains (top 5)**

| source_id | baseline | E2+KV | gain |
| --- | ---: | ---: | ---: |
| `original-routing-03-0` | 1.6487 | 4.6754 | +3.0266 |
| `original-routing-04-0` | -5.8441 | -2.9869 | +2.8572 |
| `original-routing-03-1` | 2.6470 | 5.3743 | +2.7273 |
| `original-routing-02-0` | -0.9188 | 1.6808 | +2.5996 |
| `original-routing-04-1` | -4.7962 | -2.3066 | +2.4896 |

### Equivalence: E2 vs E2+KV

- common items: 36
- prediction mismatches: 0
- max |Δ logit|: 0.238
- max |Δ prob|: 0.00765

### Verification

All stored aggregates match an independent recomputation from the items.

## choice / hard

- baseline: `results/measure/e2/base_hard.json`
- E2: `results/measure/e2/e2_hard.json`
- E2+KV: `results/measure/e2/e2kv_hard.json`

| metric | baseline | E2 | Δ E2 | E2+KV | Δ E2+KV |
| --- | ---: | ---: | ---: | ---: | ---: |
| labeled | 67 | 67 | +0 | 67 | +0 |
| accuracy | 0.2388 | 0.3731 | +0.1343 ▲ | 0.3731 | +0.1343 ▲ |
| NLL | 1.5608 | 1.3155 | -0.2453 ▲ | 1.3164 | -0.2444 ▲ |
| Brier | 0.8272 | 0.7152 | -0.1120 ▲ | 0.7160 | -0.1112 ▲ |
| ECE | 0.2443 | 0.1389 | -0.1054 ▲ | 0.1388 | -0.1055 ▲ |
| NLL (raw, T=1) | 2.0889 | 1.4529 | -0.6360 ▲ | 1.4554 | -0.6335 ▲ |
| Brier (raw, T=1) | 1.0058 | 0.7888 | -0.2171 ▲ | 0.7904 | -0.2154 ▲ |
| ECE (raw, T=1) | 0.4268 | 0.2744 | -0.1524 ▲ | 0.2659 | -0.1609 ▲ |
| mean winner margin | -1.3135 | -0.4023 | +0.9112 ▲ | -0.4067 | +0.9068 ▲ |
| median winner margin | -1.6307 | -0.2531 | +1.3776 ▲ | -0.2803 | +1.3505 ▲ |
| p10 winner margin | -3.9621 | -2.6081 | +1.3540 ▲ | -2.6081 | +1.3540 ▲ |
| min winner margin | -4.5869 | -3.8661 | +0.7209 ▲ | -3.8661 | +0.7209 ▲ |
| rotation disagreement rate | n/a | 0.7910 |  | 0.7910 |  |
| mean distinct winners | n/a | 2.4627 |  | 2.4776 |  |
| mean semantic variance | n/a | 1.3914 |  | 1.3999 |  |
| max semantic variance | n/a | 5.1757 |  | 5.1977 |  |
| evaluations per item | n/a | 4.13 |  | 4.13 |  |
| prompt tokens | 73989 | 322582 | +248593 ▼ | 322582 | +248593 ▼ |
| eval_ms | 436914 | 2500687 | +2063773 ▼ | 743689 | +306775 ▼ |

▲ better than baseline, ▼ worse.

### E2 vs baseline

- matched items: 67; margin gain mean 0.9112, median 0.9223
- margin improved / degraded / unchanged: 51 / 16 / 0
- baseline correct → E2 wrong: 2 (hard-opus-c-temporal_numeric-03, hard-sol-b-long_policy-01)
- baseline wrong → E2 correct: 11 (hard-opus-a-long_policy-01, hard-opus-a-long_policy-17, hard-opus-b-ambiguous-03, hard-opus-b-probability-04, hard-opus-b-probability-06, hard-opus-b-tradeoff-03, hard-sol-b-long_policy-06, hard-sol-b-routing_hard-03, hard-sol-b-routing_hard-09, hard-sol-b-temporal_numeric-03, hard-sol-c-multi_hop-07)
- quality metrics better: 11, worse: 0
- latency: 2500687 ms vs 436914 ms (×5.72)

**Largest margin losses (top 5)**

| source_id | baseline | E2 | gain |
| --- | ---: | ---: | ---: |
| `hard-opus-c-temporal_numeric-03` | 2.7612 | -0.5255 | -3.2866 |
| `hard-sol-a-trap-10` | 2.7954 | 0.3371 | -2.4583 |
| `hard-opus-c-temporal_numeric-12` | -0.2279 | -2.6014 | -2.3735 |
| `hard-sol-b-long_policy-01` | 1.3360 | -0.5591 | -1.8951 |
| `hard-sol-b-temporal_numeric-01` | -1.6307 | -3.0916 | -1.4609 |

**Largest margin gains (top 5)**

| source_id | baseline | E2 | gain |
| --- | ---: | ---: | ---: |
| `hard-opus-c-temporal_numeric-04` | -4.5869 | -0.1854 | +4.4015 |
| `hard-sol-b-routing_hard-03` | -2.0794 | 2.0156 | +4.0949 |
| `hard-opus-c-temporal_numeric-08` | -3.9166 | -0.0500 | +3.8666 |
| `hard-opus-c-temporal_numeric-06` | -4.3744 | -0.9157 | +3.4587 |
| `hard-sol-b-routing_hard-09` | -3.4116 | 0.0194 | +3.4310 |

### E2+KV vs baseline

- matched items: 67; margin gain mean 0.9068, median 0.9666
- margin improved / degraded / unchanged: 51 / 16 / 0
- baseline correct → E2+KV wrong: 2 (hard-opus-c-temporal_numeric-03, hard-sol-b-long_policy-01)
- baseline wrong → E2+KV correct: 11 (hard-opus-a-long_policy-01, hard-opus-a-long_policy-17, hard-opus-b-ambiguous-03, hard-opus-b-probability-04, hard-opus-b-probability-06, hard-opus-b-tradeoff-03, hard-sol-b-long_policy-06, hard-sol-b-routing_hard-03, hard-sol-b-routing_hard-09, hard-sol-b-temporal_numeric-03, hard-sol-c-multi_hop-07)
- quality metrics better: 11, worse: 0
- latency: 743689 ms vs 436914 ms (×1.70)

**Largest margin losses (top 5)**

| source_id | baseline | E2+KV | gain |
| --- | ---: | ---: | ---: |
| `hard-opus-c-temporal_numeric-03` | 2.7612 | -0.5255 | -3.2866 |
| `hard-sol-a-trap-10` | 2.7954 | 0.3716 | -2.4238 |
| `hard-opus-c-temporal_numeric-12` | -0.2279 | -2.6014 | -2.3735 |
| `hard-sol-b-long_policy-01` | 1.3360 | -0.5591 | -1.8951 |
| `hard-sol-b-temporal_numeric-01` | -1.6307 | -3.0916 | -1.4609 |

**Largest margin gains (top 5)**

| source_id | baseline | E2+KV | gain |
| --- | ---: | ---: | ---: |
| `hard-opus-c-temporal_numeric-04` | -4.5869 | -0.1854 | +4.4015 |
| `hard-sol-b-routing_hard-03` | -2.0794 | 2.0156 | +4.0949 |
| `hard-opus-c-temporal_numeric-08` | -3.9166 | -0.0500 | +3.8666 |
| `hard-opus-c-temporal_numeric-06` | -4.3744 | -0.9157 | +3.4587 |
| `hard-sol-b-routing_hard-09` | -3.4116 | 0.0194 | +3.4310 |

### Equivalence: E2 vs E2+KV

- common items: 67
- prediction mismatches: 0
- max |Δ logit|: 0.399
- max |Δ prob|: 0.015

### Verification

All stored aggregates match an independent recomputation from the items.

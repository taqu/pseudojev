# noul: baseline vs E1

## noul / easy

- baseline: `results/measure/base_easy.json`
- E1: `results/measure/e1_easy.json`

| metric | baseline | E1 | Δ E1 |
| --- | ---: | ---: | ---: |
| labeled | 12 | 12 | +0 |
| accuracy | 1.0000 | 1.0000 | +0.0000 |
| NLL | 0.6673 | 0.6698 | +0.0025 ▼ |
| Brier | 0.2371 | 0.2383 | +0.0012 ▼ |
| ECE | 0.4869 | 0.4882 | +0.0013 ▼ |
| NLL (raw, T=1) | 0.0188 | 0.0378 | +0.0190 ▼ |
| Brier (raw, T=1) | 0.0007 | 0.0032 | +0.0025 ▼ |
| ECE (raw, T=1) | 0.0184 | 0.0361 | +0.0176 ▼ |
| mean conf. correct (raw) | 0.9816 | 0.9639 | -0.0176 ▼ |
| mean signed margin | 5.2393 | 4.7362 | -0.5031 ▼ |
| median signed margin | 4.3495 | 4.1116 | -0.2379 ▼ |
| p10 signed margin | 3.0996 | 2.1160 | -0.9836 ▼ |
| min signed margin | 2.7732 | 1.8931 | -0.8802 ▼ |
| order disagreement rate | n/a | 0.0000 |  |
| eval_ms | 6029 | 11906 | +5877 ▼ |

▲ better than baseline, ▼ worse.

### E1 vs baseline

- matched items: 12; margin gain mean -0.5031, median -0.4576
- margin improved / degraded / unchanged: 4 / 8 / 0
- baseline correct → E1 wrong: 0
- baseline wrong → E1 correct: 0
- quality metrics better: 0, worse: 10 (worse: nll, brier, ece, raw.nll, raw.brier, raw.ece, mean_signed_margin, median_signed_margin, p10_signed_margin, min_signed_margin)
- latency: 11906 ms vs 6029 ms (×1.97)

**Largest margin losses (top 5)**

| source_id | baseline | E1 | gain |
| --- | ---: | ---: | ---: |
| `easy-fact-07` | 3.5089 | 1.8931 | -1.6158 |
| `easy-fact-09` | 3.6599 | 2.0655 | -1.5944 |
| `easy-fact-00` | 7.5985 | 6.4900 | -1.1085 |
| `easy-fact-11` | 4.1858 | 3.3451 | -0.8408 |
| `easy-fact-05` | 3.0772 | 2.5703 | -0.5069 |

**Largest margin gains (top 5)**

| source_id | baseline | E1 | gain |
| --- | ---: | ---: | ---: |
| `easy-fact-01` | 2.7732 | 3.1131 | +0.3399 |
| `easy-fact-03` | 3.3009 | 3.5524 | +0.2516 |
| `easy-fact-04` | 4.5131 | 4.6708 | +0.1577 |
| `easy-fact-08` | 7.3376 | 7.4190 | +0.0814 |

### Verification

All stored aggregates match an independent recomputation from the items.

## noul / original

- baseline: `results/measure/base_original.json`
- E1: `results/measure/e1_original.json`

| metric | baseline | E1 | Δ E1 |
| --- | ---: | ---: | ---: |
| labeled | 24 | 24 | +0 |
| accuracy | 0.5000 | 0.5417 | +0.0417 ▲ |
| NLL | 0.6920 | 0.6923 | +0.0004 ▼ |
| Brier | 0.2494 | 0.2496 | +0.0002 ▼ |
| ECE | 0.0042 | 0.0371 | +0.0329 ▼ |
| NLL (raw, T=1) | 0.9602 | 1.0491 | +0.0889 ▼ |
| Brier (raw, T=1) | 0.3199 | 0.3455 | +0.0255 ▼ |
| ECE (raw, T=1) | 0.3467 | 0.2838 | -0.0629 ▲ |
| mean conf. correct (raw) | 0.8279 | 0.8116 | -0.0163 ▼ |
| mean signed margin | 0.2438 | 0.1723 | -0.0715 ▼ |
| median signed margin | 0.2543 | 0.3155 | +0.0613 ▲ |
| p10 signed margin | -2.3228 | -2.1833 | +0.1396 ▲ |
| min signed margin | -3.3343 | -3.7766 | -0.4423 ▼ |
| order disagreement rate | n/a | 0.2500 |  |
| eval_ms | 13496 | 24584 | +11088 ▼ |

▲ better than baseline, ▼ worse.

### E1 vs baseline

- matched items: 24; margin gain mean -0.0715, median 0.0042
- margin improved / degraded / unchanged: 12 / 12 / 0
- baseline correct → E1 wrong: 1 (original-adequacy-04-1)
- baseline wrong → E1 correct: 2 (original-policy-04-0, original-adequacy-02-1)
- quality metrics better: 4, worse: 7 (worse: nll, brier, ece, raw.nll, raw.brier, mean_signed_margin, min_signed_margin)
- latency: 24584 ms vs 13496 ms (×1.82)

**Largest margin losses (top 5)**

| source_id | baseline | E1 | gain |
| --- | ---: | ---: | ---: |
| `original-policy-03-0` | -0.4655 | -2.1950 | -1.7294 |
| `original-adequacy-06-0` | 2.8254 | 1.3437 | -1.4817 |
| `original-adequacy-06-1` | 3.2419 | 1.9120 | -1.3298 |
| `original-policy-06-0` | 1.4010 | 0.1965 | -1.2045 |
| `original-adequacy-04-1` | 0.7193 | -0.3692 | -1.0884 |

**Largest margin gains (top 5)**

| source_id | baseline | E1 | gain |
| --- | ---: | ---: | ---: |
| `original-policy-04-0` | -0.4969 | 1.6726 | +2.1695 |
| `original-adequacy-02-1` | -0.0680 | 1.3802 | +1.4482 |
| `original-policy-01-1` | 1.7476 | 2.8545 | +1.1069 |
| `original-policy-05-0` | 0.7094 | 1.6931 | +0.9837 |
| `original-policy-01-0` | 4.0728 | 4.9089 | +0.8361 |

### Verification

All stored aggregates match an independent recomputation from the items.

## noul / hard

- baseline: `results/measure/base_hard.json`
- E1: `results/measure/e1_hard.json`

| metric | baseline | E1 | Δ E1 |
| --- | ---: | ---: | ---: |
| labeled | 38 | 38 | +0 |
| accuracy | 0.3947 | 0.4474 | +0.0526 ▲ |
| NLL | 0.6953 | 0.6943 | -0.0009 ▲ |
| Brier | 0.2511 | 0.2506 | -0.0005 ▲ |
| ECE | 0.1085 | 0.0554 | -0.0532 ▲ |
| NLL (raw, T=1) | 1.1659 | 1.0081 | -0.1577 ▲ |
| Brier (raw, T=1) | 0.3951 | 0.3431 | -0.0520 ▲ |
| ECE (raw, T=1) | 0.3641 | 0.3316 | -0.0325 ▲ |
| mean conf. correct (raw) | 0.7366 | 0.7129 | -0.0237 ▼ |
| mean signed margin | -0.4152 | -0.2322 | +0.1830 ▲ |
| median signed margin | -0.5108 | -0.1982 | +0.3126 ▲ |
| p10 signed margin | -2.5972 | -1.9923 | +0.6048 ▲ |
| min signed margin | -4.3225 | -3.8171 | +0.5054 ▲ |
| order disagreement rate | n/a | 0.2105 |  |
| eval_ms | 142546 | 292868 | +150322 ▼ |

▲ better than baseline, ▼ worse.

### E1 vs baseline

- matched items: 38; margin gain mean 0.1830, median 0.1109
- margin improved / degraded / unchanged: 21 / 17 / 0
- baseline correct → E1 wrong: 1 (hard-opus-a-temporal_numeric-03)
- baseline wrong → E1 correct: 3 (hard-opus-a-probability-03, hard-opus-b-probability-03, hard-sol-a-adversarial-06)
- quality metrics better: 11, worse: 0
- latency: 292868 ms vs 142546 ms (×2.05)

**Largest margin losses (top 5)**

| source_id | baseline | E1 | gain |
| --- | ---: | ---: | ---: |
| `hard-opus-c-long_policy-11` | -2.2608 | -3.7504 | -1.4895 |
| `hard-opus-a-temporal_numeric-03` | 0.8652 | -0.3352 | -1.2004 |
| `hard-opus-b-multi_hop-08` | -2.7171 | -3.8171 | -1.1000 |
| `hard-opus-b-tradeoff-08` | 1.1670 | 0.2274 | -0.9396 |
| `hard-opus-a-long_policy-13` | 2.3039 | 1.5914 | -0.7125 |

**Largest margin gains (top 5)**

| source_id | baseline | E1 | gain |
| --- | ---: | ---: | ---: |
| `hard-opus-a-temporal_numeric-09` | -2.7850 | -0.1043 | +2.6807 |
| `hard-sol-a-trap-04` | -4.3225 | -2.0268 | +2.2957 |
| `hard-opus-a-probability-03` | -0.2782 | 1.7068 | +1.9850 |
| `hard-opus-b-probability-03` | -1.3196 | 0.5599 | +1.8795 |
| `hard-sol-a-adversarial-06` | -0.0985 | 1.1871 | +1.2857 |

### Verification

All stored aggregates match an independent recomputation from the items.
